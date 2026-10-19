#include "strata/kernels/mtp_fp8.hpp"
#include <cuda_runtime.h>

namespace strata::kernels {
namespace {
__device__ float e4m3(uint8_t v) {
    const unsigned e=(v>>3)&15, m=v&7;
    const float a=e ? __uint_as_float(((e+120)<<23)|(m<<20)) : float(m)*(1.0f/512.0f);
    return v&128 ? -a : a;
}
__device__ float weight(const uint8_t* p,int row,int col,int cols) {
    const auto* s=reinterpret_cast<const uint16_t*>(p+kMtpFp8Matrix);
    return e4m3(p[row*cols+col])*__uint_as_float(unsigned(s[(row/128)*(cols/128)+col/128])<<16);
}
template<bool Nvfp4>
__device__ float expert_weight(const uint8_t* expert,int projection,int row,int col,int cols) {
    if constexpr (!Nvfp4) return weight(expert+projection*kMtpFp8Projection,row,col,cols);
    else {
        const auto index=uint64_t(row)*cols+col,block=index/64,sub=(index%64)/16,j=index%16;
        const auto* p=expert+projection*kMtpNvfp4Projection+block*36;
        const auto code=(p[4+sub*8+j%8]>>(4*(j/8)))&15;
        const float values[]={0,.5f,1,1.5f,2,3,4,6};
        const float value=(code&8)?-values[code&7]:values[code&7];
        const float global=reinterpret_cast<const float*>(expert+3*kMtpNvfp4Projection)[projection];
        return (value*e4m3(p[sub]))*global;
    }
}
__device__ float sum_warp(float x) {
    for(int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
    return x;
}
template<bool Nvfp4>
__global__ void gate_up(const uint8_t* experts,const int32_t* ids,const float* x,float* h,int k) {
    const int pair=blockIdx.y,row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=640) return;
    const uint8_t* p=experts+uint64_t(ids[pair])*(Nvfp4?kMtpNvfp4Expert:kMtpFp8Expert);
    const float* a=x+(pair/k)*2560;
    float g=0,u=0;
    for(int c=lane;c<2560;c+=32) {
        g=fmaf(expert_weight<Nvfp4>(p,0,row,c,2560),a[c],g);
        u=fmaf(expert_weight<Nvfp4>(p,1,row,c,2560),a[c],u);
    }
    g=sum_warp(g); u=sum_warp(u);
    if(lane==0) h[pair*640+row]=(g/(1.0f+expf(-g)))*u;
}
template<bool Nvfp4>
__global__ void down(const uint8_t* experts,const int32_t* ids,const float* h,float* y) {
    const int pair=blockIdx.y,row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x%32;
    const uint8_t* p=experts+uint64_t(ids[pair])*(Nvfp4?kMtpNvfp4Expert:kMtpFp8Expert);
    float v=0;
    for(int c=lane;c<640;c+=32) v=fmaf(expert_weight<Nvfp4>(p,2,row,c,640),h[pair*640+c],v);
    v=sum_warp(v);
    if(lane==0) y[pair*2560+row]=v;
}
}
void mtp_fp8_experts(const uint8_t* experts,const int32_t* ids,const float* x,
                     float* hidden,float* parts,int tokens,int topk,void* stream) {
    const auto cs=static_cast<cudaStream_t>(stream);
    gate_up<false><<<dim3(80,tokens*topk),256,0,cs>>>(experts,ids,x,hidden,topk);
    down<false><<<dim3(320,tokens*topk),256,0,cs>>>(experts,ids,hidden,parts);
}
void mtp_nvfp4_experts(const uint8_t* experts,const int32_t* ids,const float* x,
                     float* hidden,float* parts,int tokens,int topk,void* stream) {
    const auto cs=static_cast<cudaStream_t>(stream);
    gate_up<true><<<dim3(80,tokens*topk),256,0,cs>>>(experts,ids,x,hidden,topk);
    down<true><<<dim3(320,tokens*topk),256,0,cs>>>(experts,ids,hidden,parts);
}

}
