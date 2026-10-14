// SM120 block-scaled W4A4 MMA, adapted from local commit d167eb89.
// One calibrated activation term; no compensated/residual activation modes.
// MMA register mapping/PTX follows llama.cpp ggml-cuda/mma.cuh (MIT,
// third_party/ggml/LICENSE) and NVIDIA's warp-level block-scaling specification.
#include "strata/kernels/nvfp4_tensor.hpp"
#ifdef STRATA_NVFP4_CUTLASS_BUILT
#include "strata/kernels/nvfp4_cutlass.hpp"
#endif
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <string>
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

namespace strata::kernels {
namespace {
struct QBlock { unsigned char scale[4]; unsigned int q[8]; };
static_assert(sizeof(QBlock)==36);
size_t aligned(size_t n) { return (n+255)&~size_t(255); }
__device__ unsigned pack4(unsigned v) {
    return (v&15)|((v>>4)&240)|((v>>8)&3840)|((v>>12)&61440);
}
__device__ unsigned weight_word(const block_nvfp4* b, int word) {
    const int sub=word/2, shift=(word%2)*4;
    const unsigned* p=reinterpret_cast<const unsigned*>(b->qs+sub*8);
    return pack4(p[0]>>shift)|(pack4(p[1]>>shift)<<16);
}
__device__ float scale_value(unsigned char c) {
    __nv_fp8_e4m3 v; v.__x=c; return float(v);
}
__device__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2], unsigned sa, unsigned sb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
#endif
}

template<bool DOWN, int TERMS>
__global__ void quantize_rows(const float* x, QBlock* q, const int32_t* starts,
    const int32_t* ng, const int32_t* tok, int cols, int ff, float global_scale, float* row_scales) {
    if (*ng==0) return;
    const int entry=blockIdx.x;
    if (entry>=starts[*ng]) return;
    const int source=DOWN ? entry : tok[entry], blocks=cols/64;
    if(threadIdx.x==0) row_scales[entry]=global_scale;
    for (int sub=threadIdx.x; sub<cols/16; sub+=blockDim.x) {
        float v[16];
#pragma unroll
        for(int j=0;j<16;++j) {
            const int k=sub*16+j;
            float a;
            if constexpr(DOWN) {
                const float g=x[size_t(source)*2*ff+k], u=x[size_t(source)*2*ff+ff+k];
                a=(g/(1.0f+expf(-g)))*u;
            } else a=x[size_t(source)*cols+k];
            v[j]=__fdiv_rn(a,global_scale);
        }
#pragma unroll
        for(int term=0;term<TERMS;++term) {
            float amax=0;
#pragma unroll
            for(int j=0;j<16;++j) amax=fmaxf(amax,fabsf(v[j]));
            const unsigned char sc=__nv_cvt_float_to_fp8(__fdiv_rn(amax,6.0f),__NV_SATFINITE,__NV_E4M3);
            const float scale=scale_value(sc), inv=scale>0 ? __fdiv_rn(1.0f,scale) : 0.0f;
            unsigned words[2]={0,0};
#pragma unroll
            for(int j=0;j<16;++j) {
                const unsigned char code=__nv_cvt_float_to_fp4(v[j]*inv,__NV_E2M1,cudaRoundNearest);
                words[j/8]|=unsigned(code&15)<<(4*(j%8));
                if constexpr(TERMS>1) v[j]-=__half2float(__nv_cvt_fp4_to_halfraw(code,__NV_E2M1))*scale;
            }
            QBlock& dest=q[(size_t(entry)*TERMS+term)*blocks+sub/4];
            dest.scale[sub%4]=sc;
            dest.q[2*(sub%4)]=words[0]; dest.q[2*(sub%4)+1]=words[1];
        }
    }
}

template<bool DOWN,int TERMS>
__global__ void product(NativeExpertLayout L, const unsigned long long* ptr,
    const int32_t* starts,const int32_t* ng,const int32_t* dst,const QBlock* q,
    float* out,const float* row_scales) {
    const int lane=threadIdx.x, row0=(blockIdx.x*blockDim.y+threadIdx.y)*16;
    const int rows=DOWN ? L.n_embd : 2*L.n_ff, cols=DOWN ? L.n_ff : L.n_embd;
    if(row0>=rows) return;
    const int blocks=cols/64;
    for(int group=blockIdx.y;group<*ng;group+=gridDim.y) {
        const auto* blob=reinterpret_cast<const unsigned char*>(ptr[group]);
        const auto* weights=reinterpret_cast<const block_nvfp4*>(blob+(DOWN?L.down_off:0));
        const float* tails=reinterpret_cast<const float*>(blob+L.tail_off);
        for(int first=starts[group]+int(blockIdx.z)*8;first<starts[group+1];first+=int(gridDim.z)*8) {
            float acc[4]={0,0,0,0};
            for(int k=0;k<blocks;++k) {
                unsigned a[4];
#pragma unroll
                for(int z=0;z<4;++z) {
                    const int row=row0+(z%2)*8+lane/4;
                    a[z]=weight_word(weights+size_t(row)*blocks+k,lane%4+(z/2)*4);
                }
                const int scale_row=row0+lane/4+(lane%2)*8;
                const unsigned sa=*reinterpret_cast<const unsigned*>((weights+size_t(scale_row)*blocks+k)->d);
                const int entry=first+lane/4;
#pragma unroll
                for(int term=0;term<TERMS;++term) {
                    unsigned b[2]={0,0}, sb=0;
                    if(entry<starts[group+1]) {
                        const QBlock& qb=q[(size_t(entry)*TERMS+term)*blocks+k];
                        b[0]=qb.q[lane%4]; b[1]=qb.q[4+lane%4];
                        sb=*reinterpret_cast<const unsigned*>(qb.scale);
                    }
                    mma(acc,a,b,sa,sb);
                }
            }
#pragma unroll
            for(int z=0;z<4;++z) {
                const int row=row0+(z/2)*8+lane/4;
                const int entry=first+(lane%4)*2+z%2;
                if(entry<starts[group+1]) {
                    const float ws=DOWN ? tails[2] : tails[row>=L.n_ff ? 1 : 0];
                    const int output_entry=DOWN && dst ? dst[entry] : entry;
                    out[size_t(output_entry)*rows+row]=acc[z]*row_scales[entry]*ws;
                }
            }
        }
    }
}

template<int TERMS>
void launch(const NativeExpertLayout& L,const unsigned long long* ptr,const int32_t* starts,
    const int32_t* ng,const int32_t* dst,const int32_t* tok,int64_t groups,int64_t entries,
    const float* x,void* scratch,float* out,cudaStream_t stream,float gu_scale,float down_scale,int token_tiles) {
    auto* qx=static_cast<QBlock*>(scratch);
    const size_t qx_bytes=aligned(size_t(entries)*TERMS*(L.n_embd/64)*sizeof(QBlock));
    auto* gu=reinterpret_cast<float*>(static_cast<unsigned char*>(scratch)+qx_bytes);
    auto* qh=reinterpret_cast<QBlock*>(reinterpret_cast<unsigned char*>(gu)+aligned(size_t(entries)*2*L.n_ff*4));
    auto* gx=reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(qh)+aligned(size_t(entries)*TERMS*(L.n_ff/64)*sizeof(QBlock)));
    float* gh=gx+entries;
    quantize_rows<false,TERMS><<<unsigned(entries),128,0,stream>>>(x,qx,starts,ng,tok,int(L.n_embd),int(L.n_ff),gu_scale,gx);
    product<false,TERMS><<<dim3(unsigned((2*L.n_ff+63)/64),unsigned(groups),unsigned(token_tiles)),dim3(32,4),0,stream>>>(L,ptr,starts,ng,dst,qx,gu,gx);
    quantize_rows<true,TERMS><<<unsigned(entries),128,0,stream>>>(gu,qh,starts,ng,tok,int(L.n_ff),int(L.n_ff),down_scale,gh);
    product<true,TERMS><<<dim3(unsigned((L.n_embd+63)/64),unsigned(groups),unsigned(token_tiles)),dim3(32,4),0,stream>>>(L,ptr,starts,ng,dst,qh,out,gh);
}
}

void nvfp4_tensor_grouped(const NativeExpertLayout& L,const unsigned long long* ptr,const int32_t* starts,
    const int32_t* ng,const int32_t* dst,const int32_t* tok,int64_t groups,int64_t entries,
    const float* x,void* scratch,float* out,void* stream,int token_tiles) {
    if(token_tiles<1 || token_tiles>128) throw std::invalid_argument("NVFP4 token tiles must be 1..128");
    if(!x || !ptr || !starts || !ng || !tok || !scratch || !out || L.gu_type!=40 || L.d_type!=40 || L.n_embd!=2560 || L.n_ff!=640 || !L.tail_off) throw std::invalid_argument("NVFP4 Tensor decode requires FP32 input and the supported 2560/640 expert geometry");
    if(groups<=0 || entries<=0) return;
    // Cache by host thread AND current device, never reuse one device's success
    // for another. The normal warm path validates before graph capture.
    static thread_local int checked_device=-1;
    int dev=-1;
    if(cudaGetDevice(&dev)!=cudaSuccess) throw std::runtime_error("NVFP4: cannot query CUDA device");
    if(dev!=checked_device) {
        cudaDeviceProp prop{};
        if(cudaGetDeviceProperties(&prop,dev)!=cudaSuccess || prop.major!=12 || prop.minor!=0)
            throw std::runtime_error("NVFP4 native precision requires SM120");
        checked_device=dev;
    }
    float gs=L.input_scale_gu, ds=L.input_scale_down;
    if(!std::isfinite(gs) || !std::isfinite(ds) || gs<=0 || ds<=0) throw std::runtime_error("invalid NVFP4 checkpoint activation scales");
    launch<1>(L,ptr,starts,ng,dst,tok,groups,entries,x,scratch,out,static_cast<cudaStream_t>(stream),gs,ds,token_tiles);
    const cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(std::string("NVFP4 Tensor decode: ")+cudaGetErrorString(error));
}

size_t nvfp4_tensor_scratch_bytes(int64_t entries,int64_t embd,int64_t ff) {
    if(entries<=0) return 0;
    if(embd!=2560 || ff!=640) throw std::invalid_argument("NVFP4 workspace geometry must be 2560/640");
    return aligned(size_t(entries)*(embd/64)*sizeof(QBlock)) +
           aligned(size_t(entries)*2*ff*sizeof(float)) +
           aligned(size_t(entries)*(ff/64)*sizeof(QBlock)) +
           aligned(size_t(entries)*2*sizeof(float));
}
namespace {
struct OneGroup { unsigned long long ptr; int starts[2]; int count; };
__global__ void set_one_group(OneGroup* g,const uint8_t* blob,int entries) {
    g->ptr=reinterpret_cast<unsigned long long>(blob);
    g->starts[0]=0; g->starts[1]=entries; g->count=1;
}
struct PrefillMetadata { unsigned long long ptr[16]; int starts[17]; int count; };
static_assert(sizeof(PrefillMetadata)<=256);
__global__ void set_prefill_group(PrefillMetadata* m,Nvfp4PrefillGroup g,
    const int32_t* tokens,int32_t* tok,int32_t* dst,int entries) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0) {
        m->count=g.count; m->starts[0]=0;
        for(int j=0;j<g.count;++j) {
            m->ptr[j]=reinterpret_cast<unsigned long long>(g.weights[j]);
            m->starts[j+1]=m->starts[j]+g.rows[j];
        }
    }
    if(i>=entries) return;
    int group=0,first=0;
    while(group+1<g.count && i>=first+g.rows[group]) first+=g.rows[group++];
    const int source=g.offsets[group]+i-first;
    tok[i]=tokens[source]; dst[i]=source;
}
}
size_t nvfp4_tensor_prefill_bytes(int64_t entries) {
    const size_t custom=256+2*aligned(size_t(entries)*16*sizeof(int32_t))+nvfp4_tensor_scratch_bytes(entries*16);
#ifdef STRATA_NVFP4_CUTLASS_BUILT
    if(nvfp4_cutlass_enabled()) {
        const size_t official=nvfp4_cutlass_prefill_bytes(entries);
        return official>custom?official:custom;
    }
#endif
    return custom;
}
void nvfp4_tensor_prefill(const NativeExpertLayout& L,const uint8_t* blob,
    const int32_t* tokens,int entries,const float* x,void* scratch,float* out,void* stream) {
    if(entries<=0) return;
    auto* g=static_cast<OneGroup*>(scratch);
    set_one_group<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(g,blob,entries);
    nvfp4_tensor_grouped(L,&g->ptr,g->starts,&g->count,nullptr,tokens,1,entries,
        x,static_cast<unsigned char*>(scratch)+256,out,stream);
}
void nvfp4_tensor_prefill_group(const NativeExpertLayout& L,Nvfp4PrefillGroup group,
    const int32_t* tokens,const float* x,void* scratch,float* out,void* stream) {
#ifdef STRATA_NVFP4_CUTLASS_BUILT
    if(nvfp4_cutlass_use_group(group)) { nvfp4_cutlass_prefill(L,group,tokens,x,scratch,out,stream); return; }
#endif
    if(group.count<=0) return;
    if(group.count>16) throw std::invalid_argument("NVFP4 prefill group exceeds sixteen experts");
    int entries=0,max_rows=0;
    for(int i=0;i<group.count;++i) {
        if(!group.weights[i] || group.rows[i]<=0 || group.offsets[i]<0)
            throw std::invalid_argument("NVFP4 prefill group has invalid weights or rows");
        entries+=group.rows[i];
        if(group.rows[i]>max_rows) max_rows=group.rows[i];
    }
    auto* meta=static_cast<PrefillMetadata*>(scratch);
    auto* tok=reinterpret_cast<int32_t*>(static_cast<unsigned char*>(scratch)+256);
    auto* dst=reinterpret_cast<int32_t*>(reinterpret_cast<unsigned char*>(tok)+aligned(size_t(entries)*4));
    void* work=reinterpret_cast<unsigned char*>(dst)+aligned(size_t(entries)*4);
    set_prefill_group<<<unsigned((entries+127)/128),128,0,static_cast<cudaStream_t>(stream)>>>(meta,group,tokens,tok,dst,entries);
    nvfp4_tensor_grouped(L,meta->ptr,meta->starts,&meta->count,dst,tok,group.count,entries,x,work,out,stream,((max_rows+7)/8<8?(max_rows+7)/8:8));
}
}
