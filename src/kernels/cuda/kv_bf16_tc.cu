#include "strata/kernels/kv_bf16_tc.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int D=256, H=12, C=64, NT=256;
void check(const char* where) {
    const auto e=cudaGetLastError();
    if(e!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
__device__ uint16_t bfround(float x){return __bfloat16_as_ushort(__float2bfloat16_rn(x));}
__global__ void append_bf16(uint16_t* k,uint16_t* v,const int32_t* table,const int32_t* steps,
    int step_stride,long long pos0,const float* kc,const float* vc,int cur_stride,int nh,int hd,int ps,
    KvHostPools host,KvHostPools stage) {
    const int t=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
    const long long pos=steps?steps[t*step_stride+kStepPos]:pos0+t;
    const int page=table[pos/ps];
    const uint16_t a=bfround(kc[(size_t)t*cur_stride+h*hd+d]);
    const uint16_t b=bfround(vc[(size_t)t*cur_stride+h*hd+d]);
    if(page>=0){const long long row=((long long)page*nh+h)*ps+pos%ps;k[row*hd+d]=a;v[row*hd+d]=b;}
    const long long row=((pos/ps)*nh+h)*ps+pos%ps;
    if(host.k_pool){host.k_pool[row*hd+d]=a;host.v_pool[row*hd+d]=b;}
    if(stage.k_pool){stage.k_pool[row*hd+d]=a;stage.v_pool[row*hd+d]=b;}
}
__device__ float wmax(float x) {
    for(int s=16;s;s/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,s));
    return x;
}
__device__ float wsum(float x) {
    for(int s=16;s;s/=2)x+=__shfl_xor_sync(0xffffffffu,x,s);
    return x;
}
__device__ void mma16(float (&d)[4],const unsigned (&a)[4],const unsigned (&b)[2]) {
    {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
}
// One CTA handles 64 sparse cells and all 12 query heads of one KV head.
// QK and PV use BF16 MMA with FP32 accumulation. No expanded
// global-memory KV, allocations, host reads or host-dependent graph arguments.
template<bool FUSED=false>
__global__ void __launch_bounds__(NT) chunks(const float* q,QsaAttnPools p,
    const int32_t* ids,const int32_t* steps,int cap,int nh,int ps,
    float* acc,float* pm,float* pl,int nc,long long stride,float* result=nullptr) {
    const int qi=blockIdx.z,h=blockIdx.y;
    q+=(size_t)qi*nh*H*D+h*H*D;
    ids+=(size_t)qi*cap;
    steps+=(size_t)qi*kStepCount;
    if constexpr(!FUSED){acc+=(size_t)qi*stride;pm+=(size_t)qi*stride;pl+=(size_t)qi*stride;}
    const int tid=threadIdx.x,lane=tid&31,warp=tid/32;
    const int width=max(0,min(cap,steps[kStepWidth]));
    if constexpr(!FUSED) {
        if(blockIdx.x*C>=width) {
            if(tid<H){const size_t at=((size_t)h*nc+blockIdx.x)*H+tid;pm[at]=-FLT_MAX;pl[at]=0;}
            return;
        }
    }
    // Padding keeps the MMA fragment loads from serializing on shared-memory banks.
    __shared__ __align__(16) uint16_t bq[16][D+8];
    __shared__ long long row[C];
    __shared__ float score[H][C];
    __shared__ __align__(16) uint16_t prob[16][C+8],val[C][D+8];
    __shared__ float sm[16],sl[16];
    __shared__ float rm[16],rl[16],old_weight[16],new_weight[16];
    float fused_acc[4][4]={};
    if constexpr(FUSED) {
        if(tid<16){rm[tid]=-FLT_MAX;rl[tid]=0;}
    }
    // Round each query to BF16 for the Tensor Core product.
    {
        for(int i=tid;i<16*D;i+=NT)bq[i/D][i%D]=bfround(i/D<H?q[i]:0);
    }
    __syncthreads();
    const int iterations=FUSED?(width+C-1)/C:1;
    for(int iteration=0;iteration<iterations;++iteration) {
    const int chunk=FUSED?iteration:(int)blockIdx.x;
    for(int c=tid;c<C;c+=NT) {
        const int index=chunk*C+c;
        const int cell=index<width?ids[index]:-1;
        // Invalid selections and nonresident pages are masked before any pool read.
        const int page=cell>=0 && cell<steps[kStepNKv]?p.page_table[cell/ps]:-1;
        row[c]=page>=0?((long long)page*nh+h)*ps+cell%ps:-1;
    }
    __syncthreads();

    for(int i=tid;i<C*(D/2);i+=NT) {
        const int c=i/(D/2),d=(i%(D/2))*2;
        {
            *reinterpret_cast<unsigned*>(&val[c][d])=row[c]>=0?
                *reinterpret_cast<const unsigned*>(p.v_pool+row[c]*D+d):0;
        }
    }
    __syncthreads();
    // Eight warps cover the 64 selected cells as eight MMA columns each.
    float total[4]={};
    {
        for(int k=0;k<D;k+=16) {
            unsigned a[4],b[2]={};
            for(int z=0;z<4;++z){const int head=lane/4+(z%2)*8,d=k+(lane%4)*2+(z/2)*8;
                a[z]=*reinterpret_cast<const unsigned*>(&bq[head][d]);}
            const int c=warp*8+lane/4;
            if(row[c]>=0)for(int z=0;z<2;++z){const int d=k+(lane%4)*2+z*8;
                b[z]=*reinterpret_cast<const unsigned*>(p.k_pool+row[c]*D+d);}
            mma16(total,a,b);
        }
    }
    for(int z=0;z<4;++z) {
        const int head=lane/4+(z/2)*8,c=warp*8+(lane%4)*2+z%2;
        if(head<H)score[head][c]=row[c]>=0?total[z]*(1.0f/16.0f):-FLT_MAX;
    }
    __syncthreads();
    for(int head=warp;head<16;head+=8) {
        const float m=head<H?wmax(fmaxf(score[head][lane],score[head][lane+32])):-FLT_MAX;
        const float a=head<H && row[lane]>=0?__expf(score[head][lane]-m):0;
        const float b=head<H && row[lane+32]>=0?__expf(score[head][lane+32]-m):0;
        const float l=wsum(a+b);
        prob[head][lane]=bfround(a);
        prob[head][lane+32]=bfround(b);
        if(lane==0){
            sm[head]=m;sl[head]=l;
            if constexpr(FUSED) {
                const float merged=fmaxf(rm[head],m);
                const float a=rl[head]>0?__expf(rm[head]-merged):0;
                const float b=l>0?__expf(m-merged):0;
                old_weight[head]=a;new_weight[head]=b;
                rm[head]=merged;rl[head]=rl[head]*a+l*b;
            }
        }
    }
    __syncthreads();
    for(int tile=warp;tile<D/8;tile+=8) {
        float out[4]={};
        for(int k=0;k<C;k+=16) {
            unsigned a[4],b[2];
            for(int z=0;z<4;++z) {
                const int head=lane/4+(z%2)*8,col=k+(lane%4)*2+(z/2)*8;
                a[z]=*reinterpret_cast<const unsigned*>(&prob[head][col]);
            }
            for(int z=0;z<2;++z) {
                const int r=k+(lane%4)*2+z*8,d=tile*8+lane/4;
                b[z]=unsigned(val[r][d])|(unsigned(val[r+1][d])<<16);
            }
            mma16(out,a,b);
        }
        for(int z=0;z<4;++z) {
            const int head=lane/4+(z/2)*8,d=tile*8+(lane%4)*2+z%2;
            if(head<H) {
                if constexpr(FUSED) {
                    fused_acc[tile/8][z]=fmaf(out[z],new_weight[head],fused_acc[tile/8][z]*old_weight[head]);
                } else {
                    acc[((size_t)h*nc+chunk)*H*D+head*D+d]=out[z];
                }
            }
        }
    }
    if constexpr(!FUSED) {
        if(tid<H) {
            const size_t at=((size_t)h*nc+chunk)*H+tid;
            pm[at]=sm[tid];pl[at]=sl[tid];
        }
    }
    __syncthreads(); // No next tile may overwrite V/probabilities still in use.
    }
    if constexpr(FUSED) {
        for(int tile=warp;tile<D/8;tile+=8)for(int z=0;z<4;++z) {
            const int head=lane/4+(z/2)*8,d=tile*8+(lane%4)*2+z%2;
            if(head<H)result[((size_t)qi*nh*H+h*H+head)*D+d]=rl[head]>0?fused_acc[tile/8][z]/rl[head]:0;
        }
    }
}
void validate(const QsaShapes& s) {
    if(s.head_dim!=D || s.n_head_kv!=2 || s.n_head!=24 || s.page_size<=0)
        throw std::runtime_error("BF16 KV Tensor Core backend requires 24 Q heads, 2 KV heads, dimension 256");
}
}
void kv_append_bf16_steps(uint16_t* k,uint16_t* v,const int32_t* table,const int32_t* steps,
    int step_stride,const float* kc,const float* vc,int cur_stride,int n,const QsaShapes& s,
    void* stream,const KvHostPools* host) {
    validate(s);if(n<=0)return;
    append_bf16<<<dim3(n,2),D,0,(cudaStream_t)stream>>>(k,v,table,steps,step_stride,0,kc,vc,cur_stride,2,D,
        (int)s.page_size,host?*host:KvHostPools{},KvHostPools{});
    check("kv_append_bf16_steps");
}
void kv_append_bf16_prefill(const float* k,const float* v,int64_t n,int64_t pos,const int32_t* table,
    int ps,uint16_t* ko,uint16_t* vo,void* stream,const KvHostPools* host,const KvHostPools* stage) {
    if(n<=0)return;
    append_bf16<<<dim3((unsigned)n,2),D,0,(cudaStream_t)stream>>>(ko,vo,table,nullptr,0,pos,k,v,512,2,D,ps,
        host?*host:KvHostPools{},stage?*stage:KvHostPools{});
    check("kv_append_bf16_prefill");
}
void qsa_bf16_tc_chunks(const float* q,const QsaAttnPools& p,const int32_t* ids,const int32_t* steps,
    int64_t cap,const QsaShapes& s,float* scratch,int64_t n_q,void* stream) {
    validate(s);
    if(!p.bf16 || !p.k_pool || !p.v_pool || p.k_q || p.v_q || p.k_q4 || p.v_q4)
        throw std::runtime_error("BF16 attention requires complete BF16 pools");
    const int nc=(int)((cap+C-1)/C);
    const long long stride=(long long)qsa_decode_attn_scratch_floats(cap,s);
    chunks<false><<<dim3(nc,2,(unsigned)n_q),NT,0,(cudaStream_t)stream>>>(q,p,ids,steps,(int)cap,2,(int)s.page_size,
        scratch,scratch+(size_t)nc*24*D,scratch+(size_t)nc*24*(D+1),nc,stride);
    check("qsa_bf16_tc_chunks");
}
void qsa_bf16_tc_prompt(const float* q,const QsaAttnPools& p,const int32_t* ids,const int32_t* steps,
    int64_t cap,const QsaShapes& s,float* attn,int64_t n_q,void* stream) {
    validate(s);
    if(!p.bf16 || !p.k_pool || !p.v_pool || p.k_q || p.v_q || p.k_q4 || p.v_q4 ||
        !p.page_table || cap<=0 || n_q<=0 || n_q>65535)
        throw std::runtime_error("BF16 fused prompt attention: invalid pools or extent");
    chunks<true><<<dim3(1,2,(unsigned)n_q),NT,0,(cudaStream_t)stream>>>(q,p,ids,steps,(int)cap,2,(int)s.page_size,
        nullptr,nullptr,nullptr,(int)((cap+C-1)/C),0,attn);
    check("qsa_bf16_tc_prompt");
}
}
