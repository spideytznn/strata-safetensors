// Strata adapter to NVIDIA CUTLASS SM120 grouped NVFP4 GEMM (example 79d).
// CUTLASS is BSD-3-Clause; see docs/licenses/CUTLASS.txt.
// Original FP4 weights are repacked, never expanded to a floating-point matrix.
#include "strata/kernels/nvfp4_cutlass.hpp"
#include "strata/kernels/native_prefill.hpp"
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace strata::kernels {
namespace {
using namespace cute;
using Shape3 = Shape<int,int,int>;
using Problem = cutlass::gemm::GroupProblemShape<Shape3>;
using FP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using Tile = Shape<_128,_128,_128>;
using Cluster = Shape<_1,_1,_1>;
using Arch = cutlass::arch::Sm120;
using Op = cutlass::arch::OpClassBlockScaledTensorOp;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch,Op,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    float,float,float,RM*,4,float,RM*,4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch,Op,FP4,RM*,32,FP4,CM*,32,float,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem,Mainloop,Epilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using EA = typename Gemm::ElementA;
using EB = typename Gemm::ElementB;
using SF = typename Mainloop::ElementSF;
using SA = typename Kernel::InternalStrideA;
using SB = typename Kernel::InternalStrideB;
using SC = typename Kernel::InternalStrideC;
using SD = typename Kernel::InternalStrideD;
using LSA = typename Mainloop::InternalLayoutSFA;
using LSB = typename Mainloop::InternalLayoutSFB;
using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
struct Meta {
    Shape3 shape[16];
    const EA* a[16]; const EB* b[16];
    const SF* as[16]; const SF* bs[16];
    const float* c[16]; float* d[16];
    SA sa[16]; SB sb[16]; SC sc[16]; SD sd[16];
    LSA la[16]; LSB lb[16];
};
size_t align256(size_t n) { return (n+255)&~size_t(255); }
constexpr size_t gemm_workspace = 4*1024*1024;
struct Buffers {
    Meta* meta; unsigned char *a,*as,*b,*bs; float* gu; void* work;
    size_t a_step,as_step,b_step,bs_step,gu_step,work_bytes;
};
size_t partition(void* base,int rows,Buffers* result) {
    rows = ((rows+127)/128)*128;
    auto shape=make_shape(rows,2560,2560,1);
    const size_t astep=align256(size_t(rows)*2560/2);
    const size_t asstep=align256(size(filter_zeros(ScaleConfig::tile_atom_to_shape_SFA(shape))));
    const size_t bstep=align256(size_t(1280)*2560/2);
    const size_t bsstep=align256(size(filter_zeros(ScaleConfig::tile_atom_to_shape_SFB(make_shape(rows,1280,2560,1)))));
    const size_t gustep=align256(size_t(rows)*1280*4);
    size_t pos=0;
    auto take=[&](size_t n) { size_t at=pos;pos+=align256(n);return reinterpret_cast<unsigned char*>(reinterpret_cast<uintptr_t>(base)+at); };
    Buffers w{}; w.meta=reinterpret_cast<Meta*>(take(sizeof(Meta)));
    w.a=take(16*astep);w.as=take(16*asstep);w.b=take(16*bstep);w.bs=take(16*bsstep);
    w.gu=reinterpret_cast<float*>(take(16*gustep));w.work=take(gemm_workspace);
    w.a_step=astep;w.as_step=asstep;w.b_step=bstep;w.bs_step=bsstep;w.gu_step=gustep;
    if(result)*result=w;return pos;
}
template<bool Down>
__device__ void metadata(Buffers w,Nvfp4PrefillGroup g,float* out,int e) {
    int m=g.rows[e],n=Down?2560:1280,k=Down?640:2560;
    Meta& t=*w.meta;t.shape[e]=make_shape(m,n,k);
    t.a[e]=reinterpret_cast<EA*>(w.a+e*w.a_step);
    t.b[e]=reinterpret_cast<EB*>(w.b+e*w.b_step);
    t.as[e]=reinterpret_cast<SF*>(w.as+e*w.as_step);
    t.bs[e]=reinterpret_cast<SF*>(w.bs+e*w.bs_step);
    t.c[e]=nullptr;
    t.d[e]=Down ? out+size_t(g.offsets[e])*2560 : reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(w.gu)+e*w.gu_step);
    t.sa[e]=cutlass::make_cute_packed_stride(SA{},make_shape(m,k,1));
    t.sb[e]=cutlass::make_cute_packed_stride(SB{},make_shape(n,k,1));
    t.sc[e]=cutlass::make_cute_packed_stride(SC{},make_shape(m,n,1));
    t.sd[e]=cutlass::make_cute_packed_stride(SD{},make_shape(m,n,1));
    t.la[e]=ScaleConfig::tile_atom_to_shape_SFA(make_shape(m,n,k,1));
    t.lb[e]=ScaleConfig::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
}
template<bool Down>
__device__ void repack(Buffers w,Nvfp4PrefillGroup g,size_t down_offset,int e,int i) {
    const int n=Down?2560:1280,k=Down?640:2560;
    if(i>=n*(k/16))return;
    const int row=i/(k/16),sub=i%(k/16);
    const auto* block=g.weights[e]+(Down?down_offset:0)+(size_t(row)*(k/64)+sub/4)*36;
    auto* dst=w.b+e*w.b_step+size_t(row)*k/2+sub*8;
    // Original 36-byte blocks and their 8-byte subblocks are 4-byte aligned.
    // Coalesce eight scalar byte operations into two 32-bit loads/stores.
    const auto* src=reinterpret_cast<const uint32_t*>(block+4+(sub%4)*8);
    auto compact=[] __device__ (uint32_t v) {
        v&=0x0f0f0f0fu;v=(v|(v>>4))&0x00ff00ffu;
        return (v|(v>>8))&0xffffu;
    };
    const uint32_t lo=src[0],hi=src[1];
    reinterpret_cast<uint32_t*>(dst)[0]=compact(lo)|(compact(hi)<<16);
    reinterpret_cast<uint32_t*>(dst)[1]=compact(lo>>4)|(compact(hi>>4)<<16);
    auto layout=ScaleConfig::tile_atom_to_shape_SFB(make_shape(g.rows[e],n,k,1));
    w.bs[e*w.bs_step+layout(make_coord(row,sub*16,0))]=block[sub%4];
}
template<bool Down>
__device__ void quantize(Buffers w,Nvfp4PrefillGroup g,NativeExpertLayout L,
    const int32_t* tokens,const float* x,int e,int i) {
    const int k=Down?640:2560,n=Down?2560:1280;
    const int row=i/(k/16),sub=i%(k/16);
    auto layout=ScaleConfig::tile_atom_to_shape_SFA(make_shape(g.rows[e],n,k,1));
    if(row>=g.rows[e]) {
        // CUTLASS reads scale tiles in full even when A/D rows are predicated.
        // Initialize the final tile's unused scales inside this same launch.
        if(row<((g.rows[e]+127)/128)*128)
            w.as[e*w.as_step+layout(make_coord(row,sub*16,0))]=0;
        return;
    }
    const float scale_global=Down?L.input_scale_down:L.input_scale_gu;
    const float* tails=reinterpret_cast<const float*>(g.weights[e]+L.tail_off);
    const float* gu=reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(w.gu)+e*w.gu_step)+size_t(row)*1280;
    const int source=Down?0:tokens[g.offsets[e]+row];
    float v[16],amax=0;
    #pragma unroll
    for(int j=0;j<16;++j) {
        int col=sub*16+j;float f;
        if constexpr(Down) {
            float gate=gu[col]*L.input_scale_gu*tails[0];
            float up=gu[col+640]*L.input_scale_gu*tails[1];
            f=(gate/(1.f+expf(-gate)))*up;
        } else f=x[size_t(source)*2560+col];
        v[j]=__fdiv_rn(f,scale_global);amax=fmaxf(amax,fabsf(v[j]));
    }
    const unsigned char s=__nv_cvt_float_to_fp8(__fdiv_rn(amax,6.f),__NV_SATFINITE,__NV_E4M3);
    __nv_fp8_e4m3 sf;sf.__x=s;const float fscale=float(sf),inv=fscale>0?__fdiv_rn(1.f,fscale):0.f;
    auto* dst=w.a+e*w.a_step+size_t(row)*k/2+sub*8;
    #pragma unroll
    for(int j=0;j<8;++j) dst[j]=__nv_cvt_float_to_fp4(v[2*j]*inv,__NV_E2M1,cudaRoundNearest)|
        (__nv_cvt_float_to_fp4(v[2*j+1]*inv,__NV_E2M1,cudaRoundNearest)<<4);
    w.as[e*w.as_step+layout(make_coord(row,sub*16,0))]=s;
}
template<bool Down>
__global__ void prepare(Buffers w,Nvfp4PrefillGroup g,NativeExpertLayout L,
    const int32_t* tokens,const float* x,float* out) {
    constexpr int weight_subblocks=Down?2560*40:1280*160;
    const int e=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0)metadata<Down>(w,g,out,e);
    // Grouped CUTLASS maintains mutable TMA descriptors in global workspace.
    // Initialize its used bytes even when scratch is borrowed from a cache slot.
    if(e==0)for(size_t j=size_t(i)*4;j<w.work_bytes;j+=size_t(gridDim.x)*blockDim.x*4)
        *reinterpret_cast<uint32_t*>(static_cast<unsigned char*>(w.work)+j)=0;
    if constexpr(!Down) {
        // Keep the scratch output defined before the TMA store, including for
        // tools which do not propagate initialization through that store.
        auto* gu=reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(w.gu)+e*w.gu_step);
        for(size_t j=i;j<size_t(g.rows[e])*1280;j+=size_t(gridDim.x)*blockDim.x)gu[j]=0.f;
    }
    // The two inputs and the descriptors are independent. One kernel prepares
    // all three before the following GEMM, without an intervening host launch.
    if(i<weight_subblocks)repack<Down>(w,g,L.down_off,e,i);
    else quantize<Down>(w,g,L,tokens,x,e,i-weight_subblocks);
}
__global__ void scale_output(Nvfp4PrefillGroup g,NativeExpertLayout L,float* out) {
    int e=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<g.rows[e]*2560) {
        const float* tails=reinterpret_cast<const float*>(g.weights[e]+L.tail_off);
        float& v=out[size_t(g.offsets[e])*2560+i];v=v*L.input_scale_down*tails[2];
    }
}
void check(cutlass::Status s,const char* what) {
    if(s!=cutlass::Status::kSuccess)throw std::runtime_error(std::string(what)+": "+cutlassGetStatusString(s));
}
void check_cuda(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string("CUTLASS NVFP4: ")+cudaGetErrorString(status));
}
Gemm::Arguments arguments(Buffers w,int count) {
    static thread_local int cached_device=-1,sm_count=0;
    int dev;check_cuda(cudaGetDevice(&dev));
    if(dev!=cached_device) {cudaDeviceProp p{};check_cuda(cudaGetDeviceProperties(&p,dev));sm_count=p.multiProcessorCount;cached_device=dev;}
    cutlass::KernelHardwareInfo hw{};hw.device_id=dev;hw.sm_count=sm_count;
    Meta* t=w.meta;
    return Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
        {count,t->shape,nullptr},{t->a,t->sa,t->b,t->sb,t->as,t->la,t->bs,t->lb},
        {{1.f,0.f},t->c,t->sc,t->d,t->sd},hw};
}
void gemm(const Gemm::Arguments& args,Buffers w,cudaStream_t stream) {
    Gemm op;check(op.can_implement(args),"CUTLASS can_implement");
    check(op.initialize(args,w.work,stream),"CUTLASS initialize");
    check(op.run(stream),"CUTLASS run");
}
}
bool nvfp4_cutlass_enabled() {
    return nvfp4_prefill_backend()!=Nvfp4PrefillBackend::Custom;
}
bool nvfp4_cutlass_use_group(const Nvfp4PrefillGroup& g) {
    const auto backend=nvfp4_prefill_backend();
    if(backend==Nvfp4PrefillBackend::Custom || g.count<=0)return false;
    if(backend==Nvfp4PrefillBackend::Cutlass)return true;
    // Complete expert timings on RTX 5090 include weight repacking. Small groups
    // need more rows to amortize CUTLASS metadata and grouped scheduling.
    if(g.count>16)throw std::invalid_argument("NVFP4 group exceeds sixteen experts");
    int64_t rows=0;for(int e=0;e<g.count;++e)rows+=g.rows[e];
    return rows>=int64_t(g.count)*(g.count>=8?32:64);
}
size_t nvfp4_cutlass_prefill_bytes(int64_t rows) {
    if(rows<=0)return 0;
    if(rows>1048576)throw std::invalid_argument("CUTLASS prefill capacity too large");
    return partition(nullptr,int(rows),nullptr);
}
void nvfp4_cutlass_prefill(const NativeExpertLayout& L,Nvfp4PrefillGroup g,
    const int32_t* tokens,const float* x,void* scratch,float* out,void* stream) {
    if(g.count==0)return;
    if(g.count<0 || g.count>16 || L.gu_type!=40 || L.d_type!=40 || L.n_embd!=2560 || L.n_ff!=640 ||
       !L.tail_off || !tokens || !x || !scratch || !out || !std::isfinite(L.input_scale_gu) ||
       !std::isfinite(L.input_scale_down) || L.input_scale_gu<=0 || L.input_scale_down<=0)
        throw std::invalid_argument("CUTLASS prefill requires calibrated NVFP4 2560/640 experts");
    int rows=0;for(int e=0;e<g.count;++e) {
        if(!g.weights[e] || g.rows[e]<=0 || g.offsets[e]<0)throw std::invalid_argument("invalid CUTLASS group");
        if(g.rows[e]>rows)rows=g.rows[e];
    }
    Buffers w{};partition(scratch,rows,&w);auto st=static_cast<cudaStream_t>(stream);
    auto args=arguments(w,g.count);w.work_bytes=Gemm::get_workspace_size(args);
    if(w.work_bytes>gemm_workspace)throw std::runtime_error("CUTLASS workspace exceeds reservation");
    const int padded_rows=((rows+127)/128)*128;
    prepare<false><<<dim3((1280*160+padded_rows*160+127)/128,g.count),128,0,st>>>(w,g,L,tokens,x,out);
    gemm(args,w,st);
    prepare<true><<<dim3((2560*40+padded_rows*40+127)/128,g.count),128,0,st>>>(w,g,L,tokens,x,out);
    gemm(args,w,st);
    scale_output<<<dim3((rows*2560+255)/256,g.count),256,0,st>>>(g,L,out);
    check_cuda(cudaGetLastError());
}
}
