// Original packed-weight adapter parity and same-process complete expert timing.
#include "strata/kernels/nvfp4_cutlass.hpp"
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include "strata/kernels/cpu/nvfp4_w4a4.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <filesystem>
#include <stdexcept>
using namespace strata::kernels;
void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) {ck(cudaMalloc((void**)&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void upload(const std::vector<T>& v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> read(size_t n){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
template<class F> double timing(F launch,cudaStream_t stream) {
    launch();ck(cudaStreamSynchronize(stream));
    cudaGraph_t g;cudaGraphExec_t ex;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<8;++i)launch();
    ck(cudaStreamEndCapture(stream,&g));ck(cudaGraphInstantiate(&ex,g,nullptr,nullptr,0));
    cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
    double best=1e30;
    for(int r=0;r<3;++r) {
        ck(cudaEventRecord(a,stream));
        for(int i=0;i<4;++i)ck(cudaGraphLaunch(ex,stream));
        ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
        best=std::min(best,double(ms)*1000/32);
    }
    ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));return best;
}
// Independent CUDA conversion oracle: byte-level ties/subnormals/saturation.
__global__ void quant_oracle(const float* x, unsigned char* e4, unsigned char* e2, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    e4[i]=__nv_cvt_float_to_fp8(fabsf(x[i]),__NV_SATFINITE,__NV_E4M3);
    e2[i]=__nv_cvt_float_to_fp4(x[i],__NV_E2M1,cudaRoundNearest);
}
__global__ void block_quant_oracle(const float* x, float* out, int n, float global) {
    int base=(blockIdx.x*blockDim.x+threadIdx.x)*16;if(base>=n)return;
    float v[16],amax=0;
    for(int j=0;j<16;++j) {v[j]=__fdiv_rn(x[base+j],global);amax=fmaxf(amax,fabsf(v[j]));}
    __nv_fp8_e4m3 scale;scale.__x=__nv_cvt_float_to_fp8(__fdiv_rn(amax,6.f),__NV_SATFINITE,__NV_E4M3);
    float sc=float(scale),inv=sc>0?__fdiv_rn(1.f,sc):0.f;
    for(int j=0;j<16;++j) {
        auto q=__nv_cvt_float_to_fp4(v[j]*inv,__NV_E2M1,cudaRoundNearest);
        out[base+j]=__half2float(__nv_cvt_fp4_to_halfraw(q,__NV_E2M1))*sc;
    }
}
void quant_check() {
    namespace cpu=strata::kernels::cpu;
    std::vector<float> x;
    for(int i=-65536;i<=65536;++i)x.push_back(float(i)/128.f);
    for(int c=0;c<126;++c) {
        float m=(cpu::nvfp4_e4m3_value(c)+cpu::nvfp4_e4m3_value(c+1))*.5f;
        for(float v:{std::nextafter(m,0.f),m,std::nextafter(m,INFINITY)})x.push_back(v);
    }
    for(float m:{.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f})
        for(float v:{std::nextafter(m,0.f),m,std::nextafter(m,INFINITY)}) {x.push_back(v);x.push_back(-v);}
    x.push_back(-0.f);
    while(x.size()%16)x.push_back(0.f);
    Device<float> dx(x.size()); Device<unsigned char> de4(x.size()),de2(x.size()); dx.upload(x);
    quant_oracle<<<unsigned((x.size()+255)/256),256>>>(dx.p,de4.p,de2.p,int(x.size()));
    ck(cudaDeviceSynchronize());auto e4=de4.read(x.size()),e2=de2.read(x.size());
    for(size_t i=0;i<x.size();++i)
        if(e4[i]!=cpu::nvfp4_e4m3_code(std::fabs(x[i])) || e2[i]!=cpu::nvfp4_e2m1_code(x[i]))
            throw std::runtime_error("CPU FP4/E4M3 encoding differs from CUDA");
    std::printf("CPU/CUDA FP4 and E4M3: %zu exact code comparisons passed\n",x.size());
    Device<float> dy(x.size());std::vector<float> host(x.size());
    for(float global:{.003f,1.f,3.14f})for(float amplitude:{1.f,1e-6f,1e5f}) {
        auto row=x;for(auto& v:row)v*=amplitude;dx.upload(row);
        block_quant_oracle<<<unsigned((x.size()/16+127)/128),128>>>(dx.p,dy.p,int(x.size()),global);
        ck(cudaDeviceSynchronize());auto got=dy.read(x.size());
        cpu::nvfp4_quantize_w4a4(row.data(),host.data(),int(row.size()),global);
        if(std::memcmp(got.data(),host.data(),host.size()*4))throw std::runtime_error("CPU block quantization differs from CUDA");
#ifdef STRATA_NATIVE_EXPERTS
        if(cpu::cpu_avx512_ok()) {
            cpu::nvfp4_512_quantize_w4a4(row.data(),host.data(),int(row.size()),global);
            if(std::memcmp(got.data(),host.data(),host.size()*4))throw std::runtime_error("AVX512 block quantization differs from CUDA");
        }
#endif
    }
    std::printf("CPU/CUDA calibrated block quantization: 9 exact row comparisons passed\n");
}
__global__ void silu_quant_oracle(const float* gu,float* out,int n,bool canonical) {
    int base=(blockIdx.x*blockDim.x+threadIdx.x)*16;if(base>=n)return;
    float v[16],amax=0;
    for(int j=0;j<16;++j) {
        float g=gu[2*(base+j)],u=gu[2*(base+j)+1];
        float e=canonical?float(exp(-double(g))):expf(-g);
        v[j]=__fdiv_rn((g/(1.f+e))*u,.003f);amax=fmaxf(amax,fabsf(v[j]));
    }
    __nv_fp8_e4m3 scale;scale.__x=__nv_cvt_float_to_fp8(__fdiv_rn(amax,6.f),__NV_SATFINITE,__NV_E4M3);
    float sc=float(scale),inv=sc>0?__fdiv_rn(1.f,sc):0.f;
    for(int j=0;j<16;++j) {
        auto q=__nv_cvt_float_to_fp4(v[j]*inv,__NV_E2M1,cudaRoundNearest);
        out[base+j]=__half2float(__nv_cvt_fp4_to_halfraw(q,__NV_E2M1))*sc;
    }
}
void silu_check() {
#ifdef STRATA_NATIVE_EXPERTS
    namespace cpu=strata::kernels::cpu;
    if(!cpu::cpu_avx512_ok())return;
    constexpr int n=8*1024*1024;
    std::vector<float> gu(2*n),h(n),ref(n);uint32_t state=5090;
    for(int i=0;i<2*n;++i) {state^=state<<13;state^=state>>17;state^=state<<5;gu[i]=(float(state>>8)/16777216.f-.5f)*24.f;}
    for(int i=0;i<n;++i) {float g=gu[2*i],u=gu[2*i+1];h[i]=(g/(1.f+float(std::exp(-double(g)))))*u;}
    cpu::nvfp4_512_quantize_w4a4(h.data(),ref.data(),n,.003f);
    Device<float> dg(gu.size()),dy(n);dg.upload(gu);
    for(bool canonical:{false,true}) {
        silu_quant_oracle<<<n/16/128,128>>>(dg.p,dy.p,n,canonical);ck(cudaDeviceSynchronize());
        auto out=dy.read(n);size_t count=0;
        for(int i=0;i<n;++i)count+=std::memcmp(&out[i],&ref[i],4)!=0;
        std::printf("SwiGLU GPU %s vs CPU canonical: %zu/%d FP4 activation differences\n",canonical?"canonical":"expf",count,n);
        if(canonical && count)throw std::runtime_error("canonical SwiGLU does not match CPU");
    }
#endif
}
int main(int argc,char** argv) {
try {
#ifdef _WIN32
    _putenv_s("STRATA_NVFP4_DECODE","w4a4");
#else
    setenv("STRATA_NVFP4_DECODE","w4a4",1);
#endif
    if(argc==2 && std::strcmp(argv[1],"--silu-check")==0) {silu_check();return 0;}
    quant_check();
    bool initcheck=false,check_only=false;std::string expert_dir;float amplitude=1.f;
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--initcheck")==0) initcheck=check_only=true;
        else if(std::strcmp(argv[i],"--check-only")==0) check_only=true;
        else if(std::strcmp(argv[i],"--experts-dir")==0 && i+1<argc) expert_dir=argv[++i];
        else if(std::strcmp(argv[i],"--amplitude")==0 && i+1<argc) amplitude=std::stof(argv[++i]);
        else throw std::runtime_error("usage: nvfp4_cutlass_test [--check-only|--initcheck] [--experts-dir DIR] [--amplitude FLOAT]");
    }
    const char* backend=std::getenv("STRATA_NVFP4_PREFILL_BACKEND");
    if(backend && *backend && std::strcmp(backend,"custom")!=0)
        throw std::runtime_error("parity test requires STRATA_NVFP4_PREFILL_BACKEND unset or custom");
    NativeExpertLayout L{};L.gu_type=L.d_type=40;L.n_embd=2560;L.n_ff=640;
    L.gu_row=40*36;L.d_row=10*36;L.up_off=640*L.gu_row;L.down_off=2*L.up_off;
    L.tail_off=L.down_off+2560*L.d_row;L.bytes=L.tail_off+16;
    L.input_scale_gu=.003f;L.input_scale_down=.002f;
    constexpr int E=16,T=513;
    std::vector<unsigned char> weights(E*L.bytes);
    uint32_t random_state=9950;
    auto random=[&] {random_state^=random_state<<13;random_state^=random_state>>17;random_state^=random_state<<5;return random_state;};
    for(int e=0;e<E;++e) {
        auto* p=weights.data()+e*L.bytes;
        for(size_t i=0;i<L.tail_off;i+=36) {
            for(int s=0;s<4;++s)p[i+s]=uint8_t(14+random()%65);
            for(int j=0;j<32;++j)p[i+4+j]=uint8_t(random());
        }
        const float scales[]={.03f,.04f,.05f,0};std::memcpy(p+L.tail_off,scales,16);
    }
    if(!expert_dir.empty()) {
        const char* keys[]={"0-0","0-7","12-100","24-300","40-5","47-511"};
        for(int e=0;e<E;++e) {
            auto file=std::filesystem::path(expert_dir)/(std::string("expert-")+keys[e%6]+".bin");
            std::ifstream in(file,std::ios::binary);in.read(reinterpret_cast<char*>(weights.data()+e*L.bytes),L.bytes);
            if(!in)throw std::runtime_error("failed to read real expert export");
        }
    }
    std::vector<float> x(T*2560);for(size_t i=0;i<x.size();++i)x[i]=amplitude*std::sin(float(i)*.073f);
    std::vector<int32_t> tokens(E*T);for(int i=0;i<E*T;++i)tokens[i]=(i*17)%T;
    Device<uint8_t> dw(weights.size()),oldwork(nvfp4_tensor_prefill_bytes(T)),newwork(nvfp4_cutlass_prefill_bytes(T));
    Device<float> dx(x.size()),a(E*T*2560),b(E*T*2560);Device<int32_t> dt(tokens.size());
    if(initcheck) {std::printf("CUTLASS scratch %p bytes %zu\n",static_cast<void*>(newwork.p),nvfp4_cutlass_prefill_bytes(T));std::fflush(stdout);}
    dw.upload(weights);dx.upload(x);dt.upload(tokens);
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    // Decode plans: shuffled expert/destination order, mixed per-expert scales,
    // separate resident/PCIe groups with a nonzero starting entry, and empty
    // groups. Compare actual gathered decode outputs with NVIDIA CUTLASS.
    {
        constexpr int C=80;
        std::vector<int32_t> starts(E+1),dst(C),tok(C),ids(C);
        std::vector<unsigned long long> ptr(E);
        std::vector<float> scales(2*E);
        Nvfp4PrefillGroup refgroup{};refgroup.count=E;
        for(int e=0;e<E;++e) {
            scales[2*e]=.001f*(1+e*.17f);scales[2*e+1]=.002f*(1+e*.11f);
        }
        for(int g=0;g<E;++g) {
            const int e=(g*7+3)%E;starts[g]=g*5;ptr[g]=(unsigned long long)(dw.p+e*L.bytes);
            refgroup.weights[g]=dw.p+e*L.bytes;refgroup.offsets[g]=g*5;refgroup.rows[g]=5;
            refgroup.input_scale_gu[g]=scales[2*e];refgroup.input_scale_down[g]=scales[2*e+1];
            for(int r=0;r<5;++r) {const int i=g*5+r;dst[i]=(i*17)%C;tok[i]=(i*11)%T;ids[dst[i]]=e;}
        }
        starts[E]=C;
        Device<int32_t> ds(E+1),dd(C),dtok(C),di(C),ng(1);
        Device<unsigned long long> dp(E);Device<float> scale(2*E),group_scale(2*C);
        ds.upload(starts);dd.upload(dst);dtok.upload(tok);di.upload(ids);dp.upload(ptr);scale.upload(scales);
        ck(cudaMemsetAsync(a.p,0,E*T*2560*4,st));ck(cudaMemsetAsync(b.p,0,E*T*2560*4,st));
        nvfp4_cutlass_prefill(L,refgroup,dtok.p,dx.p,newwork.p,b.p,st);
        ck(cudaStreamSynchronize(st));
        ng.upload({7});
        nvfp4_tensor_decode(L,dp.p,ds.p,ng.p,dd.p,dtok.p,di.p,scale.p,C,dx.p,oldwork.p,group_scale.p,a.p,st);
        ck(cudaStreamSynchronize(st));ng.upload({E-7});
        nvfp4_tensor_decode(L,dp.p+7,ds.p+7,ng.p,dd.p,dtok.p,di.p,scale.p,C,dx.p,oldwork.p,group_scale.p,a.p,st);
        ck(cudaStreamSynchronize(st));ng.upload({0});
        nvfp4_tensor_decode(L,dp.p,ds.p,ng.p,dd.p,dtok.p,di.p,scale.p,C,dx.p,oldwork.p,group_scale.p,a.p,st);
        ck(cudaStreamSynchronize(st));
        auto got=a.read(C*2560),ref=b.read(C*2560);double num=0,den=0;
        for(int i=0;i<C;++i)for(int k=0;k<2560;++k) {
            const double x=got[dst[i]*2560+k],y=ref[i*2560+k];
            if(!std::isfinite(x))throw std::runtime_error("nonfinite decode result");
            num+=(x-y)*(x-y);den+=y*y;
        }
        const double rel=std::sqrt(num/std::max(1e-30,den));
        if(rel>2e-3)throw std::runtime_error("decode routing/scales differ from CUTLASS");
        std::printf("decode plan vs CUTLASS relative L2 %.9g\n",rel);
#ifdef STRATA_NATIVE_EXPERTS
        namespace cpu=strata::kernels::cpu;
        if(cpu::cpu_avx512_ok()) {
        cpu::NativeFmt f{};
        f.gu_type=f.d_type=40;f.n_embd=2560;f.n_ff=640;f.gu_row=L.gu_row;f.d_row=L.d_row;
        f.up_off=L.up_off;f.down_off=L.down_off;f.tail_off=L.tail_off;f.bytes=L.bytes;f.fp32_activations=true;
        for(int e=0;e<E;++e)f.expert_input_scales.push_back({scales[2*e],scales[2*e],scales[2*e+1]});
        std::vector<cpu::ExpertJobMulti> jobs(E);std::vector<float> host(C*2560);
        for(int g=0;g<E;++g) {
            const int e=(g*7+3)%E;auto& j=jobs[g];j.expert_id=e;j.blob=weights.data()+e*L.bytes;j.nt=5;
            for(int t=0;t<5;++t) {int i=5*g+t;j.nact[t]=x.data()+tok[i]*2560;j.out[t]=host.data()+i*2560;}
        }
        {cpu::ExpertPool pool(3,false,true,cpu::PoolAffinity::All,40);pool.run_split_multi_native(f,jobs.data(),E);}
        double cn=0,cd=0,max_abs=0;
        for(size_t i=0;i<host.size();++i) {
            if(!std::isfinite(host[i]))throw std::runtime_error("nonfinite CPU result");
            double delta=double(host[i])-ref[i];cn+=delta*delta;cd+=double(ref[i])*ref[i];max_abs=std::max(max_abs,std::abs(delta));
        }
        double cr=std::sqrt(cn/std::max(1e-30,cd));
        std::printf("CPU W4A4 pool vs CUTLASS relative L2 %.9g max_abs %.9g\n",cr,max_abs);
        if(cr>2e-3)throw std::runtime_error("CPU W4A4 differs from CUTLASS beyond tolerance");
        // Diagnose arithmetic at the GU boundary and after down-input quantization.
        auto align256=[](size_t n){return (n+255)&~size_t(255);};
        const size_t gu_offset=align256(C*40*36),qh_offset=gu_offset+align256(C*1280*4);
        std::vector<float> gpu_gu(C*1280),cpu_gu(C*1280),qx(5*2560),qh(640),hidden(640);
        std::vector<unsigned char> gpu_qh(C*10*36);
        ck(cudaMemcpy(gpu_gu.data(),oldwork.p+gu_offset,gpu_gu.size()*4,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(gpu_qh.data(),oldwork.p+qh_offset,gpu_qh.size(),cudaMemcpyDeviceToHost));
        size_t gu_diff=0,qh_diff=0;double gu_num=0,gu_den=0;
        for(int g=0;g<E;++g) {
            int e=(g*7+3)%E;const auto* blob=weights.data()+e*L.bytes;
            const void* acts[5];float* gp[5];float* up[5];float ws[4];std::memcpy(ws,blob+L.tail_off,16);
            for(int t=0;t<5;++t) {int i=g*5+t;cpu::nvfp4_512_quantize_w4a4(x.data()+tok[i]*2560,qx.data()+t*2560,2560,scales[2*e]);
                acts[t]=qx.data()+t*2560;gp[t]=cpu_gu.data()+i*1280;up[t]=gp[t]+640;}
            cpu::nvfp4_512_f32_rows(blob,L.gu_row,2560,acts,5,gp,0,640,ws[0],scales[2*e]);
            cpu::nvfp4_512_f32_rows(blob+L.up_off,L.gu_row,2560,acts,5,up,0,640,ws[1],scales[2*e]);
            for(int t=0;t<5;++t) {
                const int i=g*5+t;
                for(int k=0;k<640;++k) {float v=gp[t][k];hidden[k]=(v/(1.f+float(std::exp(-double(v)))))*up[t][k];}
                cpu::nvfp4_512_quantize_w4a4(hidden.data(),qh.data(),640,scales[2*e+1]);
                for(int k=0;k<640;++k) {
                    const unsigned char* block=gpu_qh.data()+(i*10+k/64)*36;
                    uint32_t word;std::memcpy(&word,block+4+(k%64/8)*4,4);
                    float v=cpu::nvfp4_e2m1_value((word>>(4*(k%8)))&15)*cpu::nvfp4_e4m3_value(block[k%64/16]);
                    qh_diff+=std::memcmp(&v,&qh[k],4)!=0;
                }
            }
        }
        for(size_t i=0;i<cpu_gu.size();++i) {gu_diff+=std::memcmp(&cpu_gu[i],&gpu_gu[i],4)!=0;
            double d=double(cpu_gu[i])-gpu_gu[i];gu_num+=d*d;gu_den+=double(gpu_gu[i])*gpu_gu[i];}
        std::printf("CPU/GPU GU: %zu/%zu bit differences, relative L2 %.9g; down quant: %zu/%d differences\n",
            gu_diff,cpu_gu.size(),std::sqrt(gu_num/std::max(1e-30,gu_den)),qh_diff,C*640);
        const auto host_ref=host;
        // A token alone, different task boundaries: same CPU quantization and dot order.
        {cpu::ExpertPool pool(1,false,true,cpu::PoolAffinity::All,17);
            for(auto j:jobs)for(int t=0;t<j.nt;++t) {auto one=j;one.nt=1;one.nact[0]=j.nact[t];one.out[0]=j.out[t];pool.run_split_multi_native(f,&one,1);}}
        if(std::memcmp(host_ref.data(),host.data(),host.size()*4))throw std::runtime_error("CPU W4A4 depends on token grouping");
        }
#endif

    }
    std::printf("groups,max_rows,skewed,relative_l2,custom_us,cutlass_us,speedup\n");
    for(int skew:{0,1,2})for(int count:{1,4,16})for(int rows:{1,8,16,32,37,64,80,129,513}) {
        // Vary calibration independently of pointer order, token routing and
        // output offsets; compare the batch against one-expert launches too.
        if(skew==2 && (count!=16 || rows!=129))continue;
        if(skew && (count==1 || rows<129))continue;
        if(initcheck && (skew || (count!=1 && count!=16) || rows!=1))continue;
        Nvfp4PrefillGroup g{};g.count=count;
        for(int i=0;i<count;++i) {g.weights[i]=dw.p+((i*7+3)%E)*L.bytes;g.offsets[i]=i*T;g.rows[i]=skew && i>0?1:std::max(1,rows-i%3);}
        if(skew==2)for(int i=0;i<count;++i) {
            g.input_scale_gu[i]=L.input_scale_gu*(.5f+float(i)/13);
            g.input_scale_down[i]=L.input_scale_down*(.4f+float(i)/7);
        }
        ck(cudaMemsetAsync(a.p,0,E*T*2560*4,st));ck(cudaMemsetAsync(b.p,0,E*T*2560*4,st));
        auto old=[&]{nvfp4_tensor_prefill_group(L,g,dt.p,dx.p,oldwork.p,a.p,st);};
        auto official=[&]{nvfp4_cutlass_prefill(L,g,dt.p,dx.p,newwork.p,b.p,st);};
        old();official();ck(cudaStreamSynchronize(st));
        auto ref=a.read(E*T*2560),got=b.read(E*T*2560);double num=0,den=0;
        for(size_t i=0;i<ref.size();++i) {
            if(!std::isfinite(got[i]))throw std::runtime_error("nonfinite CUTLASS output");
            num+=double(got[i]-ref[i])*(got[i]-ref[i]);den+=double(ref[i])*ref[i];
        }
        double rel=std::sqrt(num/std::max(1e-30,den));
        if(rel>2e-3) {std::printf("FAIL %d %d relative L2 %.9g\n",count,rows,rel);return 1;}
        if(skew==2) {
            ck(cudaMemsetAsync(b.p,0,E*T*2560*4,st));
            for(int i=0;i<count;++i) {
                auto single=L;single.input_scale_gu=g.input_scale_gu[i];single.input_scale_down=g.input_scale_down[i];
                nvfp4_tensor_prefill(single,g.weights[i],dt.p+g.offsets[i],g.rows[i],dx.p,oldwork.p,b.p+size_t(g.offsets[i])*2560,st);
            }
            ck(cudaStreamSynchronize(st));auto individual=b.read(E*T*2560);
            if(std::memcmp(individual.data(),ref.data(),ref.size()*sizeof(float)))
                throw std::runtime_error("per-expert scale batch differs from individual calibrated launches");
        }
        double old_us=check_only?0:timing(old,st),new_us=check_only?0:timing(official,st);
        std::printf("%d,%d,%d,%.9g,%.3f,%.3f,%.3f\n",count,rows,skew,rel,old_us,new_us,new_us>0?old_us/new_us:0);std::fflush(stdout);
    }
    ck(cudaStreamDestroy(st));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
