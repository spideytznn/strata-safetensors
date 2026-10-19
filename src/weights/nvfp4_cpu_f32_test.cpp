#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef STRATA_TEST_GPU
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
namespace {
void cuda_ok(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void compare_gpu(const std::vector<uint8_t>& blob,const std::vector<float>& x,const std::vector<float>& cpu) {
    namespace k=strata::kernels;
    const auto L=k::native_expert_layout(40,40,2560,640);
    struct Buffers { std::vector<void*> p; ~Buffers(){for(auto v:p)cudaFree(v);} } buffers;
    auto alloc=[&](size_t n,const void* src=nullptr) {
        void* p=nullptr;cuda_ok(cudaMalloc(&p,n));buffers.p.push_back(p);
        if(src)cuda_ok(cudaMemcpy(p,src,n,cudaMemcpyHostToDevice));return p;
    };
    auto* weights=alloc(blob.size(),blob.data());
    auto* input=static_cast<float*>(alloc(x.size()*4,x.data()));
    auto* output=static_cast<float*>(alloc(cpu.size()*4));
    auto* scratch=alloc(k::native_expert_scratch_bytes(4,640));
    const unsigned long long ptr=reinterpret_cast<unsigned long long>(weights);
    const int32_t starts[]={0,4},count=1,dst[]={3,1,0,2},tok[]={2,0,3,1};
    auto* dp=static_cast<unsigned long long*>(alloc(8,&ptr));
    auto* ds=static_cast<int32_t*>(alloc(8,starts));
    auto* dn=static_cast<int32_t*>(alloc(4,&count));
    auto* dd=static_cast<int32_t*>(alloc(16,dst));
    auto* dt=static_cast<int32_t*>(alloc(16,tok));
    k::native_expert_grouped(L,dp,ds,dn,dd,dt,1,4,nullptr,scratch,output,nullptr,1,input);
    std::vector<float> got(cpu.size());cuda_ok(cudaMemcpy(got.data(),output,got.size()*4,cudaMemcpyDeviceToHost));
    size_t different=0;float max_abs=0;
    for(int i=0;i<4;++i)for(int j=0;j<2560;++j) {
        const float a=got[dst[i]*2560+j],b=cpu[tok[i]*2560+j];
        different+=std::memcmp(&a,&b,4)!=0;max_abs=std::max(max_abs,std::abs(a-b));
    }
    std::printf("CPU/GPU canonical: %zu / %zu different, max_abs %.9g\n",different,cpu.size(),max_abs);
    if(different)throw std::runtime_error("CPU/GPU canonical expert output differs");
    const int32_t single[]={0,1};cuda_ok(cudaMemcpy(ds,single,8,cudaMemcpyHostToDevice));
    for(int i=0;i<4;++i) {
        cuda_ok(cudaMemcpy(dd,dst+i,4,cudaMemcpyHostToDevice));
        cuda_ok(cudaMemcpy(dt,tok+i,4,cudaMemcpyHostToDevice));
        k::native_expert_grouped(L,dp,ds,dn,dd,dt,1,4,nullptr,scratch,output,nullptr,1,input);
    }
    std::vector<float> singles(cpu.size());cuda_ok(cudaMemcpy(singles.data(),output,singles.size()*4,cudaMemcpyDeviceToHost));
    if(std::memcmp(got.data(),singles.data(),got.size()*4))throw std::runtime_error("GPU T=1/4 output differs");
}
}
#endif

namespace c=strata::kernels::cpu;
int main(int argc,char** argv) {
    try {
#ifdef STRATA_TEST_GPU
        for(const char* name:{"STRATA_NVFP4_F32","STRATA_NVFP4_F32_CANONICAL","STRATA_NVFP4_F32_UNROLL"}) {
#ifdef _WIN32
            _putenv_s(name,"1");
#else
            setenv(name,"1",1);
#endif
        }
#endif
        if(argc<2 || argc>3) throw std::runtime_error("usage: nvfp4_cpu_f32_test experts.bin|single-expert.bin [output-bits.bin]");
        c::NativeFmt fmt; std::string err;
        if(!c::native_fmt(40,40,2560,640,fmt,err)) throw std::runtime_error(err);
        fmt.fp32_activations=true; fmt.act_bytes=2560*4; fmt.h_bytes=640*4;
        constexpr int T=4,N=2560,F=640;
        std::ifstream input(argv[1],std::ios::binary);
        std::ofstream dump;
        if(argc==3) { dump.open(argv[2],std::ios::binary);if(!dump) throw std::runtime_error("output open failed"); }
        std::mt19937 rng(9950); std::normal_distribution<float> random;
        double worst=0;
        const std::vector<int> experts=std::filesystem::file_size(argv[1])==fmt.bytes
            ?std::vector<int>{0}:std::vector<int>{0,47*512+511};
        for(int expert:experts) {
            std::vector<uint8_t> b(fmt.bytes);
            input.seekg(uint64_t(expert)*fmt.bytes); input.read(reinterpret_cast<char*>(b.data()),b.size());
            if(!input) throw std::runtime_error("expert read failed");
            std::vector<float> x(T*N),h(T*F),y(T*N),one(T*N),wg(F*N),wu(F*N),wd(N*F);
            for(auto& v:x) v=random(rng);
            const auto* tr=ggml_get_type_traits(GGML_TYPE_NVFP4);
            tr->to_float(b.data(),wg.data(),F*N); tr->to_float(b.data()+fmt.up_off,wu.data(),F*N);
            tr->to_float(b.data()+fmt.down_off,wd.data(),N*F);
            const void* a[T]; const void* hp[T]; float* hs[T]; float* ys[T];
            for(int t=0;t<T;++t) { a[t]=x.data()+t*N; hp[t]=hs[t]=h.data()+t*F; ys[t]=y.data()+t*N; }
            std::vector<uint8_t> copy(c::kNativeActBytes); c::native_quant_act(fmt,x.data(),copy.data());
            if(std::memcmp(copy.data(),x.data(),N*4)) throw std::runtime_error("input rounded");
            c::native_gu_rows(fmt,b.data(),a,T,hs,0,F); c::native_down_rows(fmt,b.data(),hp,T,ys,0,N);
            for(int t=0;t<T;++t) {
                float* dest=one.data()+t*N;
                c::native_gu_rows(fmt,b.data(),a+t,1,hs+t,0,F); c::native_down_rows(fmt,b.data(),hp+t,1,&dest,0,N);
            }
            if(std::memcmp(y.data(),one.data(),y.size()*4)) throw std::runtime_error("token grouping changed output");
#ifdef STRATA_TEST_GPU
            compare_gpu(b,x,y);
#endif
            if(dump.is_open()) {
                dump.write(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(float));
                dump.write(reinterpret_cast<const char*>(y.data()),y.size()*sizeof(float));
                if(!dump) throw std::runtime_error("output write failed");
            }
            float scales[4]; std::memcpy(scales,b.data()+fmt.tail_off,16);
            double se=0,sr=0;
            for(int t=0;t<T;++t) {
                std::vector<double> hidden(F);
                for(int r=0;r<F;++r) {
                    double g=0,u=0;
                    for(int col=0;col<N;++col) { g+=double(wg[r*N+col])*x[t*N+col]; u+=double(wu[r*N+col])*x[t*N+col]; }
                    g*=scales[0]; hidden[r]=g/(1+std::exp(-g))*(u*scales[1]);
                }
                for(int r=0;r<N;++r) {
                    double sum=0; for(int col=0;col<F;++col) sum+=double(wd[r*F+col])*hidden[col];
                    sum*=scales[2]; const double d=y[t*N+r]-sum;
                    if(!std::isfinite(d)) throw std::runtime_error("nonfinite output");
                    se+=d*d; sr+=sum*sum;
                }
            }
            const double relative=std::sqrt(se/sr); worst=std::max(worst,relative);
            std::printf("expert %d relative_l2 %.9g; T=1/4 bit-identical\n",expert,relative);
        }
        return worst<1e-4?0:1;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
