// Original packed-weight adapter parity and same-process complete expert timing.
#include "strata/kernels/nvfp4_cutlass.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
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
int main(int argc,char** argv) {
try {
    const bool initcheck=argc==2 && std::strcmp(argv[1],"--initcheck")==0;
    const bool check_only=initcheck || (argc==2 && std::strcmp(argv[1],"--check-only")==0);
    if(argc>1 && !check_only)throw std::runtime_error("usage: nvfp4_cutlass_test [--check-only|--initcheck]");
    const char* backend=std::getenv("STRATA_NVFP4_PREFILL_BACKEND");
    if(backend && *backend && std::strcmp(backend,"custom")!=0)
        throw std::runtime_error("parity test requires STRATA_NVFP4_PREFILL_BACKEND unset or custom");
    NativeExpertLayout L{};L.gu_type=L.d_type=40;L.n_embd=2560;L.n_ff=640;
    L.gu_row=40*36;L.d_row=10*36;L.up_off=640*L.gu_row;L.down_off=2*L.up_off;
    L.tail_off=L.down_off+2560*L.d_row;L.bytes=L.tail_off+16;
    L.input_scale_gu=.003f;L.input_scale_down=.002f;
    constexpr int E=16,T=513;
    std::vector<unsigned char> weights(E*L.bytes);
    for(int e=0;e<E;++e) {
        auto* p=weights.data()+e*L.bytes;
        for(size_t i=0;i<L.tail_off;i+=36) {
            for(int s=0;s<4;++s)p[i+s]=uint8_t(28+(i/36+s+e)%6);
            for(int j=0;j<32;++j)p[i+4+j]=uint8_t((i/36*17+j*11+e*13)%256);
        }
        const float scales[]={.03f,.04f,.05f,0};std::memcpy(p+L.tail_off,scales,16);
    }
    std::vector<float> x(T*2560);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.073f);
    std::vector<int32_t> tokens(E*T);for(int i=0;i<E*T;++i)tokens[i]=(i*17)%T;
    Device<uint8_t> dw(weights.size()),oldwork(nvfp4_tensor_prefill_bytes(T)),newwork(nvfp4_cutlass_prefill_bytes(T));
    Device<float> dx(x.size()),a(E*T*2560),b(E*T*2560);Device<int32_t> dt(tokens.size());
    if(initcheck) {std::printf("CUTLASS scratch %p bytes %zu\n",static_cast<void*>(newwork.p),nvfp4_cutlass_prefill_bytes(T));std::fflush(stdout);}
    dw.upload(weights);dx.upload(x);dt.upload(tokens);
    cudaStream_t st;ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    std::printf("groups,max_rows,skewed,relative_l2,custom_us,cutlass_us,speedup\n");
    for(int skew:{0,1})for(int count:{1,4,16})for(int rows:{1,8,16,32,37,64,80,129,513}) {
        if(skew && (count==1 || rows<129))continue;
        if(initcheck && (skew || (count!=1 && count!=16) || rows!=1))continue;
        Nvfp4PrefillGroup g{};g.count=count;
        for(int i=0;i<count;++i) {g.weights[i]=dw.p+((i*7+3)%E)*L.bytes;g.offsets[i]=i*T;g.rows[i]=skew && i>0?1:std::max(1,rows-i%3);}
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
        double old_us=check_only?0:timing(old,st),new_us=check_only?0:timing(official,st);
        std::printf("%d,%d,%d,%.9g,%.3f,%.3f,%.3f\n",count,rows,skew,rel,old_us,new_us,new_us>0?old_us/new_us:0);std::fflush(stdout);
    }
    ck(cudaStreamDestroy(st));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
