#include "strata/prefill/gemm.hpp"
#include "strata/kernels/nvfp4_w4a16.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace strata;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void need(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& v) { ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) { std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v; }
};
float as_float(uint32_t u) { float f;std::memcpy(&f,&u,4);return f; }
uint16_t round_bf(float f) {
    uint32_t u;std::memcpy(&u,&f,4);
    return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
float rounded(float f) { return as_float(uint32_t(round_bf(f))<<16); }
float weight(const uint8_t* p,int i) {
    static const double table[]={0,.5,1,1.5,2,3,4,6};
    const auto* b=p+(i/64)*36;
    int sub=(i%64)/16,col=i%16;
    unsigned c=(b[4+sub*8+col%8]>>(col<8?0:4))&15,s=b[sub];
    const double scale=(s>>3)?std::ldexp(double(8+(s&7)),int(s>>3)-10):double(s&7)/512;
    return float((c&8?-1:1)*table[c&7]*scale);
}
int main() try {
    kernels::NativeExpertLayout L{};L.gu_type=L.d_type=40;L.n_embd=2560;L.n_ff=640;
    L.gu_row=40*36;L.d_row=10*36;L.up_off=640*L.gu_row;L.down_off=2*L.up_off;
    L.tail_off=L.down_off+2560*L.d_row;L.bytes=L.tail_off+16;
    L.input_scale_gu=.003f;L.input_scale_down=.002f; // A16 does not quantize using these A4 scales.
    constexpr int T=41,E=3,OUT=55,GU=1280,N=2560,F=640;
    std::vector<uint8_t> weights(E*L.bytes);
    for(int e=0;e<E;++e) {
        auto* p=weights.data()+e*L.bytes;
        for(size_t b=0;b<L.tail_off;b+=36) {
            for(int s=0;s<4;++s)p[b+s]=uint8_t((b/36+s*7+e*3)%127);
            for(int j=0;j<32;++j)p[b+4+j]=uint8_t((b/36*17+j*11+e*13)%256);
        }
        const float scales[]={.001f*(e+1),.0017f*(e+1),.0023f*(e+1),0};
        std::memcpy(p+L.tail_off,scales,16);
    }
    std::vector<float> x(T*N);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.073f)*.23f;
    std::vector<int32_t> ids(OUT);for(int i=0;i<OUT;++i)ids[i]=(i*17+3)%T;
    std::vector<float> init(OUT*N,-123.25f);
    Device<uint8_t> dw(weights.size()),work(kernels::nvfp4_w4a16_prefill_bytes(T)+256);
    Device<float> dx(x.size()),dy(init.size());Device<int32_t> di(ids.size());
    dw.put(weights);dx.put(x);di.put(ids);dy.put(init);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    prefill::Gemm gemm;std::string error;need(gemm.init(stream,3*2560*41,error),error.c_str());
    kernels::Nvfp4PrefillGroup group;group.count=E;
    const int offsets[]={2,7,13},rows[]={1,3,37},experts[]={2,0,1};
    for(int e=0;e<E;++e){group.weights[e]=dw.p+experts[e]*L.bytes;group.offsets[e]=offsets[e];group.rows[e]=rows[e];}
    ck(cudaMemset(work.p+kernels::nvfp4_w4a16_prefill_bytes(T),0xa5,256));
    // Inspect the production preparation against a separately decoded CPU reference.
    const auto b=kernels::nvfp4_w4a16_buffers(work.p,T);
    kernels::nvfp4_w4a16_prepare(L,group.weights[0],di.p+offsets[0],rows[0],dx.p,b,stream);
    ck(cudaStreamSynchronize(stream));
    std::vector<uint16_t> expanded(GU*N+N*F),gathered(N);
    ck(cudaMemcpy(expanded.data(),b.gate_up,GU*N*2,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(expanded.data()+GU*N,b.down,N*F*2,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(gathered.data(),b.input,N*2,cudaMemcpyDeviceToHost));
    const auto* wp=weights.data()+experts[0]*L.bytes;
    for(int i=0;i<F*N;++i) {
        need(expanded[i]==round_bf(weight(wp,i)),"gate dequant mismatch");
        need(expanded[F*N+i]==round_bf(weight(wp+L.up_off,i)),"up dequant mismatch");
        need(expanded[GU*N+i]==round_bf(weight(wp+L.down_off,i)),"down dequant mismatch");
    }
    for(int c=0;c<N;++c)need(gathered[c]==round_bf(x[ids[offsets[0]]*N+c]),"gather mismatch");
    gemm.nvfp4_w4a16(L,group,di.p,dx.p,work.p,dy.p);ck(cudaStreamSynchronize(stream));
    const auto got=dy.get(init.size());double num=0,den=0,max_error=0;
    std::vector<bool> written(OUT,false);
    for(int e=0;e<E;++e) {
        wp=weights.data()+experts[e]*L.bytes;float scale[4];std::memcpy(scale,wp+L.tail_off,16);
        std::vector<float> guw(GU*N),dnw(N*F);
        for(int i=0;i<F*N;++i){guw[i]=weight(wp,i);guw[F*N+i]=weight(wp+L.up_off,i);dnw[i]=weight(wp+L.down_off,i);}
        for(int r=0;r<rows[e];++r) {
            const int dest=offsets[e]+r;written[dest]=true;std::vector<float> h(F);
            for(int j=0;j<F;++j) {
                double g=0,u=0;
                for(int c=0;c<N;++c){double a=rounded(x[ids[dest]*N+c]);g+=guw[j*N+c]*a;u+=guw[(j+F)*N+c]*a;}
                const float gf=float(g)*scale[0],uf=float(u)*scale[1];
                h[j]=rounded((gf/(1.f+std::exp(-gf)))*uf);
            }
            for(int j=0;j<N;++j) {
                double sum=0;for(int c=0;c<F;++c)sum+=dnw[j*F+c]*double(h[c]);
                const double expected=float(sum)*scale[2],actual=got[dest*N+j],delta=actual-expected;
                need(std::isfinite(actual),"nonfinite output");num+=delta*delta;den+=expected*expected;max_error=std::max(max_error,std::abs(delta));
            }
        }
    }
    for(int r=0;r<OUT;++r)if(!written[r])for(int c=0;c<N;++c)need(got[r*N+c]==init[r*N+c],"inactive output overwritten");
    std::vector<uint8_t> guard(256);ck(cudaMemcpy(guard.data(),work.p+kernels::nvfp4_w4a16_prefill_bytes(T),256,cudaMemcpyDeviceToHost));
    need(std::all_of(guard.begin(),guard.end(),[](uint8_t v){return v==0xa5;}),"workspace overrun");
    const double rel=std::sqrt(num/std::max(den,1e-30));need(rel<.002,"W4A16 FP64 oracle error too large");
    // The warmed production GEMMs and preparation must also survive graph replay.
    cudaGraph_t graph;cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));gemm.nvfp4_w4a16(L,group,di.p,dx.p,work.p,dy.p);
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));need(dy.get(init.size())==got,"graph replay changed output");
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
    bool rejected=false;auto invalid=group;invalid.rows[0]=0;
    try{gemm.nvfp4_w4a16(L,invalid,di.p,dx.p,work.p,dy.p);}catch(const std::invalid_argument&){rejected=true;}
    need(rejected,"invalid group accepted");
    // Exercise the exact desktop workspace size; zero input has an exact zero oracle.
    constexpr int BIG=8192;
    Device<uint8_t> bigwork(kernels::nvfp4_w4a16_prefill_bytes(BIG)+256);
    Device<int32_t> bigids(BIG+2);Device<float> bigout(size_t(BIG+2)*N);
    std::vector<int32_t> zeros(BIG+2,0);bigids.put(zeros);
    std::vector<float> marked(size_t(BIG+2)*N,-123.25f);bigout.put(marked);
    std::fill(x.begin(),x.end(),0);dx.put(x);
    ck(cudaMemset(bigwork.p+kernels::nvfp4_w4a16_prefill_bytes(BIG),0xa5,256));
    auto large=group;large.count=1;large.rows[0]=BIG;large.offsets[0]=1;
    gemm.nvfp4_w4a16(L,large,bigids.p,dx.p,bigwork.p,bigout.p);ck(cudaStreamSynchronize(stream));
    auto large_y=bigout.get(marked.size());
    for(int r=0;r<BIG+2;++r)for(int c=0;c<N;++c)
        need(large_y[size_t(r)*N+c]==(r==0 || r==BIG+1?-123.25f:0.f),"8192-row output/guard mismatch");
    ck(cudaMemcpy(guard.data(),bigwork.p+kernels::nvfp4_w4a16_prefill_bytes(BIG),256,cudaMemcpyDeviceToHost));
    need(std::all_of(guard.begin(),guard.end(),[](uint8_t v){return v==0xa5;}),"8192-row workspace overrun");
    std::printf("PASS W4A16: 3 reordered experts / 1,3,37 rows; all finite nonnegative E4M3 scale codes; exact BF16 preparation, FP64 oracle relative_l2=%.9g max_abs=%.9g, inactive rows, workspace guards, graph replay, 8192-row zero oracle\n",rel,max_error);
    ck(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
