#include "strata/kernels/mtp_fp8.hpp"
#include "strata/weights/safetensors.hpp"
#include "strata/weights/nvfp4.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata;
void ck(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
double decode(uint8_t v) {
    const int e=(v>>3)&15,m=v&7;
    const double a=e?std::ldexp(1.0+m/8.0,e-7):std::ldexp(double(m),-9);
    return v&128?-a:a;
}
double dot(const uint8_t* w,const float* x,int r,int cols) {
    double y=0;
    for(int c=0;c<cols;++c) {
        uint16_t bits;
        std::memcpy(&bits,w+kernels::kMtpFp8Matrix+2*((r/128)*(cols/128)+c/128),2);
        y+=decode(w[r*cols+c])*std::bit_cast<float>(uint32_t(bits)<<16)*x[c];
    }
    return y;
}
int run(const std::filesystem::path& path,bool nvfp4=false) {
    try {
        std::unique_ptr<weights::SafetensorsSource> source;
        if(!path.empty()) source=std::make_unique<weights::SafetensorsSource>(path);
        constexpr int E=2,T=2,K=2;
        if (source) nvfp4=source->tensor("mtp.layers.0.mlp.experts.0.gate_proj.weight").dtype==weights::DType::U8;
        const auto expert_bytes=nvfp4?kernels::kMtpNvfp4Expert:kernels::kMtpFp8Expert;
        std::vector<uint8_t> packed(E*expert_bytes);
        std::vector<float> reference_weights(nvfp4?E*3*kernels::kMtpFp8Matrix:0);
        std::mt19937 rng(120);
        const char* names[]={"gate_proj","up_proj","down_proj"};
        for(int e=0;e<E;++e) for(int p=0;p<3;++p) {
            auto* dst=packed.data()+e*expert_bytes+p*(nvfp4?kernels::kMtpNvfp4Projection:kernels::kMtpFp8Projection);
            if (nvfp4) {
                const int rows=p==2?2560:640,cols=p==2?640:2560;
                std::vector<uint8_t> w(rows*cols/2),s(rows*cols/16);
                float global=std::ldexp(float(e+p+1),-10);
                if (source) {
                    const std::string stem="mtp.layers.0.mlp.experts."+std::to_string(e?511:0)+"."+names[p];
                    w=source->read(source->tensor(stem+".weight"));s=source->read(source->tensor(stem+".weight_scale"));
                    const auto g=source->read(source->tensor(stem+".weight_scale_2"));std::memcpy(&global,g.data(),4);
                } else {
                    for (auto& v:w) v=uint8_t(rng());
                    for (auto& v:s) v=uint8_t(32+rng()%40);
                }
                weights::pack_nvfp4(w,s,rows,cols,{dst,kernels::kMtpNvfp4Projection});
                std::memcpy(packed.data()+e*expert_bytes+3*kernels::kMtpNvfp4Projection+p*4,&global,4);
                constexpr float values[]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
                for (int i=0;i<rows*cols;++i)
                    reference_weights[(e*3+p)*kernels::kMtpFp8Matrix+i]=float(values[(w[i/2]>>(4*(i%2)))&15]*decode(s[i/16]))*global;
            } else if(source) {
                const std::string stem="mtp.layers.0.mlp.experts."+std::to_string(e?511:0)+"."+names[p];
                auto w=source->read(source->tensor(stem+".weight")),s=source->read(source->tensor(stem+".weight_scale_inv"));
                std::memcpy(dst,w.data(),w.size()); std::memcpy(dst+kernels::kMtpFp8Matrix,s.data(),s.size());
            } else {
                for(uint64_t i=0;i<kernels::kMtpFp8Matrix;++i) { auto v=uint8_t(rng()); if((v&127)==127) v=0; dst[i]=v; }
                for(int i=0;i<100;++i) { uint16_t s=uint16_t(0x3600+(rng()%384)); std::memcpy(dst+kernels::kMtpFp8Matrix+i*2,&s,2); }
            }
        }
        std::vector<float> x(T*2560),h(T*K*640),y(T*K*2560),ref(y.size());
        for(auto& v:x) v=(int(rng()%2001)-1000)*0.001f;
        const int32_t ids[]={0,1,1,0};
        uint8_t* dw=nullptr; float *dx=nullptr,*dh=nullptr,*dy=nullptr; int32_t* di=nullptr;
        ck(cudaMalloc(&dw,packed.size())); ck(cudaMalloc(&dx,x.size()*4)); ck(cudaMalloc(&dh,h.size()*4)); ck(cudaMalloc(&dy,y.size()*4)); ck(cudaMalloc(&di,sizeof(ids)));
        ck(cudaMemcpy(dw,packed.data(),packed.size(),cudaMemcpyHostToDevice)); ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice)); ck(cudaMemcpy(di,ids,sizeof(ids),cudaMemcpyHostToDevice));
        if(nvfp4) kernels::mtp_nvfp4_experts(dw,di,dx,dh,dy,T,K,nullptr);
        else kernels::mtp_fp8_experts(dw,di,dx,dh,dy,T,K,nullptr);
        ck(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<T*K;++i) {
            const auto* w=packed.data()+ids[i]*expert_bytes;
            auto dot_projection=[&](int p,const float* input,int r,int cols) {
                if (!nvfp4) return dot(w+p*kernels::kMtpFp8Projection,input,r,cols);
                const auto* values=reference_weights.data()+(ids[i]*3+p)*kernels::kMtpFp8Matrix+r*cols;
                double sum=0;for(int c=0;c<cols;++c) sum+=double(values[c])*input[c];return sum;
            };
            // FP64 dot-products, with the ABI's FP32 intermediate storage.
            for(int r=0;r<640;++r) {
                const double g=dot_projection(0,x.data()+(i/K)*2560,r,2560),u=dot_projection(1,x.data()+(i/K)*2560,r,2560);
                h[i*640+r]=float(g/(1+std::exp(-g))*u);
            }
            for(int r=0;r<2560;++r) ref[i*2560+r]=float(dot_projection(2,h.data()+i*640,r,640));
        }
        double se=0,sr=0,mx=0,peak=0;
        for(size_t i=0;i<y.size();++i) {
            if(!std::isfinite(y[i])) throw std::runtime_error("nonfinite output");
            const double d=y[i]-ref[i]; se+=d*d; sr+=double(ref[i])*ref[i]; mx=std::max(mx,std::abs(d)); peak=std::max(peak,std::abs(double(ref[i])));
        }
        const double rel=std::sqrt(se/std::max(sr,1e-30));
        std::printf("{\"source\":\"%s\",\"outputs\":%zu,\"relative_l2\":%.9g,\"max_abs\":%.9g,\"scaled_max\":%.9g}\n",source?"checkpoint":(nvfp4?"synthetic NVFP4":"synthetic FP8"),y.size(),rel,mx,mx/std::max(peak,1e-30));
        cudaFree(dw);cudaFree(dx);cudaFree(dh);cudaFree(dy);cudaFree(di);
        return rel<5e-5 && mx/std::max(peak,1e-30)<1e-4?0:1;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
#if defined(_WIN32)
int wmain(int argc,wchar_t** argv) { return argc==2?run(std::filesystem::path(argv[1])):(run({}) || run({},true)); }
#else
int main(int argc,char** argv) { return argc==2?run(std::filesystem::path(argv[1])):(run({}) || run({},true)); }
#endif
