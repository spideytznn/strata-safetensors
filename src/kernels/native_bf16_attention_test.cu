#include "strata/kernels/kv_bf16_tc.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::kernels;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
uint16_t round_bf(float x){uint32_t u;std::memcpy(&u,&x,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
float from_bf(uint16_t x){uint32_t u=uint32_t(x)<<16;float v;std::memcpy(&v,&u,4);return v;}
template<class T>struct Device {
    T* p=nullptr;size_t n;
    explicit Device(size_t count):n(count){ck(cudaMalloc((void**)&p,n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    void put(const std::vector<T>& v){ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
    std::vector<T> pages(int cells,int row_width,bool reversed){
        std::vector<T> v(n);const int full=cells/4,tail=cells%4;
        const size_t start=tail&&reversed?8*row_width:0;
        if(full)ck(cudaMemcpy(v.data()+start,p+start,size_t(full)*8*row_width*sizeof(T),cudaMemcpyDeviceToHost));
        if(tail)for(int h=0;h<2;++h){size_t at=((reversed?0:full)*8+h*4)*row_width;
            ck(cudaMemcpy(v.data()+at,p+at,size_t(tail)*row_width*sizeof(T),cudaMemcpyDeviceToHost));}
        return v;
    }
};
int main(int argc,char** argv)try {
    const bool check_only=argc>1 && std::string(argv[1])=="--check-only";
#ifdef _WIN32
    _putenv_s("STRATA_KV_BF16_TC","1");
#else
    setenv("STRATA_KV_BF16_TC","1",1);
#endif
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    QsaShapes s{};s.n_head=24;s.n_head_kv=2;s.head_dim=256;s.page_size=4;
    std::puts("queries,cells,api,us,max_oracle_error");
    for(int nq:{1,3,8,32})for(int cells:{5,65,2051}) {
        const int pages=(cells+3)/4,rows=pages*8;
        std::vector<float> k(cells*512),v(k.size()),q(nq*24*256),qr(q.size());
        for(size_t i=0;i<k.size();++i){k[i]=std::sin(float(i)*.13f)*3;v[i]=std::cos(float(i)*.071f)*.7f;}
        std::fill(k.begin(),k.begin()+64,0.0f);std::fill(v.begin(),v.begin()+64,1e-9f);
        for(size_t i=0;i<q.size();++i){q[i]=std::sin(float(i)*.031f)*.15f;qr[i]=from_bf(round_bf(q[i]));}
        std::vector<int32_t> pt(pages),as(cells*kStepCount),steps(nq*kStepCount),ids(nq*cells);
        for(int i=0;i<pages;++i)pt[i]=pages-1-i;
        for(int i=0;i<cells;++i){as[i*kStepCount+kStepPos]=i;as[i*kStepCount+kStepNKv]=i+1;}
        for(int j=0;j<nq;++j){steps[j*kStepCount+kStepNKv]=cells;steps[j*kStepCount+kStepWidth]=cells-j%3;
            for(int i=0;i<cells;++i)ids[j*cells+i]=(i+j*13)%cells;}
        Device<float> dk(k.size()),dv(v.size()),dq(q.size()),out(q.size()),scratch(nq*qsa_decode_attn_scratch_floats(cells,s));
        Device<uint16_t> kp(rows*256),vp(rows*256),hk(rows*256),hv(rows*256);
        Device<int32_t> dpt(pt.size()),das(as.size()),ds(steps.size()),di(ids.size());
        dk.put(k);dv.put(v);dq.put(q);dpt.put(pt);das.put(as);ds.put(steps);di.put(ids);
        KvHostPools host{};host.k_pool=hk.p;host.v_pool=hv.p;
        kv_append_bf16_steps(kp.p,vp.p,dpt.p,das.p,kStepCount,dk.p,dv.p,512,cells,s,stream,&host);
        ck(cudaStreamSynchronize(stream));
        auto kb=kp.pages(cells,256,true),vb=vp.pages(cells,256,true);
        auto kh=hk.pages(cells,256,false),vh=hv.pages(cells,256,false);
        for(int t=0;t<cells;++t)for(int h=0;h<2;++h)for(int d=0;d<256;++d){
            int row=(pt[t/4]*2+h)*4+t%4,hr=(t/4*2+h)*4+t%4,src=t*512+h*256+d;
            if(kb[row*256+d]!=round_bf(k[src]) || vb[row*256+d]!=round_bf(v[src]) ||
               kb[row*256+d]!=kh[hr*256+d] || vb[row*256+d]!=vh[hr*256+d])throw std::runtime_error("BF16 append or shadow oracle mismatch");
        }
        if(pages>1)pt[1]=-1;dpt.put(pt);
        QsaAttnPools pool{};pool.k_pool=kp.p;pool.v_pool=vp.p;pool.page_table=dpt.p;pool.bf16=true;
        for(int api=0;api<3;++api){if(api==1&&nq!=1)continue;
            auto launch=[&]{if(api==2)qsa_bf16_tc_prompt(dq.p,pool,di.p,ds.p,cells,s,out.p,nq,stream);
                else if(api==1)qsa_decode_attn_step(dq.p,pool,di.p,ds.p,cells,s,scratch.p,out.p,stream);
                else qsa_decode_attn_batch(dq.p,pool,di.p,ds.p,cells,s,scratch.p,out.p,nq,stream);};
            cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            launch();ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));auto got=out.get();
            for(float f:got)if(!std::isfinite(f))throw std::runtime_error("nonfinite BF16 attention");
            double error=0;
            for(int j=0;j<nq;++j)for(int head:{0,11,12,23}) {
                const int width=steps[j*kStepCount+kStepWidth];std::vector<double> scores(width);double mx=-1e300;
                for(int c=0;c<width;++c){int cell=ids[j*cells+c],page=pt[cell/4];double dot=0;scores[c]=-1e300;
                    if(page>=0){int row=(page*2+head/12)*4+cell%4;
                        for(int d=0;d<256;++d)dot+=double(qr[(j*24+head)*256+d])*from_bf(kb[row*256+d]);scores[c]=dot/16;mx=std::max(mx,scores[c]);}}
                double den=0;for(double& f:scores){f=std::exp(f-mx);den+=f;}
                for(int d:{0,63,128,255}){double ref=0;for(int c=0;c<width;++c){int cell=ids[j*cells+c],page=pt[cell/4];
                        if(page>=0){int row=(page*2+head/12)*4+cell%4;ref+=scores[c]*from_bf(vb[row*256+d]);}}
                    error=std::max(error,std::abs(got[(j*24+head)*256+d]-ref/den));}
            }
            if(error>0.003)throw std::runtime_error("BF16 CPU attention oracle mismatch: "+std::to_string(error));
            auto before=got;for(int j=0;j<nq;++j)steps[j*kStepCount+kStepWidth]=1;ds.put(steps);
            ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));got=out.get();
            for(int j=0;j<nq;++j)for(int h=0;h<24;++h)for(int d=0;d<256;++d){int cell=ids[j*cells],page=pt[cell/4];
                float expected=page<0?0:from_bf(vb[((page*2+h/12)*4+cell%4)*256+d]);
                if(got[(j*24+h)*256+d]!=expected)throw std::runtime_error("BF16 changed-width graph oracle mismatch");}
            for(int j=0;j<nq;++j)steps[j*kStepCount+kStepWidth]=cells-j%3;ds.put(steps);
            ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));if(out.get()!=before)throw std::runtime_error("BF16 graph nondeterminism");
            cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));int repeat=check_only?1:100;
            ck(cudaEventRecord(a,stream));for(int i=0;i<repeat;++i)ck(cudaGraphLaunch(exec,stream));ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));
            float ms;ck(cudaEventElapsedTime(&ms,a,b));std::printf("%d,%d,%d,%.3f,%.8f\n",nq,cells,api,ms*1000/repeat,error);std::fflush(stdout);
            ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
        }
    }
    ck(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
