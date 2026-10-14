// No model. Compare the production slot-layer helper with the original per-layer graph.
#include "strata/core/slot_commit.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/rope.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
using strata::core::SlotCommitLayers;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool b, const char* s) { if (!b) throw std::runtime_error(s); }
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t size) : n(size) { ck(cudaMalloc((void**) &p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& v) { require(v.size() == n, "fixture size"); ck(cudaMemcpy(p, v.data(), n*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
template<class T> struct Mapped {
    T *host = nullptr, *device = nullptr;
    explicit Mapped(size_t n) {
        ck(cudaHostAlloc((void**) &host, n*sizeof(T), cudaHostAllocMapped));
        ck(cudaHostGetDevicePointer((void**) &device, host, 0));
    }
    ~Mapped() { cudaFreeHost(host); }
};
template<class T> void equal(const Device<T>& a, const Device<T>& b, const char* label) {
    const auto x = a.get(), y = b.get();
    require(x.size() == y.size() && !std::memcmp(x.data(), y.data(), x.size()*sizeof(T)), label);
}
constexpr int HV = 48, HK = 16, C = (2*HK+HV)*128, ID = 128, MT = 8, CB = MT+2;
constexpr int64_t CONV = int64_t(128)*HV*128, STRIDE = CONV+C*3+64;
struct State {
    Device<float> gdn, tail, dead, pooled, ple;
    Device<int32_t> block;
    std::vector<float> initial, ti, di, pi;
    int max_cells;
    State(int ng, int nq, int cells, int seed) : gdn(ng*STRIDE), tail(nq*3*ID), dead(nq*ID),
        pooled(nq*(cells/4+1)*ID), ple(64), block(nq), max_cells(cells) {
        std::mt19937 r(seed);
        auto fill = [&](size_t n) { std::vector<float> v(n); for (auto& x:v) x=float(int(r()%2001)-1000)*0.0001f; return v; };
        initial=fill(gdn.n); ti=fill(tail.n); di=fill(dead.n); pi=fill(pooled.n);
    }
    void reset() { gdn.put(initial); tail.put(ti); dead.put(di); pooled.put(pi); ple.put(std::vector<float>(64, -7)); block.put(std::vector<int32_t>(block.n, 4)); }
    void compare(const State& s) const {
        equal(gdn,s.gdn,"GDN state/history/guard changed"); equal(tail,s.tail,"indexer tail changed");
        equal(dead,s.dead,"indexer spare changed"); equal(pooled,s.pooled,"indexer pool changed");
        equal(block,s.block,"indexer block position changed"); equal(ple,s.ple,"PLE history changed");
    }
};
struct Fixture {
    int ng, nq;
    const int64_t xs = MT*C+64, gs = MT*HV+64, is = MT*ID+64;
    Device<float> qkv, h, gate, beta, raw, gamma, norm, z, dummy, history, cosine, sine;
    Device<int32_t> controls, positions[3];
    Mapped<int32_t> host_controls;
    Mapped<float> snapshots;
    std::unique_ptr<State> ref[3], got[3];
    cudaStream_t stream = nullptr;
    Fixture(int g, int q) : ng(g),nq(q),qkv(g*xs),h(g*xs),gate(g*gs),beta(g*gs),raw(q*is),
        gamma(q*ID),norm(128),z(MT*HV*128),dummy(MT*HV*128),history(MT*64),cosine(128*32),sine(128*32),
        controls(3*CB),positions{Device<int32_t>(128*3),Device<int32_t>(128*3),Device<int32_t>(128*3)},
        host_controls(3*CB),snapshots(3*q*3*ID) {
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        std::mt19937 r(619);
        auto fill=[&](Device<float>& d,float base,float scale) {
            std::vector<float> v(d.n);for(auto& x:v)x=base+float(int(r()%2001)-1000)*scale;d.put(v);
        };
        fill(qkv,0,.0001f);fill(h,0,.0001f);fill(gate,-1,.0003f);fill(beta,.5f,.0001f);
        fill(raw,0,.001f);fill(gamma,1,.0002f);fill(history,0,.01f);
        norm.put(std::vector<float>(128,1));z.put(std::vector<float>(z.n,0));
        for(int i=0;i<3;++i) {
            ref[i]=std::make_unique<State>(g,q,65+16*i,77+i);got[i]=std::make_unique<State>(g,q,65+16*i,77+i);
            std::vector<int32_t> p(positions[i].n);
            for(int cell=0;cell<128;++cell)for(int axis=0;axis<3;++axis)p[cell*3+axis]=cell/3+(i+1)*(axis+1)+cell%(axis+2);
            positions[i].put(p);
        }
        for(int i=0;i<3*q*3*ID;++i)snapshots.host[i]=float(int(r()%1000)-500)*.0002f;
        k::mrope_table_set(positions[1].p); // sentinel: even a text slot must override, then restore this pointer
    }
    ~Fixture() { k::mrope_table_set(nullptr);k::rope_table_release(cosine.p);cudaStreamDestroy(stream); }
    void reset() { for(int s=0;s<3;++s){ref[s]->reset();got[s]->reset();} }
    void control(int na,int nb,int ka,int kb,int base) {
        std::fill(host_controls.host,host_controls.host+3*CB,-1);
        // Nonconsecutive physical slots 2 and 0, in the opposite order from their input row offsets.
        for(int j=0;j<2;++j) {
            const int slot=j?0:2,n=j?nb:na,keep=j?kb:ka;
            auto* p=host_controls.host+slot*CB;p[0]=keep;p[1]=keep?keep-1:-1;
            for(int t=0;t<n;++t)p[2+t]=t<keep?base+j*8+t:-1;
        }
    }
    SlotCommitLayers args(int slot,int first,int n,bool optimized,bool images) {
        State& s=*(optimized?got[slot]:ref[slot]);SlotCommitLayers a;
        a.state=s.gdn.p;a.state_stride=STRIDE;a.conv_offset=CONV;
        a.qkv=qkv.p+first*C;a.h=h.p+first*C;a.gate=gate.p+first*HV;a.beta=beta.p+first*HV;
        a.qkv_stride=a.h_stride=xs;a.gate_stride=gs;a.conv_channels=C;a.h_k=HK;a.h_v=HV;
        a.steps=n;a.gdn_layers=ng;a.control=controls.p+slot*CB;
        a.index_raw=raw.p+first*ID;a.index_stride=is;a.qsa_layers=nq;
        a.shapes=k::qsa_real_shapes();a.max_cells=s.max_cells;a.mrope=images?positions[slot].p:nullptr;
        for(int l=0;l<nq;++l) {
            a.indexer.gamma[l]=gamma.p+l*ID;a.indexer.tail_src[l]=snapshots.device+(slot*nq+l)*3*ID;
            a.indexer.tail[l]=s.tail.p+l*3*ID;a.indexer.dead[l]=s.dead.p+l*ID;
            a.indexer.pooled[l]=s.pooled.p+l*(s.max_cells/4+1)*ID;a.indexer.block_pos[l]=s.block.p+l;
        }
        return a;
    }
    void record(int na,int nb,bool optimized,bool images,bool canonical,const k::RopeScaling& scaling) {
        for(int slot:{2,0})k::copy_i32_from_mapped(controls.p+slot*CB,host_controls.device+slot*CB,CB,stream);
        for(int j=0;j<2;++j) {
            const int slot=j?0:2,first=j?na:0,n=j?nb:na;
            const auto a=args(slot,first,n,optimized,images);
            const auto* saved=k::mrope_table();
            auto done=optimized?strata::core::commit_slot_layers(a,scaling,stream):strata::core::SlotCommitResult{};
            require(k::mrope_table()==saved,"slot helper leaked image positions");
            if(optimized) require(done.qsa,"fixture did not exercise the layer-batched indexer");
            if(!done.gdn)for(int l=0;l<ng;++l) {
                auto* st=a.state+l*STRIDE;k::gdn_conv_commit(st+CONV,a.qkv+l*xs,C,a.control,stream);
                k::gdn_step_norm_multi(st,a.h+l*xs,C,a.gate+l*gs,a.beta+l*gs,z.p,norm.p,1e-6f,dummy.p,
                                      HK,HV,n,a.control,stream,n>1?n:0);
            }
            k::mrope_table_set(a.mrope);
            for(int l=0;l<nq;++l) {
                const auto& b=a.indexer;
                if(!done.qsa) {
                    k::copy_from_mapped(b.tail[l],b.tail_src[l],3*ID,stream);
                    k::QsaIndexerBuffers ib{b.tail[l],b.dead[l],b.pooled[l],b.block_pos[l]};
                    for(int t=0;t<n;++t)k::native_qsa_indexer_append(a.index_raw+l*is+t*ID,a.control+2+t,0,
                        b.gamma[l],1e-6f,ib,a.shapes,a.max_cells,scaling,stream);
                }
                if(canonical)k::qsa_commit_spare(b.pooled[l],b.dead[l],b.block_pos[l],a.control,ID,4,int(a.max_cells),stream);
            }
            k::mrope_table_set(saved);
            State& s=*(optimized?got[slot]:ref[slot]);
            k::copy_indexed(s.ple.p,history.p+first*64,64,a.control+1,64,stream);
        }
    }
    struct Graph {
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;size_t nodes=0;
        ~Graph(){cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}
    };
    void capture(Graph& g,int na,int nb,bool optimized,bool images,bool canonical,const k::RopeScaling& scaling) {
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));record(na,nb,optimized,images,canonical,scaling);
        ck(cudaStreamEndCapture(stream,&g.graph));ck(cudaGraphGetNodes(g.graph,nullptr,&g.nodes));
        ck(cudaGraphInstantiate(&g.exec,g.graph,0));
    }
};
int main(int argc,char** argv) try {
    const bool bench=argc==2 && std::string(argv[1])=="--bench";
    require(argc==1 || bench,"usage: slot_commit_layers_test [--bench]");
    Fixture f(bench?36:3,bench?12:3);
    int cases=0;
    const std::vector<std::pair<int,int>> lengths=bench?std::vector<std::pair<int,int>>{{2,2}}:
        std::vector<std::pair<int,int>>{{1,1},{1,2},{2,1},{2,2},{3,2},{2,3},{4,4}};
    for(int mode=0;mode<(bench?1:3);++mode)for(bool table:bench?std::vector<bool>{true}:std::vector<bool>{false,true})
        for(bool images:{false,true})for(bool canonical:{false,true}) {
        k::RopeScaling scaling;scaling.type=k::RopeScalingType(mode);scaling.factor=mode?2:1;scaling.ext_factor=mode==2?1:0;
        k::rope_table_release(f.cosine.p);
        if(table) {
            std::vector<float> c(128*32),s(c.size());k::build_rope_table(64,scaling,128,c.data(),s.data());
            f.cosine.put(c);f.sine.put(s);k::rope_table_set(f.cosine.p,f.sine.p,128,scaling);
        }
        for(auto [na,nb]:lengths) {
            Fixture::Graph ref,got;f.capture(ref,na,nb,false,images,canonical,scaling);f.capture(got,na,nb,true,images,canonical,scaling);
            require(got.nodes<ref.nodes,"optimized graph did not reduce node count");
            for(int phase=0;phase<4;++phase) {
                const int ka=phase==0?0:phase==1?na:phase==2?1:na/2;
                const int kb=phase==0?nb:phase==1?0:phase==2?nb/2:nb;
                f.reset();f.control(na,nb,ka,kb,phase==1?0:phase);
                ck(cudaGraphLaunch(ref.exec,f.stream));ck(cudaGraphLaunch(got.exec,f.stream));ck(cudaStreamSynchronize(f.stream));
                for(int slot=0;slot<3;++slot)f.ref[slot]->compare(*f.got[slot]);
                ++cases;
            }
            if(bench) {
                std::vector<float> times[2];cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
                for(int pair=0;pair<9;++pair)for(int step=0;step<2;++step) {
                    const int arm=step^(pair&1);ck(cudaEventRecord(a,f.stream));
                    for(int r=0;r<10;++r)ck(cudaGraphLaunch(arm?got.exec:ref.exec,f.stream));
                    ck(cudaEventRecord(b,f.stream));ck(cudaEventSynchronize(b));float ms=0;ck(cudaEventElapsedTime(&ms,a,b));times[arm].push_back(ms*100);
                }
                for(auto& v:times)std::sort(v.begin(),v.end());
                std::printf("BENCH images=%d canonical=%d nodes=%zu->%zu median_us=%.3f->%.3f ratio=%.3f\n",
                    images,canonical,ref.nodes,got.nodes,times[0][4],times[1][4],times[0][4]/times[1][4]);
                ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
            }
        }
    }
    // An invalid QSA descriptor must restore the caller's M-RoPE even when validation throws.
    {
        SlotCommitLayers a;a.steps=2;a.qsa_layers=k::QsaIndexerLayers::kMax+1;
        a.shapes=k::qsa_real_shapes();a.max_cells=65;a.control=f.controls.p;a.mrope=f.positions[0].p;
        const auto* saved=k::mrope_table();bool rejected=false;
        try {strata::core::commit_slot_layers(a,k::RopeScaling{},f.stream);}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected && k::mrope_table()==saved,"validation failure leaked slot positions");
    }
    std::printf("PASS: %d graph replay cases, independent acceptance, inactive slot, guards, PLE, M-RoPE and canonical cleanup\n",cases);
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
