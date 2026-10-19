#include "strata/weights/nvfp4.hpp"
#include "strata/weights/csf.hpp"
#include "strata/weights/dense.hpp"
#include <fstream>
#include <bit>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace w = strata::weights;
namespace {
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct MockSource : w::WeightSource {
    std::map<std::string,w::TensorDesc> descriptors;
    std::map<std::string,std::vector<uint8_t>> data;
    w::IoStats stats;
    size_t batches = 0;
    const w::TensorDesc& tensor(const std::string& name) const override { return descriptors.at(name); }
    const w::TensorDesc* find_tensor(const std::string& name) const override {
        const auto it=descriptors.find(name); return it==descriptors.end()?nullptr:&it->second;
    }
    const w::IoStats& io_stats() const override { return stats; }
    void read_many(std::span<const w::ReadRequest> reads) override {
        ++batches;
        for (const auto& r : reads) {
            const auto& b=data.at(r.tensor->name);
            check(r.relative_offset<=b.size() && r.destination.size()<=b.size()-r.relative_offset,"mock bounds");
            std::memcpy(r.destination.data(),b.data()+r.relative_offset,r.destination.size());
        }
    }
    void add(const std::string& name,w::DType dtype,std::vector<uint64_t> shape,std::vector<uint8_t> bytes) {
        descriptors[name]={name,dtype,w::Family::Expert,std::move(shape),0,0,bytes.size()};
        data[name]=std::move(bytes);
    }
    void scale(const std::string& name,float value) {
        std::vector<uint8_t> b(4); std::memcpy(b.data(),&value,4);
        add(name,w::DType::F32,{},std::move(b));
    }
};
void projection_test() {
    MockSource source;
    std::array<w::Nvfp4Projection,3> projections;
    for (size_t i=0;i<3;++i) {
        const auto prefix=std::to_string(i);
        const uint64_t rows=i==2?2560:640, cols=i==2?640:2560;
        source.add(prefix+".weight",w::DType::U8,{rows,cols/2},std::vector<uint8_t>(rows*cols/2,0x98));
        source.add(prefix+".weight_scale",w::DType::F8_E4M3,{rows,cols/16},std::vector<uint8_t>(rows*cols/16,1));
        source.scale(prefix+".weight_scale_2",std::ldexp(1.0f,-130)); // cannot be retained in FP16
        source.scale(prefix+".input_scale",static_cast<float>(i+1)); // deliberately unequal gate/up
        projections[i]=w::nvfp4_projection(source,prefix,prefix,rows,cols);
    }
    const auto loaded=w::load_nvfp4_expert(source,projections,7,true);
    check(loaded.layout.bytes==2764816 && loaded.layout.tail_off==2764800 && loaded.layout.layer==7,"layout ABI");
    check(loaded.input_scales==std::array<float,3>{1,2,3} && loaded.layout.input_scale_gu==0,"unequal input scales collapsed");
    check(loaded.roundtrip_bytes==2764824,"roundtrip coverage");
    for (const auto& suffix : {"weight_scale_2","input_scale"}) {
        for (float bad : {0.f,-1.f,INFINITY,NAN}) {
            source.scale(std::string("0.")+suffix,bad);
            bool refused=false;
            try { w::load_nvfp4_expert(source,projections,0,false); } catch(const std::runtime_error&) { refused=true; }
            check(refused,"invalid global/input scale accepted");
        }
        source.scale(std::string("0.")+suffix,1.f);
    }
    bool refused=false;
    source.descriptors.at("0.weight").physical_shape={1280,640};
    try { w::nvfp4_projection(source,"0","bad",640,2560); } catch(const std::runtime_error&) { refused=true; }
    check(refused,"transposed HF shape accepted");
}
void batch_test() {
    MockSource source;
    constexpr size_t count = 5, bytes = 2764816;
    std::array<std::array<w::Nvfp4Projection,3>,count> projections;
    std::array<w::NativeExpert,count> expected;
    for (size_t e=0;e<count;++e) {
        for (size_t p=0;p<3;++p) {
            const auto prefix=std::to_string(e)+"/"+std::to_string(p);
            const uint64_t rows=p==2?2560:640,cols=p==2?640:2560;
            std::vector<uint8_t> weights(rows*cols/2),scales(rows*cols/16);
            for (size_t i=0;i<weights.size();++i) weights[i]=static_cast<uint8_t>((i*73+e*31+p*7)&255);
            for (size_t i=0;i<scales.size();++i) scales[i]=static_cast<uint8_t>((i+e+p)%127);
            source.add(prefix+".weight",w::DType::U8,{rows,cols/2},std::move(weights));
            source.add(prefix+".weight_scale",w::DType::F8_E4M3,{rows,cols/16},std::move(scales));
            source.scale(prefix+".weight_scale_2",std::ldexp(static_cast<float>(e+p+1),-130));
            source.scale(prefix+".input_scale",static_cast<float>(e*3+p+1));
            projections[e][p]=w::nvfp4_projection(source,prefix,prefix,rows,cols);
        }
        expected[e]=w::load_nvfp4_expert(source,projections[e],0,true);
    }
    std::vector<uint8_t> arena(count*bytes+2,0xcd);
    std::array<std::array<float,3>,count> scales;
    std::vector<w::Nvfp4ExpertBuffer> targets;
    for (size_t e=0;e<count;++e)
        targets.push_back({&projections[e],{arena.data()+1+e*bytes,bytes},&scales[e]});
    w::Nvfp4LoadScratch scratch;
    for (unsigned workers : {1u,4u,16u}) {
        std::fill(arena.begin(),arena.end(),uint8_t(0xcd));
        const auto before=source.batches;
        const auto stats=w::load_nvfp4_expert_batch(source,targets,workers,&scratch);
        check(source.batches==before+1,"batch did not combine source requests");
        check(stats.scratch_bytes==count*(2764800+24),"batch scratch bound");
        check(arena.front()==0xcd && arena.back()==0xcd,"batch overwrote destination canary");
        for (size_t e=0;e<count;++e) {
            check(std::memcmp(targets[e].bytes.data(),expected[e].bytes.data(),bytes)==0,"batch differs from roundtrip oracle");
            check(scales[e]==expected[e].input_scales,"batch changed or mixed input scales");
        }
    }
    std::fill(arena.begin(),arena.end(),uint8_t(0xcd));
    const std::span<const w::Nvfp4ExpertBuffer> spans(targets);
    w::load_nvfp4_expert_batch(source,spans.first(3),4,&scratch);
    w::load_nvfp4_expert_batch(source,spans.subspan(3),4,&scratch);
    for (size_t e=0;e<count;++e)
        check(std::memcmp(targets[e].bytes.data(),expected[e].bytes.data(),bytes)==0,"short final batch changed bytes");
    auto rejects=[&](auto action) {
        bool refused=false;
        try { action(); } catch (const std::runtime_error&) { refused=true; }
        check(refused,"invalid startup batch accepted");
    };
    auto before=source.batches;
    rejects([&]{ w::load_nvfp4_expert_batch(source,{},1); });
    rejects([&]{ w::load_nvfp4_expert_batch(source,targets,0); });
    rejects([&]{ w::load_nvfp4_expert_batch(source,targets,17); });
    auto overlapping=targets;
    overlapping[1].bytes=overlapping[0].bytes;
    rejects([&]{ w::load_nvfp4_expert_batch(source,overlapping,4); });
    check(source.batches==before,"invalid destination batch read source data");
    // Errors in worker threads must reach the loader instead of terminating or
    // leaving another worker writing into an arena the caller has already freed.
    source.data.at("4/2.weight_scale")[0]=0xff;
    rejects([&]{ w::load_nvfp4_expert_batch(source,targets,4); });
    source.data.at("4/2.weight_scale")[0]=1;
    source.scale("4/2.input_scale",NAN);
    rejects([&]{ w::load_nvfp4_expert_batch(source,targets,4); });
}

void csf_test() {
    auto read=[](const char* name) {
        std::ifstream f(std::string(CSF_FIXTURE_DIR)+"/"+name,std::ios::binary);
        check(bool(f),"cannot open CSF golden fixture");
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),{});
    };
    const auto fixed=read("fixed.bin"),exceptions=read("exceptions.bin"),golden=read("decoded.bin");
    check(w::csf_decode(fixed,exceptions,32,128)==golden,"CSF vLLM golden mismatch");
    // An odd number of scale columns and a partial final 16-row slab.
    std::vector<uint8_t> partial(96,0);
    for(size_t r=0;r<17;++r) {
        const auto slab=r/16*48,local=r%16;
        partial[slab+local]=0x38;
        partial[slab+16+local*2]=0x21;
        partial[slab+17+local*2]=0x03;
    }
    auto decoded=w::csf_decode(partial,{},17,48);
    for(size_t i=0;i<decoded.size();++i) check(decoded[i]==0x39+i%3,"CSF partial slab or nibble order");
    auto rejects=[](auto action) { bool failed=false; try { action(); } catch(const std::runtime_error&) { failed=true; }
        check(failed,"malformed CSF accepted"); };
    rejects([&]{ w::csf_decode(std::span(partial).first(95),{},17,48); });
    const std::vector<uint8_t> outside={51,0,0,3},duplicate={0,0,0,1,0,0,0,2};
    rejects([&]{ w::csf_decode(partial,outside,17,48); });
    rejects([&]{ w::csf_decode(partial,duplicate,17,48); });
    rejects([&]{ w::csf_decode(partial,std::span(outside).first(3),17,48); });
    partial[0]=241;
    rejects([&]{ w::csf_decode(partial,{},17,48); });

    // Compare compressed and original source paths at actual expert dimensions,
    // including one plain projection in the same batch and scratch reuse.
    MockSource source;
    std::array<w::Nvfp4Projection,3> plain,csf;
    for(size_t p=0;p<3;++p) {
        const auto prefix=std::to_string(p);
        const uint64_t rows=p==2?2560:640,cols=p==2?640:2560,sc=cols/16,cb=sc/2,stride=16*(1+cb);
        std::vector<uint8_t> fx(rows/16*stride),scales(rows*sc,0x38),ex={0,0,0,126};
        for(size_t r=0;r<rows;++r) fx[r/16*stride+r%16]=0x38;
        scales[0]=126;
        source.add(prefix+".weight",w::DType::U8,{rows,cols/2},std::vector<uint8_t>(rows*cols/2,0x98));
        source.add(prefix+".weight_scale",w::DType::F8_E4M3,{rows,sc},scales);
        source.scale(prefix+".weight_scale_2",.03125f); source.scale(prefix+".input_scale",1.f);
        plain[p]=w::nvfp4_projection(source,prefix,prefix,rows,cols);
        source.add(prefix+".weight_scale.nvfp4_csf_fixed",w::DType::U8,{rows/16,stride},fx);
        source.add(prefix+".weight_scale.nvfp4_csf_exceptions",w::DType::U32,{1},ex);
    }
    const auto oracle=w::load_nvfp4_expert(source,plain,0,true);
    for(size_t p=0;p<3;++p) {
        const auto prefix=std::to_string(p);source.descriptors.erase(prefix+".weight_scale");
        csf[p]=w::nvfp4_projection(source,prefix,prefix,p==2?2560:640,p==2?640:2560);
    }
    check(w::load_nvfp4_expert(source,csf,0,true).bytes==oracle.bytes,"CSF expert differs from plain NVFP4");
    std::vector<uint8_t> output(oracle.bytes.size()); std::array<float,3> input;
    w::Nvfp4ExpertBuffer target{&csf,output,&input}; w::Nvfp4LoadScratch scratch;
    for(int i=0;i<2;++i) {
        w::load_nvfp4_expert_batch(source,{&target,1},4,&scratch);
        check(output==oracle.bytes && input==oracle.input_scales,"CSF batch differs from plain NVFP4");
    }
    source.add("1.weight_scale",w::DType::F8_E4M3,{640,160},source.data.at("1.weight_scale"));
    source.descriptors.erase("1.weight_scale.nvfp4_csf_fixed");
    source.descriptors.erase("1.weight_scale.nvfp4_csf_exceptions");
    csf[1]=w::nvfp4_projection(source,"1","1",640,2560);
    source.descriptors.erase("0.input_scale");
    csf[0]=w::nvfp4_projection(source,"0","0",640,2560,false); // W4A16 has no activation scale
    w::load_nvfp4_expert_batch(source,{&target,1},4,&scratch);
    check(output==oracle.bytes && input==oracle.input_scales,"mixed plain/CSF or W4A16 batch mismatch");
    check(w::load_nvfp4_expert(source,csf,0,true).bytes==oracle.bytes,"W4A16 single load mismatch");
}
void mxfp8_test() {
    check(w::mxfp8_bf16(0x38,0)==0x40 && w::mxfp8_bf16(0x38,1)==0x80 &&
          w::mxfp8_bf16(4,0)==0 && w::mxfp8_bf16(5,0)==1 && w::mxfp8_bf16(0x84,0)==0x8000,
          "MXFP8 UE8M0 zero exponent or BF16 subnormal rounding mismatch");
    // All finite E4M3 codes and representative UE8M0 exponents against FP64.
    for(unsigned code=0;code<256;++code) if((code&127)!=127) for(unsigned scale=16;scale<239;++scale) {
        const int e=(code>>3)&15,m=code&7;
        double v=std::ldexp(e?1.+m/8.:double(m),int(scale)-127+(e?e-7:-9));
        if(code&128) v=-v;
        const auto got=std::bit_cast<float>(uint32_t(w::mxfp8_bf16(uint8_t(code),uint8_t(scale)))<<16);
        check(double(got)==v && std::signbit(got)==std::signbit(v),"MXFP8 exact conversion/sign mismatch");
    }
    for(const auto [code,scale]:{std::pair{127,127},std::pair{1,255},std::pair{126,254}}) {
        bool failed=false; try { w::mxfp8_bf16(uint8_t(code),uint8_t(scale)); } catch(const std::runtime_error&) { failed=true; }
        check(failed,"invalid MXFP8 accepted");
    }
    MockSource source;
    constexpr uint64_t rows=50,cols=1536;
    std::vector<uint8_t> weight(rows*cols),scale(rows*cols/32);
    for(size_t i=0;i<weight.size();++i) weight[i]=uint8_t(i%126);
    for(size_t i=0;i<scale.size();++i) scale[i]=uint8_t(120+i%12);
    source.add("w",w::DType::F8_E4M3,{rows,cols},weight);
    source.add("s",w::DType::U8,{rows,cols/32},scale);
    w::DenseBinding b{"test",&source.tensor("w"),48,cols,2,1,0,32,w::DType::BF16,w::DenseMath::Identity,true,&source.tensor("s")};
    std::vector<uint8_t> out(b.bytes());
    w::stream_dense(source,b,[&](uint64_t off,std::span<const uint8_t> tile){ std::memcpy(out.data()+off,tile.data(),tile.size()); },8192);
    for(size_t r=0;r<48;++r) for(size_t c=0;c<cols;++c) {
        const auto sr=(r%16)*3+r/16+2,head=c/32,sc=((head%16)*3+head/16)*32+c%32;
        const auto expected=w::mxfp8_bf16(weight[sr*cols+sc],scale[sr*(cols/32)+sc/32]);
        uint16_t got;std::memcpy(&got,out.data()+2*(r*cols+c),2);
        check(got==expected,"MXFP8 scale row slice/GDN permutation mismatch");
    }
    const std::array<float,3> norm={.1234567f,-.00001234567f,2.456789f};
    std::vector<uint8_t> norm_bytes(sizeof(norm));std::memcpy(norm_bytes.data(),norm.data(),sizeof(norm));
    source.add("norm",w::DType::F32,{3},norm_bytes);
    w::DenseBinding nb{"norm",&source.tensor("norm"),3,1,0,0,0,0,w::DType::F32,w::DenseMath::AddOne};
    std::array<float,3> norm_out{};
    w::stream_dense(source,nb,[&](uint64_t off,std::span<const uint8_t> tile){std::memcpy(reinterpret_cast<uint8_t*>(norm_out.data())+off,tile.data(),tile.size());},8);
    for(size_t i=0;i<norm.size();++i) check(norm_out[i]==norm[i]+1.f,"F32 norm rounded through BF16");
}
double fp4(unsigned code) {
    constexpr double magnitude[] = {0,0.5,1,1.5,2,3,4,6};
    return (code & 8 ? -1 : 1) * magnitude[code & 7];
}
double fp8(uint8_t b) {
    const auto e = (b >> 3) & 15, m = b & 7;
    return e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(static_cast<double>(m), -9);
}
// Independent direct weight indexing. No use of the inverse permutation for
// the numerical check, including negative zero and all valid micro scales.
double native_weight(const std::vector<uint8_t>& p, size_t index) {
    const auto b = index / 64, sub = index % 64 / 16, j = index % 16;
    const auto q = p[b*36+4+sub*8+j%8];
    return fp4((q >> (j/8*4)) & 15) * fp8(p[b*36+sub]);
}
void test(uint64_t rows, uint64_t cols) {
    const size_t n = static_cast<size_t>(rows*cols);
    std::vector<uint8_t> weights(n/2), scales(n/16), packed(n/64*36), back(n/2), sb(n/16);
    for (size_t i=0;i<weights.size();++i) weights[i] = static_cast<uint8_t>((i*73+19)&255);
    for (size_t i=0;i<scales.size();++i) scales[i] = static_cast<uint8_t>(i%127);
    w::pack_nvfp4(weights,scales,rows,cols,packed);
    w::unpack_nvfp4(packed,rows,cols,back,sb);
    check(weights==back && scales==sb,"bytes not restored");
    for (size_t i=0;i<n;++i) {
        const double source=fp4((weights[i/2] >> (i%2*4))&15)*fp8(scales[i/16]);
        const double dest=native_weight(packed,i);
        check(source==dest && std::signbit(source)==std::signbit(dest),"decoded value/sign changed");
    }
    for (uint8_t bad : {uint8_t(0x7f),uint8_t(0x80),uint8_t(0xff)}) {
        scales[0]=bad;
        bool refused=false;
        try { w::pack_nvfp4(weights,scales,rows,cols,packed); } catch(const std::runtime_error&) { refused=true; }
        check(refused,"invalid UE4M3 scale accepted");
    }
}
}
int main() {
    try {
        csf_test(); mxfp8_test(); test(7,128); test(640,2560); test(2560,640); projection_test(); batch_test();
        std::cout << "PASS: byte roundtrip, independent FP64 decode, signed zero, all finite micro scales, invalid scales\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
