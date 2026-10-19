#include "strata/ngram/nvfp4_table.hpp"
#ifdef TEST_PLE_TABLE
#include "strata/kernels/ngram.hpp"
#endif
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

void check(bool b,const std::string& why) { if (!b) throw std::runtime_error(why); }
int main() {
    namespace fs=std::filesystem;
    const auto root=fs::temp_directory_path()/("strata-nvfp4-ple-"+std::to_string(std::random_device{}()));
    try {
        check(fs::create_directory(root),"test directory collision");
        struct Cleanup { fs::path p;~Cleanup(){std::error_code e;fs::remove_all(p,e);} } cleanup{root};
        std::array<std::vector<uint8_t>,2> files{std::vector<uint8_t>(32768),std::vector<uint8_t>(32768)};
        auto path=[&](int i){auto p=(root/(std::to_string(i)+".bin")).u8string();return std::string(p.begin(),p.end());};
        std::vector<strata::ngram::Nvfp4Segment> segments{
            {0,17,path(0),path(1),4090,8190}, {17,19,path(1),path(0),16380,12280},
            {36,23,path(0),path(1),24570,20470}};
        constexpr float values[]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
        std::vector<float> expected(59*160);
        for (size_t i=0;i<segments.size();++i) {
            const auto& s=segments[i];
            for(size_t r=0;r<s.rows;++r) {
                auto* w=files[i==1?1:0].data()+s.weight_offset+r*80;
                auto* scale=files[i==1?0:1].data()+s.scale_offset+r*10;
                for(size_t c=0;c<80;++c) w[c]=uint8_t((r+s.first_row)*13+c*7);
                for(size_t c=0;c<10;++c) scale[c]=uint8_t(24+(r+c)%48);
                for(size_t c=0;c<160;++c) {
                    const auto v=scale[c/16];const int e=(v>>3)&15,m=v&7;
                    const auto micro=e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(float(m),-9);
                    expected[(s.first_row+r)*160+c]=values[(w[c/2]>>(4*(c%2)))&15]*micro*.25f;
                }
            }
        }
        for(int i=0;i<2;++i) {std::ofstream f(root/(std::to_string(i)+".bin"),std::ios::binary);f.write(reinterpret_cast<char*>(files[i].data()),files[i].size());}
        for(bool threaded:{false,true}) {
            strata::ngram::Nvfp4Table table;std::string err;
            check(table.open(segments,.25f,8,128,threaded,err),err);
            std::vector<uint32_t> rows{0,16,17,18,35,36,58,UINT32_MAX,59,17,0};
            // Cross the bounded gather tile and exercise cached repeated rows.
            for(size_t i=0;i<4200;++i) rows.push_back(uint32_t(i%59));
            std::vector<float> out(rows.size()*160);
            for(int repeat=0;repeat<2;++repeat) {
                check(table.gather(rows,out,err),err);
                for(size_t i=0;i<rows.size();++i) for(size_t c=0;c<160;++c) {
                    const float ref=rows[i]<59?expected[rows[i]*160+c]:0;
                    check(out[i*160+c]==ref && std::signbit(out[i*160+c])==std::signbit(ref),"NVFP4 PLE decode/order/sign mismatch");
                }
            }
#ifdef TEST_PLE_TABLE
            strata::kernels::PleTable runtime;
            strata::kernels::PleIoOptions io;
            io.mode=strata::kernels::PleIo::Direct;io.io_thread=threaded;io.max_inflight=8;io.cache_rows=128;
            check(runtime.open_nvfp4_segments(segments,.25f,err,io),err);
            check(runtime.is_open() && runtime.rows()==59 && std::string(runtime.format())=="NVFP4","runtime table state");
            check(runtime.issue(rows.data()),"runtime issue");
            check(!runtime.issue(rows.data()),"overlapping runtime issue accepted");
            std::vector<float> actual(2560);
            check(runtime.collect(actual.data(),err),err);
            check(std::equal(actual.begin(),actual.end(),out.begin()),"runtime split gather mismatch");
            check(!runtime.collect(actual.data(),err),"collect without issue accepted");
            actual.resize(2560*260);
            check(runtime.gather_batch(rows.data(),260,actual.data(),err),err);
            check(std::equal(actual.begin(),actual.end(),out.begin()),"runtime prefill gather mismatch");
            runtime.read_row(17,actual.data());
            check(std::equal(actual.begin(),actual.begin()+160,expected.begin()+17*160),"runtime read_row mismatch");
            check(runtime.issue(rows.data()),"runtime close issue");
            runtime.close(); // drains outstanding output before destroying readers
            check(!runtime.is_open(),"runtime close state");
#endif
            auto bad=segments;bad[1].first_row++;
            check(!table.open(bad,.25f,8,0,threaded,err),"accepted PLE gap");
        }
        std::cout<<"PASS: NVFP4 PLE two files, separate planes, cross-page rows, bounded batch, cache, signed zero\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
