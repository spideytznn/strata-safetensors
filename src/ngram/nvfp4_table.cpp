#include "strata/ngram/nvfp4_table.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace strata::ngram {
struct Nvfp4Table::Impl {
    struct Plane {
        std::string path;
        uint32_t width;
        uint64_t rows = 0;
        std::vector<PleReader::Segment> segments;
        PleReader reader;
    };
    struct Segment { uint64_t first, count, local[2]; size_t plane[2]; };
    std::vector<std::unique_ptr<Plane>> planes;
    std::vector<Segment> segments;
    uint64_t count = 0;
    float scale = 1;
};
Nvfp4Table::Nvfp4Table() : impl_(std::make_unique<Impl>()) {}
Nvfp4Table::~Nvfp4Table() = default;
uint64_t Nvfp4Table::rows() const { return impl_->count; }
bool Nvfp4Table::open(const std::vector<Nvfp4Segment>& segments, float scale, uint32_t depth,
                     uint64_t cache_rows, bool threaded, std::string& err) {
    impl_ = std::make_unique<Impl>();
    auto& m=*impl_;
    try {
        if (segments.empty() || !std::isfinite(scale) || scale<=0) throw std::runtime_error("invalid NVFP4 PLE table");
        std::map<std::pair<std::string,int>,size_t> ids;
        for (const auto& s:segments) {
            if (s.first_row!=m.count || !s.rows || s.rows>UINT32_MAX-m.count)
                throw std::runtime_error("invalid NVFP4 PLE segment coverage");
            Impl::Segment bound{s.first_row,s.rows,{0,0},{0,0}};
            for (int i=0;i<2;++i) {
                const auto& path=i?s.scale_path:s.weight_path;
                const auto [it,inserted]=ids.emplace(std::pair{path,i},m.planes.size());
                if (inserted) {
                    auto p=std::make_unique<Impl::Plane>(); p->path=path;p->width=i?10:80;
                    m.planes.push_back(std::move(p));
                }
                auto& p=*m.planes[it->second];
                bound.plane[i]=it->second;bound.local[i]=p.rows;
                p.segments.push_back({p.rows,s.rows,i?s.scale_offset:s.weight_offset});p.rows+=s.rows;
            }
            m.segments.push_back(bound);m.count+=s.rows;
        }
        for (auto& p:m.planes) {
            // Across both planes the combined cache stays within cache_rows*90 bytes.
            const auto budget=cache_rows*p->rows/m.count;
            if (!p->reader.open(p->path,0,p->rows,depth,budget,err,threaded,p->width,p->segments))
                throw std::runtime_error(err);
        }
        m.scale=scale;
        return true;
    } catch(const std::exception& e) { err=e.what();impl_=std::make_unique<Impl>();return false; }
}
bool Nvfp4Table::gather(std::span<const uint32_t> rows, std::span<float> out, std::string& err) {
    auto& m=*impl_;
    if (!m.count || out.size()!=rows.size()*160) { err="invalid NVFP4 PLE gather extent";return false; }
    constexpr float fp4[]={0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
    // Bounded scratch even for a long prefill. All plane reads in a tile are
    // issued before collecting, so different shards can make progress together.
    for (size_t first=0;first<rows.size();first+=4096) {
        const size_t n=std::min<size_t>(4096,rows.size()-first);
        struct Request { std::vector<uint32_t> ids;std::vector<size_t> dest;std::vector<uint8_t> raw;PleReader::Ticket ticket; };
        std::vector<Request> requests(m.planes.size());
        std::vector<uint8_t> joined(n*90,0);
        for (size_t r=0;r<n;++r) {
            const auto row=rows[first+r];if (row>=m.count) continue;
            auto it=std::upper_bound(m.segments.begin(),m.segments.end(),row,
                [](uint64_t v,const Impl::Segment& s){return v<s.first;});--it;
            for (int i=0;i<2;++i) {
                auto& q=requests[it->plane[i]];
                q.ids.push_back(static_cast<uint32_t>(it->local[i]+row-it->first));q.dest.push_back(r*90+(i?80:0));
            }
        }
        // Allocate before submitting: output buffers must outlive every ticket.
        for (size_t i=0;i<requests.size();++i) requests[i].raw.resize(requests[i].ids.size()*m.planes[i]->width);
        for (size_t i=0;i<requests.size();++i) if (!requests[i].ids.empty()) {
            auto& q=requests[i];q.ticket=m.planes[i]->reader.issue(q.ids.data(),q.ids.size(),q.raw.data());
        }
        bool ok=true;
        for (size_t i=0;i<requests.size();++i) if (!requests[i].ids.empty()) {
            auto& q=requests[i];std::string error;
            if (!m.planes[i]->reader.collect(q.ticket,error)) { if(ok) err=error;ok=false; }
            const auto width=m.planes[i]->width;
            for (size_t j=0;j<q.ids.size();++j) std::copy_n(q.raw.data()+j*width,width,joined.data()+q.dest[j]);
        }
        if (!ok) return false;
        for (size_t r=0;r<n;++r) for (size_t c=0;c<160;++c) {
            const auto* p=joined.data()+r*90;const auto s=p[80+c/16];
            if (s>126) {err="invalid NVFP4 PLE micro scale";return false;}
            const int e=(s>>3)&15,man=s&7;
            const float micro=e?std::ldexp(1.f+man/8.f,e-7):std::ldexp(float(man),-9);
            out[(first+r)*160+c]=(fp4[(p[c/2]>>(4*(c%2)))&15]*micro)*m.scale;
        }
    }
    return true;
}
}
