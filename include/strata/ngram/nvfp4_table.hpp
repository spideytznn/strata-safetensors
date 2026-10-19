#pragma once
#include "strata/ngram/ple_reader.hpp"
#include <memory>
#include <span>

namespace strata::ngram {
struct Nvfp4Segment {
    uint64_t first_row, rows;
    std::string weight_path, scale_path;
    uint64_t weight_offset, scale_offset;
};
// Separate weight/scale planes, potentially in different safetensors shards.
// Uses bounded existing direct-I/O readers and their caches, never maps the table.
class Nvfp4Table {
public:
    Nvfp4Table();
    ~Nvfp4Table();
    bool open(const std::vector<Nvfp4Segment>&, float scale, uint32_t depth,
              uint64_t cache_rows, bool threaded, std::string& err);
    bool gather(std::span<const uint32_t> rows, std::span<float> out, std::string& err);
    uint64_t rows() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
