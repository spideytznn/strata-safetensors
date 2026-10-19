#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace strata::weights {
// NVFP4 CSF slab stream (not csf_32x2_v1). columns is the logical weight
// width, so the returned E4M3 plane is [rows, columns/16]. Exceptions are
// little-endian U32 bytes: low 24 bits = row-major position, high 8 = value.
std::vector<uint8_t> csf_decode(std::span<const uint8_t> fixed,
    std::span<const uint8_t> exceptions, uint64_t rows, uint64_t columns);
}
