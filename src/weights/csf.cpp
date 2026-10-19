#include "strata/weights/csf.hpp"
#include "json_checked.hpp"

namespace strata::weights {
std::vector<uint8_t> csf_decode(std::span<const uint8_t> fixed,
    std::span<const uint8_t> exceptions, uint64_t rows, uint64_t columns) {
    using namespace detail;
    require(rows && columns && columns % 16 == 0, "invalid NVFP4 CSF geometry");
    const auto cols = columns / 16, count = mul(rows, cols);
    require(count <= (1ull << 24), "CSF plane exceeds 24-bit exception addressing");
    const auto code_bytes = (cols + 1) / 2, slab_bytes = 16 * (1 + code_bytes);
    require(fixed.size() == mul((rows + 15) / 16, slab_bytes), "CSF fixed byte size mismatch");
    require(exceptions.size() % 4 == 0, "CSF exceptions must be U32 records");
    std::vector<uint8_t> result(static_cast<size_t>(count));
    // Slab layout matches vLLM tests/quantization/csf_fixtures.py (Apache-2.0,
    // Copyright contributors to the vLLM project). 0x38 is used by that loader
    // for padded TP rows/columns only; it is not an offset for logical scales.
    for (uint64_t r = 0; r < rows; ++r) {
        const auto slab = (r / 16) * slab_bytes, local = r % 16;
        const auto base = fixed[slab + local];
        require(base <= 240, "CSF palette base overflows byte domain");
        for (uint64_t c = 0; c < cols; ++c) {
            const auto code = fixed[slab + 16 + local * code_bytes + c / 2];
            result[r * cols + c] = static_cast<uint8_t>(base + ((code >> (4 * (c % 2))) & 15));
        }
        if (cols % 2) require((fixed[slab + 16 + (local + 1) * code_bytes - 1] & 0xf0) == 0,
                             "CSF nonzero row padding bits");
    }
    uint32_t previous = 0;
    for (size_t i = 0; i < exceptions.size(); i += 4) {
        const uint32_t pos = uint32_t(exceptions[i]) | (uint32_t(exceptions[i+1]) << 8) |
                             (uint32_t(exceptions[i+2]) << 16);
        require(pos < count && (i == 0 || pos > previous), "CSF exception position out of range or not strictly sorted");
        result[pos] = exceptions[i+3];
        previous = pos;
    }
    return result;
}
}
