#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
// KV encoding/attention only. Does not select native-precision experts or MTP
// projections. Immutable while session pools and captured graphs are alive.
inline bool native_bf16_kv_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_KV_BF16_TC");
        if (!v || !*v || std::strcmp(v, "0") == 0) return false;
        if (std::strcmp(v, "1") != 0) throw std::invalid_argument("STRATA_KV_BF16_TC must be 0 or 1");
#if !defined(STRATA_BF16_KV_TC_BUILT) || defined(STRATA_USE_HIP)
        throw std::runtime_error("BF16 KV requires -DSTRATA_BF16_KV_TC=ON on CUDA SM120");
#else
        return true;
#endif
    }();
    return enabled;
}
}
