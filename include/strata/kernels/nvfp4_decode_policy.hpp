#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
inline bool nvfp4_w4a4_decode() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_NVFP4_DECODE");
        if (!v || !*v || std::strcmp(v, "fp32") == 0) return false;
        if (std::strcmp(v, "w4a4") == 0) return true;
        throw std::invalid_argument("STRATA_NVFP4_DECODE must be fp32 or w4a4");
    }();
    return enabled;
}
}
