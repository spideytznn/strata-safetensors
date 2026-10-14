#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
enum class Nvfp4PrefillBackend { Custom, Cutlass, Auto };
inline Nvfp4PrefillBackend nvfp4_prefill_backend() {
    static const auto backend = [] {
        const char* v = std::getenv("STRATA_NVFP4_PREFILL_BACKEND");
        if (!v || !*v || std::strcmp(v, "custom") == 0) return Nvfp4PrefillBackend::Custom;
        if (std::strcmp(v, "cutlass") != 0 && std::strcmp(v, "auto") != 0)
            throw std::invalid_argument("STRATA_NVFP4_PREFILL_BACKEND must be custom, cutlass or auto");
#ifndef STRATA_NVFP4_CUTLASS_BUILT
        throw std::runtime_error("CUTLASS prefill needs -DSTRATA_NVFP4_CUTLASS=ON");
#else
        return std::strcmp(v, "auto") == 0 ? Nvfp4PrefillBackend::Auto : Nvfp4PrefillBackend::Cutlass;
#endif
    }();
    return backend;
}
inline bool native_nvfp4_prefill_w4a16() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_PREFILL_NVFP4");
        return v && std::strcmp(v, "w4a16") == 0;
    }();
    return enabled;
}
// This selector is consumed ONLY by prefill and its startup/session identity.
// It never changes the CPU expert format, decode, MTP or KV attention dispatch.
// Immutable for graph/workspace lifetimes. Unset and w4a8 keep the old path.
inline bool native_nvfp4_prefill_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_PREFILL_NVFP4");
        if (!v || (std::strcmp(v, "w4a4") != 0 && std::strcmp(v, "w4a16") != 0)) return false;
#if !defined(STRATA_NVFP4_PREFILL_BUILT) || defined(STRATA_USE_HIP)
        throw std::runtime_error("W4A4/W4A16 prefill needs -DSTRATA_NVFP4_PREFILL=ON on CUDA SM120");
#else
        if (!native_nvfp4_prefill_w4a16()) (void) nvfp4_prefill_backend();
        return true;
#endif
    }();
    return enabled;
}
}
