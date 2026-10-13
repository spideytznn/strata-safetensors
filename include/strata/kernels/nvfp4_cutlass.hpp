#pragma once
#include "strata/kernels/nvfp4_tensor.hpp"
namespace strata::kernels {
bool nvfp4_cutlass_enabled();
bool nvfp4_cutlass_use_group(const Nvfp4PrefillGroup&);
size_t nvfp4_cutlass_prefill_bytes(int64_t max_rows);
void nvfp4_cutlass_prefill(const NativeExpertLayout&, Nvfp4PrefillGroup,
    const int32_t* tokens, const float* x, void* scratch, float* out, void* stream);
}
