#pragma once
#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {
// Original weight codes/scales, one E2M1 activation term, E4M3 scale per 16
// values and checkpoint activation global scale. No residual compensation.
size_t nvfp4_tensor_scratch_bytes(int64_t entries, int64_t embd = 2560, int64_t ff = 640);
void nvfp4_tensor_grouped(const NativeExpertLayout& layout,
    const unsigned long long* ptr, const int32_t* starts, const int32_t* groups,
    const int32_t* dst, const int32_t* tok, int64_t cap_groups, int64_t cap_entries,
    const float* x, void* scratch, float* out, void* stream, int token_tiles = 1);
// Prefill: one expert, token indices into x; output is contiguous in routing order.
// Workspace also covers a streamed group of up to sixteen experts. No allocations.
size_t nvfp4_tensor_prefill_bytes(int64_t entries);
void nvfp4_tensor_prefill(const NativeExpertLayout& layout, const uint8_t* blob,
    const int32_t* tokens, int entries, const float* x, void* scratch, float* out, void* stream);
struct Nvfp4PrefillGroup {
    const uint8_t* weights[16] = {};
    int offsets[16] = {}, rows[16] = {};
    int count = 0;
};
// Input/output offsets refer to the full layer's routing rows. Metadata is passed
// by value to the launch; no asynchronous access to the caller's stack remains.
void nvfp4_tensor_prefill_group(const NativeExpertLayout& layout, Nvfp4PrefillGroup group,
    const int32_t* tokens, const float* x, void* scratch, float* out, void* stream);
}
