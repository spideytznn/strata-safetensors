#pragma once
#include "strata/kernels/nvfp4_tensor.hpp"

namespace strata::kernels {
// One expert at a time. Packed NVFP4 stays the resident format; temporary
// BF16 weights contain FP4 codes times micro-scales, with global scales applied
// after each product. Activations and SwiGLU outputs are rounded once to BF16.
struct Nvfp4W4A16Buffers {
    uint16_t *gate_up, *down, *input, *hidden;
    float* projected;
};
size_t nvfp4_w4a16_prefill_bytes(int64_t rows);
Nvfp4W4A16Buffers nvfp4_w4a16_buffers(void* scratch, int64_t rows);
void nvfp4_w4a16_prepare(const NativeExpertLayout& layout, const uint8_t* blob,
    const int32_t* tokens, int rows, const float* input, Nvfp4W4A16Buffers b, void* stream);
void nvfp4_w4a16_hidden(const NativeExpertLayout& layout, const uint8_t* blob,
    int rows, Nvfp4W4A16Buffers b, void* stream);
void nvfp4_w4a16_finish(const NativeExpertLayout& layout, const uint8_t* blob,
    int rows, float* output, void* stream);
}
