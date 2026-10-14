#include "strata/kernels/nvfp4_w4a16.hpp"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int N = 2560, F = 640;
size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }
__device__ float fp4(unsigned c) {
    const float v[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -v[c & 7] : v[c & 7];
}
__device__ uint16_t bf16(float x) { return __bfloat16_as_ushort(__float2bfloat16_rn(x)); }
__device__ float weight(const uint8_t* p, int i) {
    const uint8_t* block = p + (i / 64) * 36;
    const int sub = (i % 64) / 16, col = i % 16;
    const unsigned c = (block[4 + sub * 8 + col % 8] >> (col < 8 ? 0 : 4)) & 15;
    __nv_fp8_e4m3 s; s.__x = block[sub];
    return fp4(c) * float(s);
}
__global__ void expand(NativeExpertLayout L, const uint8_t* p, uint16_t* gu, uint16_t* dn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < F * N) {
        gu[i] = bf16(weight(p, i));
        gu[F * N + i] = bf16(weight(p + L.up_off, i));
        dn[i] = bf16(weight(p + L.down_off, i));
    }
}
__global__ void gather(const float* x, const int32_t* ids, uint16_t* out, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * N) out[i] = bf16(x[size_t(ids[i / N]) * N + i % N]);
}
__global__ void hidden(const float* gu, const float* scales, uint16_t* h, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * F) return;
    const int row = i / F, col = i % F;
    const float g = gu[row * 2 * F + col] * scales[0];
    const float u = gu[row * 2 * F + F + col] * scales[1];
    h[i] = bf16((g / (1.f + expf(-g))) * u);
}
__global__ void finish(float* y, const float* scales, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] *= scales[2];
}
void check() {
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("W4A16 prefill: ") + cudaGetErrorString(e));
}
}
size_t nvfp4_w4a16_prefill_bytes(int64_t rows) {
    if (rows <= 0) return 0;
    if (rows > std::numeric_limits<int>::max() / N)
        throw std::invalid_argument("W4A16 prefill row count overflows indexing");
    return aligned(2ull * F * N * 2) + aligned(1ull * N * F * 2) +
           aligned(size_t(rows) * N * 2) + aligned(size_t(rows) * F * 2) +
           aligned(size_t(rows) * 2 * F * 4);
}
Nvfp4W4A16Buffers nvfp4_w4a16_buffers(void* scratch, int64_t rows) {
    (void) nvfp4_w4a16_prefill_bytes(rows);
    if (!scratch || rows <= 0) throw std::invalid_argument("W4A16 prefill requires workspace and rows");
    auto* p = static_cast<uint8_t*>(scratch);
    Nvfp4W4A16Buffers b{};
    b.gate_up = reinterpret_cast<uint16_t*>(p); p += aligned(2ull * F * N * 2);
    b.down = reinterpret_cast<uint16_t*>(p); p += aligned(1ull * N * F * 2);
    b.input = reinterpret_cast<uint16_t*>(p); p += aligned(size_t(rows) * N * 2);
    b.hidden = reinterpret_cast<uint16_t*>(p); p += aligned(size_t(rows) * F * 2);
    b.projected = reinterpret_cast<float*>(p);
    return b;
}
void nvfp4_w4a16_prepare(const NativeExpertLayout& L, const uint8_t* blob,
    const int32_t* tokens, int rows, const float* input, Nvfp4W4A16Buffers b, void* stream) {
    const auto st = static_cast<cudaStream_t>(stream);
    expand<<<(F * N + 255) / 256, 256, 0, st>>>(L, blob, b.gate_up, b.down);
    gather<<<(rows * N + 255) / 256, 256, 0, st>>>(input, tokens, b.input, rows);
    check();
}
void nvfp4_w4a16_hidden(const NativeExpertLayout& L, const uint8_t* blob,
    int rows, Nvfp4W4A16Buffers b, void* stream) {
    hidden<<<(rows * F + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        b.projected, reinterpret_cast<const float*>(blob + L.tail_off), b.hidden, rows);
    check();
}
void nvfp4_w4a16_finish(const NativeExpertLayout& L, const uint8_t* blob,
    int rows, float* output, void* stream) {
    finish<<<(rows * N + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        output, reinterpret_cast<const float*>(blob + L.tail_off), rows * N);
    check();
}
}
