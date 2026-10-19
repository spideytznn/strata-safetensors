#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace strata::kernels::cpu {
// Positive E4M3 finite codes, including subnormals. Round-to-nearest-even,
// saturating at 448, as CUDA __nv_cvt_float_to_fp8(..., SATFINITE, E4M3).
inline float nvfp4_e4m3_value(unsigned c) {
    return c < 8 ? std::ldexp(float(c), -9)
                 : std::ldexp(float(8 + (c & 7)), int(c >> 3) - 10);
}
inline uint8_t nvfp4_e4m3_code(float x) {
    static const auto values = [] {
        std::array<float,127> a{};
        for (unsigned c=0; c<a.size(); ++c) a[c]=nvfp4_e4m3_value(c);
        return a;
    }();
    auto it=std::lower_bound(values.begin(), values.end(), x);
    if (it==values.begin()) return 0;
    if (it==values.end()) return 126;
    unsigned hi=unsigned(it-values.begin()), lo=hi-1;
    const float dl=x-values[lo], dh=values[hi]-x;
    return uint8_t(dl<dh || (dl==dh && !(lo&1)) ? lo : hi);
}
inline uint8_t nvfp4_e2m1_code(float x) {
    constexpr float midpoint[]={.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f};
    const float a=std::fabs(x);
    unsigned c=0;
    while(c<7 && (a>midpoint[c] || (a==midpoint[c] && (c&1)))) ++c;
    return uint8_t(c | (std::signbit(x) ? 8 : 0));
}
inline float nvfp4_e2m1_value(unsigned c) {
    constexpr float values[]={0,.5f,1,1.5f,2,3,4,6};
    return std::copysign(values[c&7], (c&8) ? -1.f : 1.f);
}
// CPU has no FP4 MMA instruction: retain the FP4-quantized values expanded
// to FP32 for AVX512 FMA. The global scale is applied AFTER the dot product.
// Called once per expert/token, never once per output row or worker task.
inline void nvfp4_quantize_w4a4(const float* x, float* out, int n, float global) {
    for(int base=0; base<n; base+=16) {
        float v[16], amax=0;
        for(int j=0;j<16;++j) {v[j]=x[base+j]/global; amax=std::max(amax,std::fabs(v[j]));}
        const float scale=nvfp4_e4m3_value(nvfp4_e4m3_code(amax/6.f));
        const float inv=scale>0 ? 1.f/scale : 0.f;
        for(int j=0;j<16;++j)
            out[base+j]=nvfp4_e2m1_value(nvfp4_e2m1_code(v[j]*inv))*scale;
    }
}
} // namespace strata::kernels::cpu
