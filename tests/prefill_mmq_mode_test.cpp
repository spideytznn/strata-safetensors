// Exercise the production MMQ selector without creating a GPU context.
#include "strata/prefill/moe_mmq.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    using namespace strata::prefill::mmq;
    if (argc != 2) return 2;
    const Nvfp4Mode mode = nvfp4_mode();
    // CTest expects rejection to exit nonzero inside the production selector.
    if (std::strcmp(argv[1], "invalid") == 0) return 0;
    const bool w4a8 = std::strcmp(argv[1], "w4a8") == 0;
    const bool w4a4 = std::strcmp(argv[1], "w4a4") == 0;
    const bool fp16 = std::strcmp(argv[1], "fp16") == 0;
    const bool w4a16 = std::strcmp(argv[1], "w4a16") == 0;
    if (!(w4a8 || w4a4 || fp16 || w4a16)) return 3;
    const Nvfp4Mode expected = w4a8 ? Nvfp4Mode::W4A8 : w4a4 ? Nvfp4Mode::W4A4 :
        fp16 ? Nvfp4Mode::FP16 : Nvfp4Mode::W4A16;
    if (mode != expected || supported(40) != (w4a8 || w4a4)) return 4;
    // Non-NVFP4 formats remain available; BF16 mode must not enable A4 quantization.
    if (!supported(8) || fp4_activations(8) || ((fp16 || w4a16) && fp4_activations(40))) return 5;
    std::printf("PASS MMQ selector %s\n", argv[1]);
    return 0;
}
