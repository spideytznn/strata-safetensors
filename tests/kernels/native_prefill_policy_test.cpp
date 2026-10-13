#include "strata/kernels/native_prefill.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string expected = argv[1];
    try {
        const bool enabled = strata::kernels::native_nvfp4_prefill_enabled();
        const bool ok = expected == (enabled ? "on" : "off");
        if (ok) std::puts(enabled ? "prefill selector: enabled" : "prefill selector: original path");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("prefill selector refused: %s\n", e.what());
        return expected == "error" ? 0 : 1;
    }
}
