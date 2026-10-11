#include "strata/core/vmm.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;
void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "prefix pages: %s (%s)\n", what, cudaGetErrorString(cudaGetLastError())); std::exit(1); }
}
uint8_t byte(const VmmRange& r, int64_t c) {
    uint8_t result = 0;
    require(cudaMemcpy(&result, r.base() + c * vmm_granularity(), 1, cudaMemcpyDeviceToHost) == cudaSuccess, "read byte");
    return result;
}
uint8_t device_byte(const void* ptr) {
    uint8_t result = 0;
    require(cudaMemcpy(&result, ptr, 1, cudaMemcpyDeviceToHost) == cudaSuccess, "read state byte");
    return result;
}
void state_reset_test() {
    ModelGeometry geometry;
    qsa_set_kv_resident(0);
    qsa_set_kv_int8(true);
    qsa_set_kv_elastic(true, 16384);
    qsa_set_kv_prefix_sharing(true);
    QsaState states[2];
    void* arenas[2]{};
    const int64_t cells = 262144;
    for (int i = 0; i < 2; ++i) {
        const auto bytes = qsa_state_bytes(geometry, cells, i == 0);
        require(cudaMalloc(&arenas[i], bytes) == cudaSuccess, "allocate state arena");
        require(qsa_state_init(geometry, cells, arenas[i], states[i], i ? &states[0] : nullptr) > 0, "initialize elastic state");
        qsa_state_zero(states[i], geometry, nullptr);
    }
    const size_t count = 8192 * geometry.n_head_kv * geometry.head_dim;
    require(cudaMemset(states[0].k_q, 41, count) == cudaSuccess, "fill state K");
    require(cudaMemset(states[0].v_q, 42, count) == cudaSuccess, "fill state V");
    const auto before = qsa_kv_elastic_mapped_bytes();
    std::string error;
    require(conversation_kv_copy(states[0], states[1], geometry, 8192, true, error), error.c_str());
    require(qsa_kv_elastic_mapped_bytes() < before, "shared state saves physical bytes");
    require(device_byte(states[1].k_q) == 41 && device_byte(states[1].v_q) == 42, "cloned state content");
    const std::vector<int32_t> branch{states[1].kv_elastic};
    require(qsa_kv_cow_need(17, 18, &branch) == 2, "code chunks require COW, scale tails stay private");
    require(qsa_kv_prepare_write(17, 18, [] { return VmmChunk(0); }, &branch), "state write detaches chunks");
    require(cudaMemset(states[1].k_q, 99, 1) == cudaSuccess, "rewrite cloned state");
    require(device_byte(states[0].k_q) == 41, "source survives branch overwrite");
    qsa_state_zero(states[0], geometry, nullptr);
    require(cudaDeviceSynchronize() == cudaSuccess, "state reset finishes");
    require(device_byte(states[0].k_q) == 0 && device_byte(states[1].k_q) == 99, "reset detaches source");
    require(device_byte(states[1].v_q + 4096 * geometry.head_dim) == 42, "reset preserves another owner's later page");
    const std::vector<int32_t> both{states[0].kv_elastic, states[1].kv_elastic};
    require(qsa_kv_elastic_grow(139264, [] { return VmmChunk(0); }, &both), "grow beyond one scale chunk");
    const size_t scale_bytes = 131072 * geometry.n_head_kv * (geometry.head_dim / 64) * sizeof(uint16_t);
    require(cudaMemset(states[0].k_scale, 47, scale_bytes) == cudaSuccess, "fill K scales");
    require(cudaMemset(states[0].v_scale, 48, scale_bytes) == cudaSuccess, "fill V scales");
    require(conversation_kv_copy(states[0], states[1], geometry, 131100, true, error), error.c_str());
    require(qsa_kv_cow_need(17, 18, &branch) == 4, "code and scale chunks require COW");
    require(qsa_kv_prepare_write(17, 18, [] { return VmmChunk(0); }, &branch), "scale and code write detaches chunks");
    require(cudaMemset(states[1].k_scale, 99, 1) == cudaSuccess, "rewrite cloned K scale");
    require(device_byte(states[0].k_scale) == 47 && device_byte(states[1].k_scale) == 99, "scale COW isolation");
    require(qsa_kv_prepare_write(70000, 70001, [] { return VmmChunk(0); }, &branch), "detach an excess code page for recycling");
    std::vector<VmmChunk> recycled;
    qsa_set_kv_reset_allocator([&] {
        if (recycled.empty()) return VmmChunk(0);
        const auto chunk = recycled.back(); recycled.pop_back(); return chunk;
    }, [&](VmmChunk chunk) { recycled.push_back(chunk); });
    qsa_state_zero(states[1], geometry, nullptr);
    require(cudaDeviceSynchronize() == cudaSuccess && device_byte(states[1].k_q) == 0, "branch reset preserves zero contract");
    require(qsa_kv_elastic_cells(&branch) == 16384 && !recycled.empty(), "reset recycles excess unique pages");
    qsa_set_kv_reset_allocator({}, {});
    for (const auto chunk : recycled) vmm_chunk_free(chunk);
    for (void* arena : arenas) cudaFree(arena);
    qsa_set_kv_prefix_sharing(false);
}
int main() {
    if (!vmm_available()) { std::puts("prefix pages: skipped (CUDA VMM unavailable)"); return 77; }
    const auto gran = vmm_granularity();
    VmmRange source, branch, third;
    for (auto* r : {&source, &branch, &third}) {
        require(r->reserve(gran * 4), "reserve");
        require(r->map_range(0, 4, [] { return VmmChunk(0); }), "map");
    }
    for (int c = 0; c < 4; ++c)
        require(cudaMemset(source.base() + c * gran, 41 + c, gran) == cudaSuccess, "source fill");
    for (int c = 0; c < 3; ++c) {
        require(branch.share(c, source, c), "share prefix");
        require(third.share(c, branch, c), "share third owner");
        require(source.shared(c) && branch.shared(c) && third.shared(c), "shared identity");
        require(source.handle(c) == branch.handle(c) && branch.handle(c) == third.handle(c), "physical identity");
        require(byte(branch, c) == 41 + c, "shared content");
    }
    auto* stable = branch.base();
    require(branch.make_private(1), "COW middle chunk");
    require(branch.base() == stable && !branch.shared(1), "COW preserves address");
    require(byte(branch, 1) == 42, "COW copies bytes");
    require(cudaMemset(branch.base() + gran, 99, gran) == cudaSuccess, "branch rewrite");
    require(byte(source, 1) == 42 && byte(third, 1) == 42 && byte(branch, 1) == 99, "branch isolation");
    require(source.make_private(0), "source COW");
    require(cudaMemset(source.base(), 77, gran) == cudaSuccess, "source rewrite");
    require(byte(branch, 0) == 41 && byte(third, 0) == 41, "source isolation");
    // Removing an alias must never return a live prefix to the expert cache.
    require(branch.unmap(2) == 0 && !branch.mapped(2), "alias unmap returns no recyclable handle");
    VmmChunk returned = 0;
    const auto previous = branch.handle(3);
    require(branch.share(3, third, 0, &returned) && returned == previous && returned != 0, "alias installation recycles unique replaced backing");
    vmm_chunk_free(returned);
    source.release();
    require(byte(third, 2) == 43 && !third.shared(2), "last owner survives source eviction");
    const auto recyclable = third.unmap(2);
    require(recyclable != 0, "last owner returns recyclable handle");
    vmm_chunk_free(recyclable);
    require(!third.share(3, source, 0), "unmapped source rejected");
    state_reset_test();
    std::puts("prefix pages: sharing, source/branch COW, fixed addresses, eviction and state reset passed");
}
