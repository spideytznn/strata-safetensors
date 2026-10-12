// Two image histories at the SAME cell index must keep distinct query/key and indexer rotations.
// Captured graphs must ignore later rewrites of the solo reader's table and observe slot table updates.
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc((void**) &p, n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    void put(const std::vector<T>& x) { ck(cudaMemcpy(p, x.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const {
        std::vector<T> x(n); ck(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x;
    }
    void zero() { ck(cudaMemset(p, 0, n * sizeof(T))); }
};
struct State {
    Device<float> rotated{256}, tail{3 * 128}, dead{128}, pooled{17 * 128};
    Device<int32_t> block{1};
    void zero() { rotated.zero(); tail.zero(); dead.zero(); pooled.zero(); block.zero(); }
    k::QsaIndexerBuffers buffers() { return {tail.p, dead.p, pooled.p, block.p}; }
    bool equals(const State& x) const {
        // vector<float>::operator== would conceal signed zero differences. Require the same bits.
        auto equal = [](const auto& a, const auto& b) {
            return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
        };
        return equal(rotated.get(), x.rotated.get()) && equal(tail.get(), x.tail.get()) &&
               equal(dead.get(), x.dead.get()) && equal(pooled.get(), x.pooled.get()) && block.get() == x.block.get();
    }
};
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        Device<int32_t> main_table(64 * 3), a_table(64 * 3), b_table(64 * 3), pos(1);
        Device<float> input(256), gamma(256), raw(128), index_gamma(128);
        std::vector<int32_t> identity(64 * 3), a(64 * 3), b(64 * 3);
        for (int p = 0; p < 64; ++p) for (int j = 0; j < 3; ++j) {
            identity[p * 3 + j] = p;
            a[p * 3 + j] = p >= 4 ? 4 + (p - 4) / (j + 1) : p;
            b[p * 3 + j] = p >= 2 ? 2 + (p - 2) / (3 - j) : p;
        }
        std::vector<float> x(256), r(128);
        for (int i = 0; i < 256; ++i) x[i] = float((i * 13) % 37 - 18) / 19;
        for (int i = 0; i < 128; ++i) r[i] = float((i * 7) % 23 - 11) / 13;
        input.put(x); raw.put(r); gamma.put(std::vector<float>(256, 1));
        index_gamma.put(std::vector<float>(128, 1)); pos.put({7});
        main_table.put(identity); a_table.put(a); b_table.put(b);
        State actual_a, actual_b, ref_a, ref_b;
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        k::QsaShapes shapes{}; shapes.idx_dim = 128; shapes.idx_block = 4; shapes.n_rot = 64;
        k::RopeScaling scaling;
        auto launch = [&](State& state, const int32_t* table) {
            k::mrope_table_set(table);
            k::native_qsa_rms_norm_rope(input.p, 256, gamma.p, state.rotated.p, 1, 256, 64, 1e-6f,
                                       scaling, pos.p, stream);
            k::native_qsa_indexer_append(raw.p, pos.p, 0, index_gamma.p, 1e-6f, state.buffers(),
                                         shapes, 64, scaling, stream);
            k::mrope_table_set(main_table.p);
        };
        for (State* state : {&actual_a, &actual_b, &ref_a, &ref_b}) state->zero();
        ck(cudaDeviceSynchronize());
        cudaGraph_t graph; cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(actual_a, a_table.p); launch(actual_b, b_table.p);
        ck(cudaStreamEndCapture(stream, &graph));
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int round = 0; round < 2; ++round) {
            if (round) a_table.put(identity);  // the slot is reused for text; pointer stays fixed
            for (State* state : {&actual_a, &actual_b, &ref_a, &ref_b}) state->zero();
            main_table.put(b);               // an unrelated incoming image rewrites the main request table
            ck(cudaDeviceSynchronize());
            launch(ref_a, a_table.p); launch(ref_b, b_table.p);
            ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
            if (!actual_a.equals(ref_a) || !actual_b.equals(ref_b) || ref_a.rotated.get() == ref_b.rotated.get())
                throw std::runtime_error("per-slot image rotation/indexer state mismatch");
        }
        k::mrope_table_set(nullptr);
        ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
        std::puts("PASS: image/image and text/image at equal cells; captured per-slot RoPE and pooled indexer bit-identical to solo");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
