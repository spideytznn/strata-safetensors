// CUDA slot commits: batch layers within one session, retaining that session's controls and M-RoPE.
#pragma once

#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"

namespace strata::core {

struct SlotCommitLayers {
    float* state = nullptr;
    int64_t state_stride = 0, conv_offset = 0;
    const float *qkv = nullptr, *h = nullptr, *gate = nullptr, *beta = nullptr;
    int64_t qkv_stride = 0, h_stride = 0, gate_stride = 0;
    int conv_channels = 0, h_k = 0, h_v = 0, steps = 0, gdn_layers = 0;
    const int32_t* control = nullptr; // [accepted count, PLE snapshot index, accepted positions or -1 ...]
    const float* index_raw = nullptr;
    int64_t index_stride = 0, max_cells = 0;
    kernels::QsaIndexerLayers indexer;
    int qsa_layers = 0;
    kernels::QsaShapes shapes{};
    float epsilon = 1e-6f;
    const int32_t* mrope = nullptr;
};

struct SlotCommitResult {
    bool gdn = false, qsa = false;
};

// Records only state updates. A false result leaves that part to the original per-layer loop.
// No device allocation, synchronization or cross-session state. Safe during graph capture.
inline SlotCommitResult commit_slot_layers(const SlotCommitLayers& a,
                                           const kernels::RopeScaling& scaling, void* stream) {
    SlotCommitResult result;
    // The original one-row path also computes dummy outputs, and uses another kernel.
    // Keep it unchanged; multi-row commit already uses the identical state-only arithmetic.
    if (a.steps > 1 && a.gdn_layers > 0)
        result.gdn = kernels::gdn_commit_layers(
            a.state, a.state_stride, a.conv_offset, a.qkv, a.qkv_stride,
            a.h, a.h_stride, a.gate, a.beta, a.gate_stride, a.conv_channels,
            a.h_k, a.h_v, a.steps, a.gdn_layers, a.control, stream);
    if (a.qsa_layers > 0) {
        struct Positions {
            const int32_t* saved = kernels::mrope_table();
            explicit Positions(const int32_t* table) { kernels::mrope_table_set(table); }
            ~Positions() { kernels::mrope_table_set(saved); }
        } positions(a.mrope);
        kernels::native_qsa_indexer_append_steps_layers(
            a.index_raw, a.index_stride, a.control + 2, 1, a.steps, 0, a.epsilon,
            a.indexer, a.qsa_layers, a.shapes, a.max_cells, scaling, stream);
        result.qsa = true;
    }
    return result;
}

} // namespace strata::core
