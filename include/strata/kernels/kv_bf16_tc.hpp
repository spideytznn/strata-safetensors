#pragma once
#include "strata/kernels/kv_q8.hpp"

namespace strata::kernels {
void kv_append_bf16_steps(uint16_t* k,uint16_t* v,const int32_t* table,
    const int32_t* steps,int step_stride,const float* kc,const float* vc,int cur_stride,int n,
    const QsaShapes& s,void* stream,const KvHostPools* host);
void kv_append_bf16_prefill(const float* k,const float* v,int64_t n,int64_t pos,
    const int32_t* table,int page_size,uint16_t* ko,uint16_t* vo,void* stream,
    const KvHostPools* host,const KvHostPools* stage);
// Writes the same partial-accumulator/m/l layout and uses the existing FP32 merge.
void qsa_bf16_tc_chunks(const float* q, const QsaAttnPools& pools, const int32_t* ids,
    const int32_t* steps, int64_t cap, const QsaShapes& s, float* scratch,
    int64_t n_q, void* stream);
void qsa_bf16_tc_prompt(const float* q, const QsaAttnPools& pools, const int32_t* ids,
    const int32_t* steps, int64_t cap, const QsaShapes& s, float* attn,
    int64_t n_q, void* stream);
}
