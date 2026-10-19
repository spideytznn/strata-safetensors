#include "strata/weights/dense.hpp"
#include "json_checked.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <set>

namespace strata::weights {
using detail::require;
namespace {
uint64_t permute(uint64_t i, uint64_t head) {
    const auto h = i / head;
    return ((h % 16) * 3 + h / 16) * head + i % head;
}
}
uint16_t mxfp8_bf16(uint8_t weight, uint8_t scale) {
    if ((weight & 127) == 127 || scale == 255) throw std::runtime_error("nonfinite MXFP8 weight/UE8M0 scale");
    const uint16_t sign = uint16_t(weight & 128) << 8;
    const int e = (weight >> 3) & 15;
    unsigned significand = weight & 7;
    int exponent = int(scale) + e - 7; // BF16 biased exponent
    if (e) significand = 128 + significand * 16;
    else {
        if (!significand) return sign;
        exponent = int(scale) - 2;
        while (significand < 128) { significand <<= 1; --exponent; }
    }
    if (exponent >= 255) throw std::runtime_error("MXFP8 weight overflows BF16");
    if (exponent <= 0) {
        const int shift = 1 - exponent;
        if (shift > 8) return sign;
        // BF16 subnormal rounding, ties to even; regular checkpoint values
        // need no rounding because E4M3 has fewer significand bits than BF16.
        const unsigned half = 1u << (shift-1);
        significand = (significand + half-1 + ((significand >> shift)&1)) >> shift;
        return static_cast<uint16_t>(sign | significand);
    }
    return static_cast<uint16_t>(sign | (exponent << 7) | (significand-128));
}
Qwen4DensePlan::Qwen4DensePlan(SafetensorsSource& source) {
    Qwen4Config::load(source);
    std::set<std::string> bound;
    const std::string root = "model.language_model.";
    auto add = [&](const std::string& hf, const std::string& name, uint64_t rows, uint64_t cols,
                   DType output = DType::BF16, DenseMath math = DenseMath::Identity,
                   bool native = false, uint64_t head = 0, uint64_t prefix = 0,
                   uint64_t col_head = 0, uint64_t first = 0, uint64_t source_rows = 0) {
        const auto& t = source.tensor(hf);
        const auto total_rows = source_rows ? source_rows : rows;
        const auto& s = t.physical_shape;
        const bool mxfp8 = t.dtype == DType::F8_E4M3;
        require((t.dtype == DType::BF16 || mxfp8 || (t.dtype == DType::F32 && output == DType::F32)) && (s == std::vector<uint64_t>{total_rows,cols} ||
            (cols == 1 && s == std::vector<uint64_t>{total_rows}) ||
            s == std::vector<uint64_t>{total_rows,1,cols}), "invalid dense shape/dtype: " + hf);
        require(first + rows <= total_rows, "invalid dense row slice");
        tensors.push_back({name,&t,rows,cols,first,head,prefix,col_head,output,math,native});
        if (mxfp8) {
            require(hf.ends_with(".weight") && cols % 32 == 0 && s.size() == 2,
                    "unsupported MXFP8 dense geometry: " + hf);
            const auto& scale = source.tensor(hf.substr(0,hf.size()-7)+".weight_scale");
            require(scale.dtype == DType::U8 && scale.physical_shape == std::vector<uint64_t>{total_rows,cols/32},
                    "invalid MXFP8 UE8M0 scale: " + scale.name);
            tensors.back().block_scale = &scale;
            bound.insert(scale.name);
        }
        bound.insert(hf);
    };
    add(root+"embed_tokens.weight","token_embd.weight",248320,2560);
    add("lm_head.weight","output.weight",248320,2560,DType::BF16,DenseMath::Identity,true);
    add(root+"hyper_connection_mixer.hc_norm.weight","output_hc_norm.weight",10240,1,DType::F32,DenseMath::AddOne);
    add(root+"hyper_connection_mixer.input_mix_weight_down.weight","output_hc_down.weight",320,10240);
    add(root+"hyper_connection_mixer.input_mix_weight_up.weight","output_hc_up.weight",10240,320);
    for (int l = 0; l < 48; ++l) {
        const auto hf = root+"layers."+std::to_string(l)+".";
        const auto dst = "blk."+std::to_string(l)+".";
        auto a = [&](const char* src, const char* target, uint64_t r, uint64_t c,
                     DType type = DType::BF16, DenseMath math = DenseMath::Identity,
                     bool native = false, uint64_t head = 0, uint64_t prefix = 0,
                     uint64_t col_head = 0, uint64_t first = 0, uint64_t source_rows = 0) {
            add(hf+src,dst+target,r,c,type,math,native,head,prefix,col_head,first,source_rows);
        };
        for (const auto& p : {std::pair{"attn_hyper_connection.","hc_attn_"},
                              std::pair{"mlp_hyper_connection.","hc_ffn_"}}) {
            add(hf+p.first+"block_inject_weight.weight",dst+p.second+"inject.weight",4,10240);
            add(hf+p.first+"hc_norm.weight",dst+p.second+"norm.weight",10240,1,DType::F32,DenseMath::AddOne);
            add(hf+p.first+"input_mix_weight_down.weight",dst+p.second+"down.weight",320,10240);
            add(hf+p.first+"input_mix_weight_up.weight",dst+p.second+"up.weight",10240,320);
        }
        a("mlp.gate.weight","ffn_gate_inp.weight",512,2560);
        a("mlp.shared_expert_gate.weight","ffn_gate_inp_shexp.weight",1,2560);
        a("mlp.shared_expert.gate_proj.weight","ffn_gate_shexp.weight",640,2560,DType::BF16,DenseMath::Identity,true);
        a("mlp.shared_expert.up_proj.weight","ffn_up_shexp.weight",640,2560,DType::BF16,DenseMath::Identity,true);
        a("mlp.shared_expert.down_proj.weight","ffn_down_shexp.weight",2560,640,DType::BF16,DenseMath::Identity,true);
        if (l % 4 != 3) {
            a("linear_attn.A_log","ssm_a",48,1,DType::F32,DenseMath::NegativeExp,false,1);
            a("linear_attn.dt_bias","ssm_dt.bias",48,1,DType::F32,DenseMath::Identity,false,1);
            a("linear_attn.conv1d.weight","ssm_conv1d.weight",10240,4,DType::F32,DenseMath::Identity,false,128,4096);
            a("linear_attn.in_proj_a.weight","ssm_alpha.weight",48,2560,DType::BF16,DenseMath::Identity,false,1);
            a("linear_attn.in_proj_b.weight","ssm_beta.weight",48,2560,DType::BF16,DenseMath::Identity,false,1);
            a("linear_attn.in_proj_qkv.weight","attn_qkv.weight",10240,2560,DType::BF16,DenseMath::Identity,true,128,4096);
            a("linear_attn.in_proj_z.weight","attn_gate.weight",6144,2560,DType::BF16,DenseMath::Identity,true,128);
            a("linear_attn.norm.weight","ssm_norm.weight",128,1,DType::F32);
            a("linear_attn.out_proj.weight","ssm_out.weight",2560,6144,DType::BF16,DenseMath::Identity,true,0,0,128);
        } else {
            a("self_attn.indexer.index_qk_proj.weight","indexer.q_proj.weight",512,2560,DType::BF16,DenseMath::Identity,false,0,0,0,0,640);
            a("self_attn.indexer.index_qk_proj.weight","indexer.k_proj.weight",128,2560,DType::BF16,DenseMath::Identity,false,0,0,0,512,640);
            a("self_attn.indexer.q_layernorm.weight","indexer.q_norm.weight",128,1,DType::F32,DenseMath::AddOne);
            a("self_attn.indexer.k_layernorm.weight","indexer.k_norm.weight",128,1,DType::F32,DenseMath::AddOne);
            a("self_attn.q_norm.weight","attn_q_norm.weight",256,1,DType::F32,DenseMath::AddOne);
            a("self_attn.k_norm.weight","attn_k_norm.weight",256,1,DType::F32,DenseMath::AddOne);
            a("self_attn.q_proj.weight","attn_q.weight",12288,2560,DType::BF16,DenseMath::Identity,true);
            a("self_attn.k_proj.weight","attn_k.weight",512,2560,DType::BF16,DenseMath::Identity,true);
            a("self_attn.v_proj.weight","attn_v.weight",512,2560,DType::BF16,DenseMath::Identity,true);
            a("self_attn.o_proj.weight","attn_output.weight",2560,6144,DType::BF16,DenseMath::Identity,true);
        }
        if (l == 1) {
            a("ple.conv1d.weight","ple_conv1d.weight",10240,4);
            a("ple.key_proj.weight","ple_key.weight",10240,2560);
            a("ple.value_proj.weight","ple_value.weight",2560,2560);
            a("ple.norm_conv.weight","ple_norm_conv.weight",10240,1,DType::F32,DenseMath::AddOne);
            a("ple.norm_key.weight","ple_norm_key.weight",10240,1,DType::F32,DenseMath::AddOne);
            a("ple.norm_query.weight","ple_norm_query.weight",10240,1,DType::F32,DenseMath::AddOne);
        }
    }
    // Refuse silently omitted ordinary tensors. Integer PLE constants have a
    // separate exact I64 binding; they must never pass through a float adapter.
    for (const auto& [name,t] : source.tensors()) if (t.family == Family::Dense)
        require(bound.contains(name) || name == root+"layers.1.ple.ple_embedding.layer_multipliers",
                "unbound dense tensor: " + name);
}
const DenseBinding& Qwen4DensePlan::find(const std::string& name) const {
    for (const auto& t : tensors) if (t.name == name) return t;
    throw std::runtime_error("dense binding not found: "+name);
}
void stream_dense(WeightSource& source, const DenseBinding& b,
                  const std::function<void(uint64_t,std::span<const uint8_t>)>& emit, uint64_t tile_bytes) {
    const bool mxfp8 = b.source && b.source->dtype == DType::F8_E4M3;
    const bool f32 = b.source && b.source->dtype == DType::F32;
    require(b.source && (b.source->dtype == DType::BF16 || mxfp8 || (f32 && b.output == DType::F32)) && b.rows && b.columns &&
        (b.output == DType::BF16 || b.output == DType::F32), "invalid dense stream binding");
    require(b.math == DenseMath::Identity || b.output == DType::F32, "dense arithmetic must output FP32");
    require(b.row_prefix <= b.rows && (!b.row_head_size || b.rows-b.row_prefix == 48*b.row_head_size) &&
        (!b.column_head_size || b.columns == 48*b.column_head_size), "invalid GDN permutation geometry");
    if (mxfp8) require(b.block_scale && b.block_scale->dtype == DType::U8 && b.columns % 32 == 0 &&
        b.block_scale->physical_shape == std::vector<uint64_t>{b.source->physical_shape.at(0),b.columns/32},
        "invalid MXFP8 stream scales");
    const auto input_row = b.columns*dtype_bytes(b.source->dtype), output_row = b.columns*dtype_bytes(b.output);
    require(tile_bytes >= input_row && tile_bytes <= 64*1024*1024, "invalid dense tile budget");
    const auto scale_row = mxfp8 ? b.columns/32 : 0;
    const auto tile_rows = std::max<uint64_t>(1,tile_bytes / (input_row+output_row+scale_row));
    std::vector<uint8_t> input(tile_rows*input_row), output(tile_rows*output_row);
    std::vector<uint8_t> scales(tile_rows*scale_row);
    std::vector<ReadRequest> reads;
    for (uint64_t first = 0; first < b.rows; first += tile_rows) {
        const auto count = std::min(tile_rows,b.rows-first);
        reads.clear();
        for (uint64_t row = 0; row < count; ++row) {
            auto src_row = first+row;
            if (b.row_head_size && src_row >= b.row_prefix)
                src_row = b.row_prefix+permute(src_row-b.row_prefix,b.row_head_size);
            reads.push_back({b.source,(src_row+b.first_row)*input_row,
                             std::span(input).subspan(row*input_row,input_row)});
            if (mxfp8) reads.push_back({b.block_scale,(src_row+b.first_row)*scale_row,
                                       std::span(scales).subspan(row*scale_row,scale_row)});
        }
        source.read_many(reads);
        for (uint64_t row = 0; row < count; ++row) for (uint64_t col = 0; col < b.columns; ++col) {
            const auto src_col = b.column_head_size ? permute(col,b.column_head_size) : col;
            uint16_t bits = 0;
            if (mxfp8) bits = mxfp8_bf16(input[row*input_row+src_col],scales[row*scale_row+src_col/32]);
            else if (!f32) std::memcpy(&bits,input.data()+row*input_row+src_col*2,2);
            auto* dst = output.data()+row*output_row+col*dtype_bytes(b.output);
            if (b.output == DType::BF16) std::memcpy(dst,&bits,2);
            else {
                float value = std::bit_cast<float>(static_cast<uint32_t>(bits)<<16);
                if (f32) std::memcpy(&value,input.data()+row*input_row+src_col*4,4);
                if (b.math == DenseMath::AddOne) value += 1.0f;
                else if (b.math == DenseMath::NegativeExp) value = -std::exp(value);
                require(std::isfinite(value),"non-finite dense arithmetic: "+b.name);
                std::memcpy(dst,&value,4);
            }
        }
        emit(first*output_row,std::span(output).first(count*output_row));
    }
}
} // namespace strata::weights
