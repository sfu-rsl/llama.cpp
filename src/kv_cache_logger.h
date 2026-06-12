#pragma once

/**
 * kv_cache_logger.h
 * -----------------
 * Drop-in instrumentation for the llama.cpp KV-cache lifecycle.
 * Uses LLAMA_LOG_INFO for output — respects llama_log_set_callback,
 * log levels, and existing file routing. No separate .cpp needed.
 *
 * Three lifecycle phases are tracked:
 *
 *   INIT   – structural allocation in llama_kv_cache constructor
 *   FILL   – token slots written during apply_ubatch / prepare
 *   CLEAR  – cells freed via clear(), seq_rm(), or seq_keep() eviction
 *
 * Output format (one JSON-Lines record per event, via LLAMA_LOG_INFO)
 * --------------------------------------------------------------------
 *  {"ts_us":1234567,"phase":"INIT","event":"cache_created",
 *   "kv_size":4096,"n_layer":32,"n_stream":1,
 *   "type_k":"f16","type_v":"f16","offload":true}
 *
 *  {"ts_us":1234600,"phase":"FILL","event":"batch_applied",
 *   "n_tokens":1,"n_stream":1,"head_before":42}
 *
 *  {"ts_us":1234900,"phase":"CLEAR","event":"seq_rm",
 *   "seq_id":0,"p0":0,"p1":42,"cells_freed":17}
 */

#include "llama-impl.h"  // LLAMA_LOG_INFO

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

// ─── timestamp ───────────────────────────────────────────────────────────────

namespace kv_log {
inline int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace kv_log

// ─── macros ──────────────────────────────────────────────────────────────────

#define KV_LOG_INIT(kv_size_, n_layer_, n_stream_, type_k_, type_v_, offload_)                                  \
    do {                                                                                                        \
        LLAMA_LOG_INFO(                                                                                         \
            "{\"ts_us\":%lld,\"phase\":\"INIT\",\"event\":\"cache_created\","                                   \
            "\"kv_size\":%u,\"n_layer\":%u,\"n_stream\":%u,"                                                    \
            "\"type_k\":\"%s\",\"type_v\":\"%s\",\"offload\":%s}\n",                                            \
            (long long) kv_log::now_us(), (unsigned) (kv_size_), (unsigned) (n_layer_), (unsigned) (n_stream_), \
            ggml_type_name(type_k_), ggml_type_name(type_v_), (offload_) ? "true" : "false");                   \
    } while (0)

#define KV_LOG_BATCH_FILL(sinfo_, n_tokens_, head_start_)                                         \
    do {                                                                                          \
        LLAMA_LOG_INFO(                                                                           \
            "{\"ts_us\":%lld,\"phase\":\"FILL\",\"event\":\"batch_applied\","                     \
            "\"n_tokens\":%u,\"n_stream\":%u,\"head_before\":%u}\n",                              \
            (long long) kv_log::now_us(), (unsigned) (n_tokens_), (unsigned) (sinfo_).n_stream(), \
            (unsigned) (head_start_));                                                            \
    } while (0)

#define KV_LOG_FILL(stream_idx_, cell_idx_, token_pos_, seq_ids_vec_, n_tokens_, head_after_)                   \
    do {                                                                                                        \
        std::string _sids = "[";                                                                                \
        for (size_t _si = 0; _si < (seq_ids_vec_).size(); ++_si) {                                              \
            if (_si)                                                                                            \
                _sids += ',';                                                                                   \
            _sids += std::to_string((seq_ids_vec_)[_si]);                                                       \
        }                                                                                                       \
        _sids += ']';                                                                                           \
        LLAMA_LOG_INFO(                                                                                         \
            "{\"ts_us\":%lld,\"phase\":\"FILL\",\"event\":\"slot_filled\","                                     \
            "\"stream\":%u,\"cell\":%u,\"pos\":%d,"                                                             \
            "\"seq_ids\":%s,\"n_tokens\":%u,\"head_after\":%u}\n",                                              \
            (long long) kv_log::now_us(), (unsigned) (stream_idx_), (unsigned) (cell_idx_), (int) (token_pos_), \
            _sids.c_str(), (unsigned) (n_tokens_), (unsigned) (head_after_));                                   \
    } while (0)

#define KV_LOG_CLEAR_FULL(data_, n_stream_)                                                    \
    do {                                                                                       \
        LLAMA_LOG_INFO(                                                                        \
            "{\"ts_us\":%lld,\"phase\":\"CLEAR\",\"event\":\"cache_cleared\","                 \
            "\"data_zeroed\":%s,\"n_stream\":%u}\n",                                           \
            (long long) kv_log::now_us(), (data_) ? "true" : "false", (unsigned) (n_stream_)); \
    } while (0)

#define KV_LOG_SEQ_RM(seq_id_, p0_, p1_, cells_freed_)                                                           \
    do {                                                                                                         \
        LLAMA_LOG_INFO(                                                                                          \
            "{\"ts_us\":%lld,\"phase\":\"CLEAR\",\"event\":\"seq_rm\","                                          \
            "\"seq_id\":%d,\"p0\":%d,\"p1\":%d,\"cells_freed\":%u}\n",                                           \
            (long long) kv_log::now_us(), (int) (seq_id_), (int) (p0_), (int) (p1_), (unsigned) (cells_freed_)); \
    } while (0)

#define KV_LOG_SEQ_KEEP(seq_id_, cells_freed_)                                         \
    do {                                                                               \
        LLAMA_LOG_INFO(                                                                \
            "{\"ts_us\":%lld,\"phase\":\"CLEAR\",\"event\":\"seq_keep\","              \
            "\"seq_id\":%d,\"cells_freed\":%u}\n",                                     \
            (long long) kv_log::now_us(), (int) (seq_id_), (unsigned) (cells_freed_)); \
    } while (0)

#define KV_LOG_DEFRAG(n_moved_)                                         \
    do {                                                                \
        LLAMA_LOG_INFO(                                                 \
            "{\"ts_us\":%lld,\"phase\":\"CLEAR\",\"event\":\"defrag\"," \
            "\"cells_moved\":%u}\n",                                    \
            (long long) kv_log::now_us(), (unsigned) (n_moved_));       \
    } while (0)
