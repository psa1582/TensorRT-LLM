/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// C ABI for the Hopper XQA BLASST kernel with an online temporal pre-QK predictor.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#if defined(BLASST_TEMPORAL_TESTING)
#include <vector>
#endif

#include <cuda_runtime.h>
#include "mhaTemporal.h"

namespace
{
TemporalQkState* gTemporalState = nullptr;
uint32_t* gTemporalCounters = nullptr;
uint32_t gMaxLayers = 0;
uint32_t gMaxBatch = 0;
uint32_t gMaxHeads = 0;
uint32_t gMaxTiles = 0;
cudaDeviceProp gDeviceProps{};
constexpr uint32_t kTemporalProtectedLayers = 2;
constexpr uint32_t kMaxTimingEvents = 5000;
cudaEvent_t gTimingStart[kMaxTimingEvents]{};
cudaEvent_t gTimingStop[kMaxTimingEvents]{};
uint32_t gTimingCount = 0;
bool gTimingEnabled = false;
bool gStatsEnabled = true;
bool gNullTemporalState = false;
bool gTemporalDisableWrites = false;
uint32_t gTemporalK = 2;
uint32_t gTemporalRefreshInterval = 4;
bool gTemporalPolicyOverride = false;
}

// Configure one policy for subsequent launches without rebuilding the shared library.
// Call blasst_temporal_reset before reusing an already-populated temporal state.
extern "C" int blasst_temporal_set_policy(uint32_t temporal_k, uint32_t temporal_refresh_interval)
{
    if (temporal_k == 0 || temporal_k > temporalRunCountMask || temporal_refresh_interval == 0
        || temporal_refresh_interval > temporalRefreshAgeMask)
    {
        return -1;
    }
    gTemporalK = temporal_k;
    gTemporalRefreshInterval = temporal_refresh_interval;
    gTemporalPolicyOverride = true;
    return 0;
}

extern "C" int blasst_temporal_get_policy(uint32_t* temporal_k, uint32_t* temporal_refresh_interval)
{
    if (temporal_k == nullptr || temporal_refresh_interval == nullptr)
    {
        return -1;
    }
    *temporal_k = gTemporalK;
    *temporal_refresh_interval = gTemporalRefreshInterval;
    return 0;
}

extern "C" int blasst_temporal_prepare(
    uint32_t max_layers, uint32_t max_batch, uint32_t max_heads, uint32_t max_tiles)
{
    if (max_layers == 0 || max_batch == 0 || max_heads == 0 || max_tiles == 0)
    {
        return -1;
    }
    if (gTemporalState != nullptr)
    {
        cudaFree(gTemporalState);
        gTemporalState = nullptr;
    }
    if (gTemporalCounters != nullptr)
    {
        cudaFree(gTemporalCounters);
        gTemporalCounters = nullptr;
    }
    uint64_t const entries
        = static_cast<uint64_t>(max_layers) * max_batch * max_heads * max_tiles;
    if (entries > SIZE_MAX / sizeof(TemporalQkState))
    {
        return -2;
    }
    cudaError_t status = cudaMalloc(&gTemporalState, entries * sizeof(TemporalQkState));
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&gTemporalCounters, 5 * sizeof(uint32_t));
    if (status != cudaSuccess)
    {
        cudaFree(gTemporalState);
        gTemporalState = nullptr;
        return static_cast<int>(status);
    }
    gMaxLayers = max_layers;
    gMaxBatch = max_batch;
    gMaxHeads = max_heads;
    gMaxTiles = max_tiles;
    int device = 0;
    status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    status = cudaGetDeviceProperties(&gDeviceProps, device);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    if (gDeviceProps.major * 10 + gDeviceProps.minor != 90)
    {
        return -3;
    }
    gTimingEnabled = std::getenv("BLASST_KERNEL_TIMING") != nullptr;
    gStatsEnabled = std::getenv("BLASST_DISABLE_STATS") == nullptr;
    gNullTemporalState = std::getenv("BLASST_NULL_TEMPORAL_STATE") != nullptr;
    gTemporalDisableWrites = std::getenv("BLASST_TEMPORAL_DISABLE_WRITES") != nullptr;
    gTimingCount = 0;
    if (gTimingEnabled)
    {
        for (uint32_t i = 0; i < kMaxTimingEvents; ++i)
        {
            if (cudaEventCreate(&gTimingStart[i]) != cudaSuccess
                || cudaEventCreate(&gTimingStop[i]) != cudaSuccess)
            {
                return -5;
            }
        }
    }
    cudaMemset(gTemporalState, 0, entries * sizeof(TemporalQkState));
    cudaMemset(gTemporalCounters, 0, 5 * sizeof(uint32_t));
    return static_cast<int>(cudaGetLastError());
}

extern "C" int blasst_temporal_get_timing(float* total_ms, uint32_t* launches)
{
    if (!gTimingEnabled || total_ms == nullptr || launches == nullptr)
    {
        return -1;
    }
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    float sum = 0.0f;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < gTimingCount; ++i)
    {
        float elapsed = 0.0f;
        status = cudaEventElapsedTime(&elapsed, gTimingStart[i], gTimingStop[i]);
        if (status == cudaSuccess)
        {
            sum += elapsed;
            ++valid;
        }
    }
    *total_ms = sum;
    *launches = valid;
    return 0;
}

extern "C" int blasst_temporal_reset(void* stream_ptr)
{
    if (gTemporalState == nullptr || gTemporalCounters == nullptr)
    {
        return -1;
    }
    uint64_t const entries
        = static_cast<uint64_t>(gMaxLayers) * gMaxBatch * gMaxHeads * gMaxTiles;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t status
        = cudaMemsetAsync(gTemporalState, 0, entries * sizeof(TemporalQkState), stream);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    status = cudaMemsetAsync(gTemporalCounters, 0, 5 * sizeof(uint32_t), stream);
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

#if defined(BLASST_TEMPORAL_TESTING)
extern "C" int blasst_temporal_test_set_null_state(int enabled)
{
    gNullTemporalState = enabled != 0;
    return 0;
}

extern "C" int blasst_temporal_test_fill_layer_state(uint32_t layer_idx, uint32_t packed, void* stream_ptr)
{
    if (gTemporalState == nullptr || layer_idx >= gMaxLayers)
    {
        return -1;
    }
    uint64_t const entries = static_cast<uint64_t>(gMaxBatch) * gMaxHeads * gMaxTiles;
    std::vector<TemporalQkState> values(entries, TemporalQkState{packed});
    TemporalQkState* const layer_state = gTemporalState + layer_idx * entries;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t status
        = cudaMemcpyAsync(layer_state, values.data(), entries * sizeof(TemporalQkState), cudaMemcpyHostToDevice, stream);
    if (status == cudaSuccess)
    {
        status = cudaStreamSynchronize(stream);
    }
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

extern "C" int blasst_temporal_test_count_layer_state_mismatches(
    uint32_t layer_idx, uint32_t expected_packed, uint32_t* mismatches)
{
    if (gTemporalState == nullptr || layer_idx >= gMaxLayers || mismatches == nullptr)
    {
        return -1;
    }
    uint64_t const entries = static_cast<uint64_t>(gMaxBatch) * gMaxHeads * gMaxTiles;
    std::vector<TemporalQkState> values(entries);
    TemporalQkState const* const layer_state = gTemporalState + layer_idx * entries;
    cudaError_t const status
        = cudaMemcpy(values.data(), layer_state, entries * sizeof(TemporalQkState), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    uint32_t mismatch_count = 0;
    for (TemporalQkState const& value : values)
    {
        mismatch_count += value.packed != expected_packed;
    }
    *mismatches = mismatch_count;
    return 0;
}
#endif

extern "C" int blasst_temporal_get_stats(
    uint32_t* candidates, uint32_t* elided, uint32_t* refreshes)
{
    if (gTemporalCounters == nullptr || candidates == nullptr || elided == nullptr || refreshes == nullptr)
    {
        return -1;
    }
    uint32_t values[3]{};
    cudaError_t const status
        = cudaMemcpy(values, gTemporalCounters, sizeof(values), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    *candidates = values[0];
    *elided = values[1];
    *refreshes = values[2];
    return 0;
}

extern "C" int blasst_temporal_get_extended_stats(
    uint32_t* candidates,
    uint32_t* elided,
    uint32_t* refreshes,
    uint32_t* stock_skipped,
    uint32_t* stock_total)
{
    if (gTemporalCounters == nullptr || candidates == nullptr || elided == nullptr || refreshes == nullptr
        || stock_skipped == nullptr || stock_total == nullptr)
    {
        return -1;
    }
    uint32_t values[5]{};
    cudaError_t const status
        = cudaMemcpy(values, gTemporalCounters, sizeof(values), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        return static_cast<int>(status);
    }
    *candidates = values[0];
    *elided = values[1];
    *refreshes = values[2];
    *stock_skipped = values[3];
    *stock_total = values[4];
    return 0;
}

extern "C" int blasst_xqa_bf16_temporal(
    void* query,
    void* key_cache,
    void* value_cache,
    void* output,
    int32_t const* page_list,
    uint32_t const* sequence_length,
    uint32_t max_sequence_length,
    uint32_t num_kv_heads,
    float skip_softmax_threshold_scale_factor,
    float const* kv_cache_scale,
    uint32_t* skipped_blocks,
    uint32_t* total_blocks,
    uint32_t layer_idx,
    float temporal_epsilon,
    uint32_t temporal_refresh_interval,
    uint32_t* semaphores,
    void* scratch,
    void* stream_ptr)
{
    if (!query || !key_cache || !value_cache || !output || !page_list || !sequence_length || !semaphores || !scratch
        || !gTemporalState || !gTemporalCounters)
    {
        return -1;
    }
    uint32_t const required_tiles = (max_sequence_length + 63U) / 64U;
    if (layer_idx >= gMaxLayers || num_kv_heads > gMaxHeads || required_tiles > gMaxTiles)
    {
        return -4;
    }
    cudaError_t status = cudaSuccess;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    uint32_t timing_slot = kMaxTimingEvents;
    if (gTimingEnabled && gTimingCount < kMaxTimingEvents)
    {
        timing_slot = gTimingCount++;
        status = cudaEventRecord(gTimingStart[timing_slot], stream);
        if (status != cudaSuccess)
        {
            return static_cast<int>(status);
        }
    }
    try
    {
        uint32_t const effective_refresh_interval
            = gTemporalPolicyOverride ? gTemporalRefreshInterval : temporal_refresh_interval;
        bool const protectLayerFromTemporalQkSkip = layer_idx < kTemporalProtectedLayers;
        TemporalQkState* const effectiveTemporalState
            = (gNullTemporalState || protectLayerFromTemporalQkSkip) ? nullptr : gTemporalState;
        launchHopperF8MHA(
            gDeviceProps,
            num_kv_heads,
            1.0f,
            reinterpret_cast<OutputHead*>(output),
            reinterpret_cast<InputHead const*>(query),
            nullptr,
            reinterpret_cast<GMemCacheHead*>(key_cache),
            page_list,
            max_sequence_length,
            sequence_length,
            1,
            kv_cache_scale,
            skip_softmax_threshold_scale_factor,
            gStatsEnabled ? (skipped_blocks != nullptr ? skipped_blocks : gTemporalCounters + 3) : nullptr,
            gStatsEnabled ? (total_blocks != nullptr ? total_blocks : gTemporalCounters + 4) : nullptr,
            effectiveTemporalState,
            gMaxTiles,
            gMaxHeads,
            gMaxBatch,
            layer_idx,
            temporal_epsilon,
            gTemporalK,
            effective_refresh_interval,
            gTemporalDisableWrites,
            gStatsEnabled ? gTemporalCounters : nullptr,
            gStatsEnabled ? gTemporalCounters + 1 : nullptr,
            gStatsEnabled ? gTemporalCounters + 2 : nullptr,
            semaphores,
            scratch,
            stream);
    }
    catch (...)
    {
        return -3;
    }
    if (timing_slot < kMaxTimingEvents)
    {
        status = cudaEventRecord(gTimingStop[timing_slot], stream);
        if (status != cudaSuccess)
        {
            return static_cast<int>(status);
        }
    }
    status = cudaPeekAtLastError();
    return status == cudaSuccess ? 0 : static_cast<int>(status);
}

extern "C" const char* blasst_xqa_temporal_build_id()
{
    return "TensorRT-LLM-main-XQA-sm90-bf16-d128-gqa4-page64-TEMPORAL-RUNTIME-K-REFRESH-PACKED32";
}
