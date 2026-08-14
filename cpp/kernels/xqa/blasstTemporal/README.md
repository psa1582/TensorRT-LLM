# BLASST temporal pre-QK XQA

This directory contains the Hopper (`sm_90a`) BF16 decode kernel used to skip QK work when the same Stock BLASST KV
tile remains skippable across autoregressive decode steps. The implementation is intentionally isolated from the
upstream `mha_sm90.cu`; dense attention and Stock BLASST continue to use the normal TensorRT-LLM path.

## Supported configuration

- Hopper H100 (`sm_90a`)
- BF16 query and KV cache
- head dimension 128, GQA ratio 4
- paged KV cache with 64 tokens per block
- batch 1, beam 1, one-token decode
- pipeline mode 14, temporal chunk size 16

The packed predictor state uses one `uint32_t` per `[layer, batch, KV head, KV tile]`:

- bits `[2:0]`: consecutive qualifying count (`0..7`)
- bits `[10:3]`: refresh age (`0..255`)

Changing `k` in `1..7` or refresh in `1..255` does not change the global-memory allocation.

## Build

From the repository root:

```bash
bash tools/blasst/buildTemporalXqa.sh
```

The default output is:

```text
build/blasstTemporal/libblasst_xqa_temporal_page64.so
```

Override it with `OUTPUT_LIBRARY=/absolute/path/libblasst_xqa_temporal_page64.so`. The build requires CUDA with
`sm_90a` support and accepts `CUDA_HOME` or `NVCC` overrides.

Because `decoderXQARunner.cpp` contains the opt-in dispatch, rebuild TensorRT-LLM after checking out this branch.

## Runtime policy

Set these before TensorRT-LLM initializes XQA:

```bash
export TLLM_TEMPORAL_QK_SKIP=1
export TLLM_TEMPORAL_XQA_LIBRARY="$PWD/build/blasstTemporal/libblasst_xqa_temporal_page64.so"
export TLLM_TEMPORAL_K=2
export TLLM_TEMPORAL_REFRESH=4
export TLLM_TEMPORAL_EPSILONS="inf,1.2,3.0,..."
```

The runner loads the library once and applies `TLLM_TEMPORAL_K` and `TLLM_TEMPORAL_REFRESH` through
`blasst_temporal_set_policy`. Per-layer thresholds come from `TLLM_TEMPORAL_EPSILONS`; `TLLM_TEMPORAL_EPSILON` is the
fallback for layers not present in the list.

Allocate state once before generation and reset it between independent sequences:

```python
import ctypes

library = ctypes.CDLL("build/blasstTemporal/libblasst_xqa_temporal_page64.so")
library.blasst_temporal_prepare.argtypes = [ctypes.c_uint32] * 4
library.blasst_temporal_reset.argtypes = [ctypes.c_void_p]

# max_layers, max_batch, max_kv_heads, max_64-token_tiles
assert library.blasst_temporal_prepare(32, 1, 8, 2048) == 0
assert library.blasst_temporal_reset(None) == 0
```

If the policy changes after state is populated, call `blasst_temporal_reset` on the inference stream before the next
decode step.

## Three-way comparison

- Dense: no sparse-attention config; leave `TLLM_TEMPORAL_QK_SKIP` unset.
- Stock BLASST: enable the decode skip-softmax threshold; leave `TLLM_TEMPORAL_QK_SKIP` unset.
- Temporal QK skip: use the same Stock BLASST threshold and set the temporal variables above.

Use `tokens_per_block=64` for all three modes. The temporal route rejects unsupported shapes instead of silently
falling back, so an invalid benchmark configuration cannot be mistaken for a temporal result.

## Validation

The packed state-machine contract is CPU-only and can be run anywhere:

```bash
python tools/blasst/test_runtime_policy_state.py
```

The CUDA library and TensorRT-LLM route must be compiled and exercised on an H100 before performance claims are made.
