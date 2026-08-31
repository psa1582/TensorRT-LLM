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

## MLA experiments on H100

AutoDeploy's `trtllm_mla` backend exposes the Stock BLASST skip-softmax threshold through two opt-in environment
variables. Set them before starting Python because the values are read when the MLA backend module is imported:

```bash
export TRTLLM_BLASST_PREFILL_THRESHOLD=3
export TRTLLM_BLASST_DECODE_THRESHOLD=0
```

An unset, empty, zero, or negative value disables that threshold and is the dense baseline. Positive values are
forwarded to every MLA `thop.attention` call as `skip_softmax_threshold_scale_factor_prefill` or
`skip_softmax_threshold_scale_factor_decode`. Calibrate the prefill threshold against accuracy data rather than
assuming that a value transfers between models or context lengths.

Two model-registry configurations are provided for `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` on one H100:

```bash
# RoPE remains in the model graph; the MLA kernel receives an identity table.
python examples/auto_deploy/build_and_run_ad.py \
  --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --use-registry --registry-config-id mla_blasst_nofuse_h100

# RoPE is moved into the TRT-LLM MLA kernel.
python examples/auto_deploy/build_and_run_ad.py \
  --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --use-registry --registry-config-id mla_blasst_fused_h100
```

The no-fuse path sizes its identity RoPE table from `max_seq_len`; a 64K run no longer reuses the fallback 8K table.
The fused path restores both `q_b_proj` and the direct `q_proj` used by models with `q_lora_rank=None` to GPTJ
pairwise layout before `mla_rope_generation` applies RoPE.

The temporal pre-QK decode library in this directory is a separate optimization with the shape restrictions listed
above. DeepSeek-Coder-V2-Lite's MLA dimensions (`QK=192`, `V=128`) do not satisfy that temporal XQA contract. Do not
set `TLLM_TEMPORAL_QK_SKIP=1` for this MLA experiment; use the Stock BLASST threshold variables instead.

## Validation

The packed state-machine contract is CPU-only and can be run anywhere:

```bash
python tools/blasst/test_runtime_policy_state.py
```

The CUDA library and TensorRT-LLM route must be compiled and exercised on an H100 before performance claims are made.
