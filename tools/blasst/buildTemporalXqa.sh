#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SRC="$REPO_ROOT/cpp/kernels/xqa"
TEMPORAL_SRC="$SRC/blasstTemporal"
BUILD="${BUILD_DIR:-$REPO_ROOT/build/blasstTemporal}"
KEEP="$BUILD/keepPipeline14Chunk16"
OUT="${OUTPUT_LIBRARY:-$BUILD/libblasst_xqa_temporal_page64.so}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_ROOT/bin/nvcc}"
TEST_DEFINES=()
if [[ "${BLASST_TEMPORAL_TESTING:-0}" == "1" ]]; then
  TEST_DEFINES+=(-DBLASST_TEMPORAL_TESTING=1)
fi

mkdir -p "$KEEP" "$(dirname -- "$OUT")"
"$NVCC" \
  -O3 -DNDEBUG -std=c++17 --use_fast_math --expt-relaxed-constexpr -lineinfo -include cassert \
  -allow-unsupported-compiler -t 0 -res-usage -keep --keep-dir "$KEEP" \
  -Xcompiler=-fPIC -shared -gencode=arch=compute_90a,code=sm_90a \
  -DHEAD_ELEMS=128 -DHEAD_GRP_SIZE=4 -DINPUT_FP16=0 -DCACHE_ELEM_ENUM=0 \
  -DTOKENS_PER_PAGE=64 -DPAGED_KV_CACHE_LAYOUT=0 \
  -DSKIP_SOFTMAX_ATTN=1 -DSKIP_SOFTMAX_ATTN_BLOCK_STATS=1 -DPRE_QK_SKIP=0 -DTEMPORAL_QK_SKIP=1 \
  -DTEMPORAL_PIPELINE_MODE=14 -DTEMPORAL_CHUNK_ITERS=16 -DENABLE_PDL=2 \
  "${TEST_DEFINES[@]}" \
  "$TEMPORAL_SRC/mhaSm90Temporal.cu" "$SRC/tensorMap.cpp" "$TEMPORAL_SRC/temporalXqaApi.cpp" \
  -I"$TEMPORAL_SRC" -I"$SRC" -L"$CUDA_ROOT/lib64" -lcudart -lcuda \
  -o "$OUT"

test -s "$OUT"
sha256sum "$OUT"
