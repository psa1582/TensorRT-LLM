#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H100 regression test for the protected temporal-QK layers."""

from __future__ import annotations

import argparse
import ctypes
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Result:
    layer: int
    candidates: int
    qk_elided_tiles: int
    refreshes: int
    stock_skipped_tiles: int
    stock_total_tiles: int
    state_mismatches: int | None
    output: torch.Tensor


def _check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} returned {status}")


def _configure_api(library: ctypes.CDLL) -> None:
    void_p = ctypes.c_void_p
    uint32_p = ctypes.POINTER(ctypes.c_uint32)
    library.blasst_temporal_prepare.argtypes = [ctypes.c_uint32] * 4
    library.blasst_temporal_reset.argtypes = [void_p]
    library.blasst_temporal_set_policy.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    library.blasst_temporal_get_policy.argtypes = [uint32_p, uint32_p]
    library.blasst_temporal_get_extended_stats.argtypes = [uint32_p] * 5
    library.blasst_temporal_test_set_null_state.argtypes = [ctypes.c_int]
    library.blasst_temporal_test_fill_layer_state.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        void_p,
    ]
    library.blasst_temporal_test_count_layer_state_mismatches.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        uint32_p,
    ]
    library.blasst_xqa_bf16_temporal.argtypes = [
        void_p,
        void_p,
        void_p,
        void_p,
        ctypes.POINTER(ctypes.c_int32),
        uint32_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        uint32_p,
        uint32_p,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        uint32_p,
        void_p,
        void_p,
    ]


def _ptr(tensor: torch.Tensor, pointer_type=None):
    if pointer_type is None:
        return ctypes.c_void_p(tensor.data_ptr())
    return ctypes.cast(tensor.data_ptr(), pointer_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=7716.1939)
    parser.add_argument("--epsilon", type=float, default=-1.0e6)
    parser.add_argument("--temporal-k", type=int, default=2)
    parser.add_argument("--refresh-interval", type=int, default=4)
    args = parser.parse_args()

    if args.sequence_length <= 0 or args.sequence_length % 64 != 0:
        raise ValueError("--sequence-length must be a positive multiple of 64")
    if not 1 <= args.temporal_k <= 7 or not 1 <= args.refresh_interval <= 255:
        raise ValueError("temporal k must be 1..7 and refresh interval must be 1..255")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this regression requires an SM90/Hopper GPU")

    library = ctypes.CDLL(str(args.library.resolve()))
    _configure_api(library)

    nb_layers = 3
    nb_kv_heads = 8
    head_group_size = 4
    head_elements = 128
    tokens_per_page = 64
    nb_pages = args.sequence_length // tokens_per_page
    stale_packed = 2
    temporal_k = args.temporal_k
    refresh_interval = args.refresh_interval

    query = torch.ones(
        (nb_kv_heads * head_group_size, head_elements), device="cuda", dtype=torch.bfloat16
    )
    pool = torch.zeros(
        (2, nb_pages, nb_kv_heads, tokens_per_page, head_elements),
        device="cuda",
        dtype=torch.bfloat16,
    )
    pool[0, : min(32, nb_pages)].fill_(1.0)
    for page in range(nb_pages):
        pool[1, page].fill_((page + 1) / nb_pages)
    page_list = torch.arange(2 * nb_pages, device="cuda", dtype=torch.int32).reshape(2, nb_pages)
    sequence_length = torch.tensor([args.sequence_length], device="cuda", dtype=torch.int32)
    kv_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    semaphores = torch.zeros(4096, device="cuda", dtype=torch.int32)
    scratch = torch.zeros(4 << 20, device="cuda", dtype=torch.uint8)
    output = torch.empty_like(query)

    _check(library.blasst_temporal_prepare(nb_layers, 1, nb_kv_heads, nb_pages), "prepare")
    _check(library.blasst_temporal_set_policy(temporal_k, refresh_interval), "set policy")
    policy_k = ctypes.c_uint32()
    policy_refresh = ctypes.c_uint32()
    _check(
        library.blasst_temporal_get_policy(ctypes.byref(policy_k), ctypes.byref(policy_refresh)),
        "get policy",
    )
    assert (policy_k.value, policy_refresh.value) == (temporal_k, refresh_interval)

    def run(layer: int, *, null_state: bool, seed_stale_state: bool) -> Result:
        _check(library.blasst_temporal_reset(None), "reset")
        _check(library.blasst_temporal_test_set_null_state(int(null_state)), "set null state")
        if seed_stale_state:
            _check(
                library.blasst_temporal_test_fill_layer_state(layer, stale_packed, None),
                "seed stale state",
            )
        output.zero_()
        semaphores.zero_()
        scratch.zero_()
        torch.cuda.synchronize()
        for _ in range(args.steps):
            _check(
                library.blasst_xqa_bf16_temporal(
                    _ptr(query),
                    _ptr(pool),
                    _ptr(pool),
                    _ptr(output),
                    _ptr(page_list, ctypes.POINTER(ctypes.c_int32)),
                    _ptr(sequence_length, ctypes.POINTER(ctypes.c_uint32)),
                    args.sequence_length,
                    nb_kv_heads,
                    args.threshold,
                    _ptr(kv_scale, ctypes.POINTER(ctypes.c_float)),
                    None,
                    None,
                    layer,
                    args.epsilon,
                    refresh_interval,
                    _ptr(semaphores, ctypes.POINTER(ctypes.c_uint32)),
                    _ptr(scratch),
                    None,
                ),
                f"layer {layer} launch",
            )
        torch.cuda.synchronize()

        counters = [ctypes.c_uint32() for _ in range(5)]
        _check(
            library.blasst_temporal_get_extended_stats(
                *(ctypes.byref(value) for value in counters)
            ),
            "get stats",
        )
        state_mismatches = None
        if seed_stale_state:
            mismatch_count = ctypes.c_uint32()
            _check(
                library.blasst_temporal_test_count_layer_state_mismatches(
                    layer, stale_packed, ctypes.byref(mismatch_count)
                ),
                "check stale state",
            )
            state_mismatches = mismatch_count.value
        return Result(
            layer=layer,
            candidates=counters[0].value,
            qk_elided_tiles=counters[1].value,
            refreshes=counters[2].value,
            stock_skipped_tiles=counters[3].value,
            stock_total_tiles=counters[4].value,
            state_mismatches=state_mismatches,
            output=output.detach().clone(),
        )

    rows = []
    for layer in (0, 1):
        full_qk = run(layer, null_state=True, seed_stale_state=False)
        protected = run(layer, null_state=False, seed_stale_state=True)
        reproduced = run(layer, null_state=False, seed_stale_state=True)
        max_abs_error = (protected.output.float() - full_qk.output.float()).abs().max().item()
        assert protected.candidates == 0
        assert protected.qk_elided_tiles == 0
        assert protected.refreshes == 0
        assert protected.stock_skipped_tiles > 0
        assert protected.state_mismatches == 0
        assert reproduced.qk_elided_tiles == protected.qk_elided_tiles
        assert reproduced.stock_skipped_tiles == protected.stock_skipped_tiles
        torch.testing.assert_close(protected.output, full_qk.output, rtol=0.0, atol=0.0)
        torch.testing.assert_close(reproduced.output, protected.output, rtol=0.0, atol=0.0)
        rows.append((protected, max_abs_error))

    layer2_full_qk = run(2, null_state=True, seed_stale_state=False)
    layer2_temporal = run(2, null_state=False, seed_stale_state=False)
    layer2_reproduced = run(2, null_state=False, seed_stale_state=False)
    layer2_max_abs_error = (
        (layer2_temporal.output.float() - layer2_full_qk.output.float()).abs().max().item()
    )
    assert layer2_temporal.qk_elided_tiles > 0
    assert layer2_temporal.refreshes > 0
    assert layer2_temporal.stock_skipped_tiles > 0
    assert layer2_reproduced.qk_elided_tiles > 0
    assert layer2_reproduced.refreshes > 0
    torch.testing.assert_close(layer2_temporal.output, layer2_full_qk.output, rtol=0.0, atol=0.02)
    torch.testing.assert_close(layer2_reproduced.output, layer2_temporal.output, rtol=0.0, atol=0.0)
    rows.append((layer2_temporal, layer2_max_abs_error))

    for result, max_abs_error in rows:
        print(
            json.dumps(
                {
                    "layer": result.layer,
                    "candidate_tiles": result.candidates,
                    "qk_elided_tiles": result.qk_elided_tiles,
                    "refresh_tiles": result.refreshes,
                    "stock_skipped_tiles": result.stock_skipped_tiles,
                    "stock_total_tiles": result.stock_total_tiles,
                    "state_mismatches": result.state_mismatches,
                    "max_abs_error_vs_full_qk": max_abs_error,
                },
                sort_keys=True,
            )
        )
    print("temporal protected-layer H100 regression: PASS")


if __name__ == "__main__":
    main()
