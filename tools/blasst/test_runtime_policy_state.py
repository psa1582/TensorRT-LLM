#!/usr/bin/env python3
"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0

CPU contract test for the packed runtime k/refresh temporal state.
"""

from __future__ import annotations

import random


RUN_MASK = 0x7
REFRESH_SHIFT = 3
REFRESH_MASK = 0xFF
PACKED_MASK = RUN_MASK | (REFRESH_MASK << REFRESH_SHIFT)


def unpack(packed: int) -> tuple[int, int]:
    return packed & RUN_MASK, (packed >> REFRESH_SHIFT) & REFRESH_MASK


def step_packed(packed: int, qualifies: bool, k: int, refresh: int) -> tuple[int, bool]:
    run_count, refresh_age = unpack(packed)
    candidate = run_count >= k
    pre_skip = candidate and refresh_age < refresh
    if pre_skip:
        return run_count | (min(255, refresh_age + 1) << REFRESH_SHIFT), True
    next_run = min(k, run_count + 1) if qualifies else 0
    return next_run, False


def step_reference(state: tuple[int, int], qualifies: bool, k: int, refresh: int):
    run_count, refresh_age = state
    candidate = run_count >= k
    pre_skip = candidate and refresh_age < refresh
    if pre_skip:
        return (run_count, min(255, refresh_age + 1)), True
    return (min(k, run_count + 1) if qualifies else 0, 0), False


def step_old_k2(packed: int, qualifies: bool, refresh: int) -> tuple[int, bool]:
    run_count = packed & 0x3
    refresh_age = (packed >> 2) & 0xFF
    pre_skip = run_count >= 2 and refresh_age < refresh
    if pre_skip:
        return run_count | (min(255, refresh_age + 1) << 2), True
    next_run = min(2, run_count + 1) if qualifies else 0
    return next_run, False


def main() -> None:
    rng = random.Random(20260814)
    for k in (1, 2, 4, 7):
        for refresh in (1, 2, 4, 8, 255):
            packed = 0
            reference = (0, 0)
            for _ in range(10_000):
                qualifies = rng.random() < 0.73
                packed, packed_skip = step_packed(packed, qualifies, k, refresh)
                reference, reference_skip = step_reference(reference, qualifies, k, refresh)
                assert unpack(packed) == reference
                assert packed_skip == reference_skip
                assert packed & ~PACKED_MASK == 0
    for refresh in (1, 2, 4, 8, 255):
        old_packed = 0
        new_packed = 0
        rng = random.Random(20260814 + refresh)
        for _ in range(10_000):
            qualifies = rng.random() < 0.73
            old_packed, old_skip = step_old_k2(old_packed, qualifies, refresh)
            new_packed, new_skip = step_packed(new_packed, qualifies, 2, refresh)
            assert old_skip == new_skip
            assert (old_packed & 0x3) == (new_packed & RUN_MASK)
            assert ((old_packed >> 2) & 0xFF) == ((new_packed >> REFRESH_SHIFT) & REFRESH_MASK)
    print("runtime temporal state contract: PASS")


if __name__ == "__main__":
    main()
