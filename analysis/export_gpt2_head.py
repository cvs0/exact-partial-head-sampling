#!/usr/bin/env python3
"""Write the GPT-2 fp32 head and WikiText-2 validation hidden states.

The dump is the input to ephs_bench. Weights are not committed.
Layout (little endian):
  magic[8] = EPHS0001
  u32 vocab, u32 hidden, u32 n_states
  f32 W[vocab, hidden] row major
  f32 H[n_states, hidden]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "analysis" / "cache"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else CACHE / "gpt2_head.bin"
    w = np.load(CACHE / "gpt2_wte_fp32.npy").astype("<f4", copy=False)
    h = np.load(CACHE / "gpt2_wikitext_h_train4000_eval256_seq128_st128.npz")["h_eval"].astype("<f4", copy=False)
    if w.ndim != 2 or h.ndim != 2 or w.shape[1] != h.shape[1]:
        raise SystemExit(f"shape mismatch W {w.shape} H {h.shape}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(b"EPHS0001")
        f.write(struct.pack("<III", w.shape[0], w.shape[1], h.shape[0]))
        f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(h, dtype="<f4").tobytes())
    print(f"wrote {out} vocab={w.shape[0]} hidden={w.shape[1]} states={h.shape[0]} bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
