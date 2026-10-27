#!/usr/bin/env python3
"""Row-quantized shadows for the GPT-2 head: int8, int4, fp16.

The low-rank PCA shadow leaves a large Cauchy-Schwarz residual on this model.
This script bounds each logit by a quantized row plus ||e|| ||h|| (optionally
grouped) and estimates certified Gumbel-max bytes against the fp32 head.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from measure_shadow import (
    count_opens,
    count_topk_rows,
    gumbel,
    shadow_bytes,
    slack_stats,
    softmax_entropy,
    summarize,
)

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "analysis" / "cache"
OUT = REPO / "analysis" / "results"


def load() -> tuple[np.ndarray, np.ndarray]:
    w = np.load(CACHE / "gpt2_wte_fp32.npy").astype(np.float32)
    blob = np.load(CACHE / "gpt2_wikitext_h_train4000_eval256_seq128_st128.npz")
    h = blob["h_eval"].astype(np.float32)
    return w, h


def quantize_rows(w: np.ndarray, levels: int, group: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric quantization. levels is the max abs code (127 or 7).

    group is the contiguous width along the hidden axis. Returns recon float32
    and a list is not returned; recon is exact dequant.
    """
    vocab, hidden = w.shape
    if hidden % group != 0:
        raise ValueError(group)
    x = w.reshape(vocab, hidden // group, group)
    max_abs = np.max(np.abs(x), axis=2)
    scale = np.maximum(max_abs / float(levels), 1e-12).astype(np.float32)
    q = np.clip(np.round(x / scale[:, :, None]), -levels, levels)
    recon = (q * scale[:, :, None]).reshape(vocab, hidden).astype(np.float32)
    return recon, scale


def residual_term(err: np.ndarray, h: np.ndarray, groups: int) -> np.ndarray:
    """sum_g ||e_g|| ||h_g||, shape [N, V]. groups=1 is ||e|| ||h||."""
    vocab, hidden = err.shape
    n = h.shape[0]
    if hidden % groups != 0:
        raise ValueError(groups)
    eg = np.linalg.norm(err.reshape(vocab, groups, hidden // groups), axis=2)  # [V, G]
    hg = np.linalg.norm(h.reshape(n, groups, hidden // groups), axis=2)  # [N, G]
    return hg @ eg.T


def bytes_for(vocab: int, hidden: int, code_bytes: float, n_scales: int, groups: int) -> dict:
    """code_bytes is bytes per weight (1 for int8, 0.5 for packed int4, 2 for fp16)."""
    proj = int(round(vocab * hidden * code_bytes))
    scales = vocab * n_scales * 4
    norms = vocab * groups * 4
    # No PCA basis. Center term unused.
    total = proj + scales + norms
    dense = vocab * hidden * 4
    return {
        "projection": proj,
        "scales": scales,
        "residual_norms": norms,
        "shadow_total": total,
        "dense_fp32": dense,
        "row_fp32": hidden * 4,
        "shadow_fraction": total / dense,
    }


def eval_upper(
    logits: np.ndarray,
    upper: np.ndarray,
    info: dict,
    temperatures: list[float],
    blocks: list[int],
    replicates: int,
    seed: int,
) -> dict:
    if np.min(upper - logits) < -1e-3:
        raise AssertionError(f"bound violated: {np.min(upper - logits)}")
    n, vocab = logits.shape
    rng = np.random.default_rng(seed)
    shadow = info["shadow_total"]
    dense = info["dense_fp32"]
    row_bytes = info["row_fp32"]
    out: dict = {"slack": slack_stats(upper - logits), "temperature": {}, "greedy_topk": {}}
    for k in (1, 50):
        out["greedy_topk"][str(k)] = {}
        for b in blocks:
            rows = count_topk_rows(upper, logits, k, b)
            out["greedy_topk"][str(k)][str(b)] = summarize(rows, shadow, dense, row_bytes)
    for t in temperatures:
        ent = softmax_entropy(logits, t)
        out["temperature"][str(t)] = {"entropy_mean": float(np.mean(ent))}
        for mode in ("aligned", "split"):
            acc = {b: [] for b in blocks}
            ratios_b1 = []
            ent_b1 = []
            for _ in range(replicates):
                g = gumbel(rng, (n, vocab))
                score = logits * (1.0 / t) + g
                for b in blocks:
                    rows = count_opens(upper, g, score, t, b, mode)
                    acc[b].append(rows)
                    if b == 1:
                        ratios_b1.append(shadow + rows.astype(np.float64) * row_bytes)
                        ent_b1.append(ent)
            by_block = {}
            for b in blocks:
                rows = np.concatenate(acc[b])
                by_block[str(b)] = summarize(rows, shadow, dense, row_bytes)
            if ratios_b1:
                ratio = np.concatenate(ratios_b1) / dense
                ent_rep = np.concatenate(ent_b1)
                qs = np.quantile(ent_rep, [0.25, 0.5, 0.75])
                bins = np.digitize(ent_rep, qs)
                quart = []
                for bi in range(4):
                    sel = ratio[bins == bi]
                    quart.append(
                        {
                            "entropy_mean": float(np.mean(ent_rep[bins == bi])) if sel.size else None,
                            "byte_ratio_mean": float(np.mean(sel)) if sel.size else None,
                            "n": int(sel.size),
                        }
                    )
                by_block["1"]["entropy_quartiles"] = quart
            out["temperature"][str(t)][mode] = by_block
    return out


def main() -> None:
    t0 = time.time()
    w, h = load()
    vocab, hidden = w.shape
    print(f"W {w.shape} h {h.shape}", flush=True)
    logits = (h.astype(np.float64) @ w.astype(np.float64).T).astype(np.float64)
    temperatures = [0.5, 0.8, 1.0, 1.2]
    blocks = [1, 32, 128, 256]
    configs = []

    specs = [
        ("int8", 127, 1.0, hidden, 1),  # one scale per row, scalar CS
        ("int8", 127, 1.0, hidden, 8),
        ("int8", 127, 1.0, hidden, 16),
        ("int8", 127, 1.0, hidden, 32),
        ("int8g64", 127, 1.0, 64, 1),  # quant group 64, scalar CS on the residual
        ("int8g64", 127, 1.0, 64, 12),
        ("int4", 7, 0.5, hidden, 1),
        ("int4", 7, 0.5, hidden, 16),
        ("fp16", None, 2.0, hidden, 1),
        ("fp16", None, 2.0, hidden, 16),
    ]

    for name, levels, code_bytes, qgroup, groups in specs:
        ts = time.time()
        if name.startswith("fp16"):
            recon = w.astype(np.float16).astype(np.float32)
            n_scales = 0
            # fp16 has no scale vector; the code itself is the shadow.
        else:
            recon, scale = quantize_rows(w, levels, qgroup)
            n_scales = hidden // qgroup
        err = w.astype(np.float32) - recon
        # Bound in float64.
        hat = (h.astype(np.float64) @ recon.astype(np.float64).T)
        extra = residual_term(err.astype(np.float64), h.astype(np.float64), groups)
        upper = hat + extra
        info = bytes_for(vocab, hidden, code_bytes, n_scales, groups)
        label = f"{name}/q{qgroup}/g{groups}"
        print(
            f"{label} slack p50 soon, shadow {info['shadow_fraction']:.3f}",
            flush=True,
        )
        stats = eval_upper(logits, upper, info, temperatures, blocks, 2, seed=1)
        ratios = [
            stats["temperature"][str(t)]["aligned"]["1"]["byte_ratio_mean"] for t in temperatures
        ]
        rec = {
            "label": label,
            "bytes": info,
            "seconds": time.time() - ts,
            "mean_byte_ratio_block1_aligned": float(np.mean(ratios)),
            "stats": stats,
        }
        configs.append(rec)
        gap = stats["slack"]
        print(
            f"  slack p50={gap['p50']:.4f} p99={gap['p99']:.4f} min={gap['min']:.4e}"
            f"  block1 aligned mean {rec['mean_byte_ratio_block1_aligned']:.4f}"
            f"  ({rec['seconds']:.1f}s)",
            flush=True,
        )
        # Free large temporaries before the next quantizer.
        del recon, err, hat, extra, upper

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gpt2_quant_row.json"
    path.write_text(
        json.dumps(
            {
                "temperatures": temperatures,
                "blocks": blocks,
                "replicates": 2,
                "n_eval": int(h.shape[0]),
                "configs": configs,
            }
        )
    )
    print(f"wrote {path} in {time.time() - t0:.1f}s", flush=True)
    print(f"{'config':<22} {'sh%':>6} " + " ".join(f"T={t:<4}" for t in temperatures) + "  top1  top50")
    for rec in configs:
        sh = 100 * rec["bytes"]["shadow_fraction"]
        cells = []
        for t in temperatures:
            cell = rec["stats"]["temperature"][str(t)]["aligned"]["1"]["byte_ratio_mean"]
            cells.append(f"{cell:.3f}")
        g1 = rec["stats"]["greedy_topk"]["1"]["1"]["byte_ratio_mean"]
        g50 = rec["stats"]["greedy_topk"]["50"]["1"]["byte_ratio_mean"]
        print(f"{rec['label']:<22} {sh:6.2f} " + " ".join(f"{c:>6}" for c in cells) + f"  {g1:.3f}  {g50:.3f}")


if __name__ == "__main__":
    main()
