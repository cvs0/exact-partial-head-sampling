#!/usr/bin/env python3
"""Measure shadow-bound tightness and certified Gumbel-max block opens.

Go/no-go analysis only. For held-out LM-head inputs, builds a provable upper
bound on every logit and estimates how many vocabulary blocks the certified
partial-read sampler would open. Prints byte ratios against a dense head read.

Shadow (affine PCA on hidden states, the certificate used by the knlp greedy
decoder, adapted to a stored dtype):

    h = mu + B q + r_h,    q = B^T (h - mu),   B orthonormal
    W_i h = W_i mu + (W_i B) · q + W_i · r_h
    |W_i · r_h| <= ||(I - B B^T) W_i||_2 * ||r_h||_2

An int8 (or fp32) shadow replaces (W_i B) and adds a Cauchy-Schwarz term on
the quantization residual. An optional grouped residual splits the same inner
product across coordinate groups.

Block rule (coupled to one Gumbel draw):

    M_b = max_{i in b} G_i
    ceiling_b = max_{i in b} U_i / T + M_b
    S* = max_i (z_i / T + G_i)
    open b if ceiling_b > S*, and always open the block that holds the argmax.

That set is exactly the set read by opening blocks in decreasing ceiling order
and stopping when the incumbent score meets every remaining ceiling (ties have
probability zero under continuous Gumbel noise).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "analysis" / "results"


def gumbel(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    # Gumbel(0, 1) via the inverse CDF. Clip only to keep log finite; the
    # measurement uses the same sample for S* and the block maxima.
    u = rng.random(shape, dtype=np.float64)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


def softmax_entropy(logits: np.ndarray, temperature: float) -> np.ndarray:
    """nats, shape [N]. logits is [N, V]."""
    scaled = logits * (1.0 / temperature)
    m = np.max(scaled, axis=1, keepdims=True)
    e = np.exp(scaled - m)
    z = np.sum(e, axis=1, keepdims=True)
    p = e / z
    # 0 log 0 -> 0
    logp = np.log(p, where=p > 0, out=np.zeros_like(p))
    return -np.sum(p * logp, axis=1)


def project_affine(
    h: np.ndarray, mu: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return q [N, r] and rho [N] for h ≈ mu + q @ basis.T.

    basis is [D, r], orthonormal columns. rho is ||h - mu - q basis.T||,
    computed directly (not via ||h||^2 - ||q||^2).
    """
    hc = h - mu
    q = hc @ basis
    recon = q @ basis.T
    rho = np.linalg.norm(hc - recon, axis=1)
    return q, rho


def quantize_int8_rows(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row symmetric int8. Returns (q int8, scale fp32, e_l2 fp32).

    Reconstructed row is q * scale. Round-to-nearest error on each entry is
    at most half a bin; we store the true residual L2 norm so the runtime
    bound is ||e||_2 ||q||_2 rather than the looser bin bound.
    """
    max_abs = np.max(np.abs(p), axis=1)
    scale = (max_abs / 127.0).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-12))
    q = np.clip(np.round(p / scale[:, None]), -127, 127).astype(np.int32)
    recon = q.astype(np.float64) * scale.astype(np.float64)[:, None]
    err = p.astype(np.float64) - recon
    e_l2 = np.linalg.norm(err, axis=1).astype(np.float32)
    return q.astype(np.int8), scale, e_l2


def shadow_bytes(
    vocab: int,
    hidden: int,
    rank: int,
    kind: str,
    groups: int,
) -> dict[str, int]:
    """Bytes of certificate state touched once per token, plus the dense row size."""
    basis = hidden * rank * 4 + hidden * 4  # B and mu, fp32
    center = vocab * 4  # c[v] = W[v] · mu, fp32
    delta = vocab * groups * 4  # fp32 group residual norms (groups=1 is scalar delta)
    if kind == "fp32":
        proj = vocab * rank * 4
        extra = 0
    elif kind == "int8":
        proj = vocab * rank  # int8 codes
        extra = vocab * 4 + vocab * 4  # per-row scale and ||e||_2
    else:
        raise ValueError(kind)
    total = basis + center + delta + proj + extra
    dense = vocab * hidden * 4
    return {
        "basis_mu": basis,
        "token_center": center,
        "residual_norms": delta,
        "projection": proj,
        "quant_aux": extra,
        "shadow_total": total,
        "dense_fp32": dense,
        "row_fp32": hidden * 4,
    }


def block_slices(vocab: int, block: int) -> list[tuple[int, int]]:
    return [(s, min(s + block, vocab)) for s in range(0, vocab, block)]


def _pad_blocks(a: np.ndarray, block: int, fill: float) -> tuple[np.ndarray, int, int]:
    """Pad the last axis up to a multiple of `block`. Returns (padded, n_blocks, pad)."""
    vocab = a.shape[1]
    n_blocks = (vocab + block - 1) // block
    pad = n_blocks * block - vocab
    if pad:
        a = np.pad(a, ((0, 0), (0, pad)), constant_values=fill)
    return a, n_blocks, pad


def count_opens(
    upper: np.ndarray,
    g: np.ndarray,
    score: np.ndarray,
    temperature: float,
    block: int,
    mode: str,
) -> np.ndarray:
    """Return dense rows read per sample, shape [N].

    mode "aligned": block ceiling is max_i (U_i/T + G_i). Valid once per-token
    bounds and per-token Gumbels exist. Same shadow traffic as taking a block
    max of U first, and a tighter certificate.

    mode "split": block ceiling is max_i(U_i)/T + max_i(G_i). This is the
    block-max-noise bound: it does not use the alignment of noise with logits.
    """
    n, vocab = score.shape
    winner_score = np.max(score, axis=1)
    winner = np.argmax(score, axis=1)
    if mode == "aligned":
        token_ceil = upper * (1.0 / temperature) + g
        padded, n_blocks, pad = _pad_blocks(token_ceil, block, -np.inf)
        block_ceil = np.max(padded.reshape(n, n_blocks, block), axis=2)
    elif mode == "split":
        u_pad, n_blocks, pad = _pad_blocks(upper, block, -np.inf)
        g_pad, _, _ = _pad_blocks(g, block, -np.inf)
        max_u = np.max(u_pad.reshape(n, n_blocks, block), axis=2)
        max_g = np.max(g_pad.reshape(n, n_blocks, block), axis=2)
        block_ceil = max_u * (1.0 / temperature) + max_g
    else:
        raise ValueError(mode)
    take = block_ceil > winner_score[:, None]
    take[np.arange(n), winner // block] = True
    widths = np.full(n_blocks, block, dtype=np.int32)
    if pad:
        widths[-1] = block - pad
    return (take.astype(np.int32) * widths[None, :]).sum(axis=1)


def count_topk_rows(upper: np.ndarray, logits: np.ndarray, k: int, block: int) -> np.ndarray:
    """Rows read to certify the exact top-k logit set at this block size."""
    n, vocab = logits.shape
    if k > vocab:
        raise ValueError(k)
    kth = np.partition(logits, vocab - k, axis=1)[:, vocab - k]
    top_idx = np.argpartition(logits, vocab - k, axis=1)[:, vocab - k :]
    u_pad, n_blocks, pad = _pad_blocks(upper, block, -np.inf)
    block_u = np.max(u_pad.reshape(n, n_blocks, block), axis=2)
    holds = np.zeros((n, n_blocks), dtype=bool)
    bidx = top_idx // block
    for j in range(k):
        holds[np.arange(n), bidx[:, j]] = True
    take = (block_u > kth[:, None]) | holds
    widths = np.full(n_blocks, block, dtype=np.int32)
    if pad:
        widths[-1] = block - pad
    return (take.astype(np.int32) * widths[None, :]).sum(axis=1)


def summarize(rows_read: np.ndarray, shadow: int, dense: int, row_bytes: int) -> dict:
    head_bytes = shadow + rows_read.astype(np.float64) * row_bytes
    ratio = head_bytes / dense
    frac_rows = rows_read.astype(np.float64) / (dense / row_bytes)
    return {
        "rows_mean": float(np.mean(rows_read)),
        "rows_median": float(np.median(rows_read)),
        "rows_p90": float(np.quantile(rows_read, 0.90)),
        "row_fraction_mean": float(np.mean(frac_rows)),
        "bytes_mean": float(np.mean(head_bytes)),
        "bytes_median": float(np.median(head_bytes)),
        "byte_ratio_mean": float(np.mean(ratio)),
        "byte_ratio_median": float(np.median(ratio)),
        "byte_ratio_p90": float(np.quantile(ratio, 0.90)),
    }


def fit_affine_pca(h: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-`rank` PCA of h. Returns mu [D], basis [D, rank], explained fraction."""
    mu = np.mean(h, axis=0)
    hc = h - mu
    # Economy SVD of the centered data. Rows of Vt are principal axes.
    _, s, vt = np.linalg.svd(hc, full_matrices=False)
    r = min(rank, vt.shape[0])
    basis = vt[:r].T.astype(np.float64)
    # Re-orthonormalize against numeric drift.
    basis, _ = np.linalg.qr(basis)
    energy = s**2
    explained = float(np.sum(energy[:r]) / np.sum(energy))
    return mu.astype(np.float64), basis[:, :r], explained


def fit_svd_basis(w: np.ndarray, rank: int) -> np.ndarray:
    """Top right singular vectors of W [V, D], as a [D, rank] orthonormal basis.

    Computed from the Gram matrix so we never form the full SVD workspace.
    """
    gram = w.T @ w
    gram = 0.5 * (gram + gram.T)
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    basis = evecs[:, order[:rank]]
    basis, _ = np.linalg.qr(basis)
    return basis[:, :rank]


def load_gpt2_wikitext(
    n_train: int,
    n_eval: int,
    seq_len: int,
    stride: int,
    cache: Path,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache.mkdir(parents=True, exist_ok=True)
    h_path = cache / f"gpt2_wikitext_h_train{n_train}_eval{n_eval}_seq{seq_len}_st{stride}.npz"
    w_path = cache / "gpt2_wte_fp32.npy"
    if h_path.exists() and w_path.exists():
        blob = np.load(h_path)
        w = np.load(w_path)
        return w, blob["h_train"], blob["h_eval"], json.loads(str(blob["meta"]))

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    torch.set_num_threads(4)
    w = model.lm_head.weight.detach().float().numpy().astype(np.float32)
    if model.lm_head.bias is not None:
        raise RuntimeError("expected GPT-2 lm_head bias to be None")

    def texts_from(split: str) -> list[str]:
        # datasets.load_dataset("wikitext") is broken against current hub URIs
        # (repo id parsed as "wikitext" rather than "Salesforce/wikitext").
        # Read the raw parquet directly.
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        filename = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
        path = hf_hub_download("Salesforce/wikitext", filename, repo_type="dataset")
        table = pq.read_table(path, columns=["text"])
        docs: list[str] = []
        buf: list[str] = []
        for line in table.column("text").to_pylist():
            if line is None or line.strip() == "":
                if buf:
                    docs.append("\n".join(buf))
                    buf = []
            else:
                buf.append(line)
        if buf:
            docs.append("\n".join(buf))
        return [d for d in docs if len(d) > 200]

    def harvest(docs: list[str], limit: int, take_every: int) -> np.ndarray:
        out: list[np.ndarray] = []
        for doc in docs:
            ids = tok(doc, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if int(ids.numel()) < 16:
                continue
            for start in range(0, int(ids.numel()) - 8, stride):
                chunk = ids[start : start + seq_len]
                if int(chunk.numel()) < 16:
                    continue
                with torch.no_grad():
                    hidden = model.transformer(chunk.unsqueeze(0)).last_hidden_state[0]
                # last_hidden_state is post ln_f, which is the lm_head input.
                arr = hidden.float().numpy()
                for pos in range(0, arr.shape[0], take_every):
                    out.append(arr[pos])
                    if len(out) >= limit:
                        return np.stack(out).astype(np.float32)
        if not out:
            raise RuntimeError("no hidden states collected")
        return np.stack(out).astype(np.float32)

    train_docs = texts_from("train")
    eval_docs = texts_from("validation")
    h_train = harvest(train_docs, n_train, take_every=8)
    h_eval = harvest(eval_docs, n_eval, take_every=4)

    # Parity: lm_head logits vs explicit matmul, a handful of rows.
    with torch.no_grad():
        ht = torch.from_numpy(h_eval[:4])
        ref = torch.nn.functional.linear(ht, torch.from_numpy(w)).numpy()
    got = h_eval[:4].astype(np.float64) @ w.T.astype(np.float64)
    err = np.max(np.abs(ref - got))
    meta = {
        "model": "openai-community/gpt2",
        "dataset": "wikitext wikitext-2-raw-v1",
        "train_split": "train",
        "eval_split": "validation",
        "seq_len": seq_len,
        "stride": stride,
        "n_train": int(h_train.shape[0]),
        "n_eval": int(h_eval.shape[0]),
        "vocab": int(w.shape[0]),
        "hidden": int(w.shape[1]),
        "logit_parity_max_abs": float(err),
        "dtype_weights": "fp32",
        "head_bias": None,
    }
    np.save(w_path, w)
    np.savez_compressed(h_path, h_train=h_train, h_eval=h_eval, meta=json.dumps(meta))
    return w, h_train, h_eval, meta


def self_check() -> None:
    """Affine + int8 + grouped bounds dominate W h on a random instance."""
    rng = np.random.default_rng(0)
    vocab, hidden, rank, n, groups = 40, 16, 5, 7, 4
    w = rng.normal(size=(vocab, hidden))
    h = rng.normal(size=(n, hidden))
    mu, basis, _ = fit_affine_pca(np.vstack([h, rng.normal(size=(30, hidden))]), rank)
    q, rho = project_affine(h, mu, basis)
    p = w @ basis  # [V, r]
    c = w @ mu
    row_sq = np.sum(w * w, axis=1)
    delta = np.sqrt(np.maximum(row_sq - np.sum(p * p, axis=1), 0.0))
    logits = h @ w.T
    u_fp = c[None, :] + q @ p.T + rho[:, None] * delta[None, :]
    if np.min(u_fp - logits) < -1e-8:
        raise AssertionError(f"fp32-subspace bound violated: {np.min(u_fp - logits)}")
    q_i, scale, e_l2 = quantize_int8_rows(p.astype(np.float64))
    p_hat = q_i.astype(np.float64) * scale.astype(np.float64)[:, None]
    qn = np.linalg.norm(q, axis=1)
    u_i8 = (
        c[None, :]
        + q @ p_hat.T
        + rho[:, None] * delta[None, :]
        + qn[:, None] * e_l2[None, :]
    )
    if np.min(u_i8 - logits) < -1e-8:
        raise AssertionError(f"int8 bound violated: {np.min(u_i8 - logits)}")
    # Grouped residual must also dominate, and must be <= scalar CS + a hair.
    res_w = w - p @ basis.T
    assert hidden % groups == 0
    gw = np.linalg.norm(res_w.reshape(vocab, groups, hidden // groups), axis=2)
    hc = h - mu
    res_h = hc - q @ basis.T
    gh = np.linalg.norm(res_h.reshape(n, groups, hidden // groups), axis=2)
    grouped = gh @ gw.T
    u_g = c[None, :] + q @ p.T + grouped
    if np.min(u_g - logits) < -1e-8:
        raise AssertionError(f"grouped bound violated: {np.min(u_g - logits)}")
    print(
        f"self-check ok  fp_slack_med={np.median(u_fp - logits):.4f}"
        f"  int8_slack_med={np.median(u_i8 - logits):.4f}"
        f"  grouped_slack_med={np.median(u_g - logits):.4f}"
    )


def slack_stats(gap: np.ndarray) -> dict[str, float]:
    flat = gap.reshape(-1)
    return {
        "min": float(np.min(flat)),
        "p50": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
        "p99": float(np.quantile(flat, 0.99)),
        "max": float(np.max(flat)),
    }


def eval_config(
    logits: np.ndarray,
    upper: np.ndarray,
    temperatures: list[float],
    blocks: list[int],
    replicates: int,
    seed: int,
    shadow: int,
    dense: int,
    row_bytes: int,
    topk_list: list[int],
) -> dict:
    if np.min(upper - logits) < -1e-4:
        raise AssertionError(
            f"bound violated by more than 1e-4: min gap {np.min(upper - logits)}"
        )
    rng = np.random.default_rng(seed)
    n, vocab = logits.shape
    out: dict = {"temperature": {}, "greedy_topk": {}, "slack": slack_stats(upper - logits)}
    for k in topk_list:
        by_block = {}
        for block in blocks:
            rows = count_topk_rows(upper, logits, k, block)
            by_block[str(block)] = summarize(rows, shadow, dense, row_bytes)
        out["greedy_topk"][str(k)] = by_block

    entropies = {}
    for t in temperatures:
        ent = softmax_entropy(logits, t)
        entropies[str(t)] = {
            "mean": float(np.mean(ent)),
            "p50": float(np.median(ent)),
            "p10": float(np.quantile(ent, 0.10)),
            "p90": float(np.quantile(ent, 0.90)),
        }
        by_mode: dict = {}
        for mode in ("aligned", "split"):
            acc = {b: [] for b in blocks}
            byte_by_entropy = {b: [] for b in blocks}
            for _ in range(replicates):
                g = gumbel(rng, (n, vocab))
                score = logits * (1.0 / t) + g
                for b in blocks:
                    rows = count_opens(upper, g, score, t, b, mode)
                    acc[b].append(rows)
                    head_bytes = shadow + rows.astype(np.float64) * row_bytes
                    byte_by_entropy[b].append(head_bytes / dense)
            by_block = {}
            for b in blocks:
                rows = np.concatenate(acc[b], axis=0)
                by_block[str(b)] = summarize(rows, shadow, dense, row_bytes)
                ratios = np.concatenate(byte_by_entropy[b], axis=0)
                ent_rep = np.tile(ent, replicates)
                qs = np.quantile(ent_rep, [0.25, 0.5, 0.75])
                bins = np.digitize(ent_rep, qs)
                quart = []
                for bi in range(4):
                    sel = ratios[bins == bi]
                    quart.append(
                        {
                            "entropy_mean": float(np.mean(ent_rep[bins == bi])) if sel.size else None,
                            "byte_ratio_mean": float(np.mean(sel)) if sel.size else None,
                            "n": int(sel.size),
                        }
                    )
                by_block[str(b)]["entropy_quartiles"] = quart
            by_mode[mode] = by_block
        out["temperature"][str(t)] = by_mode
    out["entropy"] = entropies
    return out


def build_upper_from_proj(
    h: np.ndarray,
    mu: np.ndarray,
    basis: np.ndarray,
    p: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    kind: str,
    groups: int,
    row_sq: np.ndarray,
) -> np.ndarray:
    """Upper bound [N, V]. `p` is W @ basis, `c` is W @ mu, `row_sq` is ||W_i||^2."""
    q, rho = project_affine(h.astype(np.float64), mu, basis)
    hidden = w.shape[1]
    if groups == 1:
        delta = np.sqrt(np.maximum(row_sq - np.sum(p * p, axis=1), 0.0))
        resid_term = rho[:, None] * delta[None, :]
    else:
        if hidden % groups != 0:
            raise ValueError(f"hidden {hidden} not divisible by groups {groups}")
        res_w = w.astype(np.float64) - p @ basis.T
        gw = np.linalg.norm(res_w.reshape(w.shape[0], groups, hidden // groups), axis=2)
        hc = h.astype(np.float64) - mu
        res_h = hc - q @ basis.T
        gh = np.linalg.norm(res_h.reshape(h.shape[0], groups, hidden // groups), axis=2)
        resid_term = gh @ gw.T
    if kind == "fp32":
        upper = c[None, :] + q @ p.T + resid_term
    elif kind == "int8":
        q_i, scale, e_l2 = quantize_int8_rows(p)
        p_hat = q_i.astype(np.float64) * scale.astype(np.float64)[:, None]
        qn = np.linalg.norm(q, axis=1)
        upper = (
            c[None, :]
            + q @ p_hat.T
            + resid_term
            + qn[:, None] * e_l2.astype(np.float64)[None, :]
        )
    else:
        raise ValueError(kind)
    return upper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 192, 256, 384])
    ap.add_argument("--groups", type=int, nargs="+", default=[1])
    ap.add_argument("--kinds", nargs="+", default=["fp32", "int8"], choices=["fp32", "int8"])
    ap.add_argument("--bases", nargs="+", default=["affine_pca", "svd_w"])
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.8, 1.0, 1.2])
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 32, 128, 256])
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 50])
    ap.add_argument("--max-rank", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-svd-w", action="store_true")
    args = ap.parse_args()

    self_check()
    t0 = time.time()
    cache = REPO / "analysis" / "cache"
    print("loading GPT-2 and WikiText-2 hidden states...", flush=True)
    w32, h_train, h_eval, meta = load_gpt2_wikitext(
        args.n_train, args.n_eval, args.seq_len, args.stride, cache
    )
    print(json.dumps(meta), flush=True)
    w = w32.astype(np.float64)
    h_train64 = h_train.astype(np.float64)
    h_eval64 = h_eval.astype(np.float64)
    vocab, hidden = w.shape
    print(
        f"W {w.shape}  h_train {h_train64.shape}  h_eval {h_eval64.shape}",
        flush=True,
    )

    print("computing eval logits...", flush=True)
    logits = h_eval64 @ w.T
    print(f"logits {logits.shape} in {time.time() - t0:.1f}s", flush=True)

    max_rank = max(args.max_rank, max(args.ranks))
    bases: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    print("fitting affine PCA on train hidden states...", flush=True)
    mu, basis, explained = fit_affine_pca(h_train64, max_rank)
    bases["affine_pca"] = (
        mu,
        basis,
        {"explained_variance_fraction": explained, "max_rank": max_rank},
    )
    print(f"affine PCA rank-{max_rank} explained {explained:.4f}", flush=True)

    if "svd_w" in args.bases and not args.skip_svd_w:
        print("fitting SVD basis of W (Gram eigh)...", flush=True)
        ts = time.time()
        b = fit_svd_basis(w, max_rank)
        # SVD basis is through the origin. Encode it as affine with mu = 0,
        # which is the correct Cauchy-Schwarz certificate for that subspace.
        bases["svd_w"] = (np.zeros(hidden, dtype=np.float64), b, {"max_rank": max_rank})
        print(f"svd basis in {time.time() - ts:.1f}s", flush=True)

    results = {
        "meta": meta,
        "temperatures": args.temperatures,
        "blocks": args.blocks,
        "replicates": args.replicates,
        "n_eval": int(h_eval64.shape[0]),
        "configs": [],
    }

    for basis_name, (mu_b, basis_full, bmeta) in bases.items():
        if basis_name not in args.bases:
            continue
        print(f"projecting W onto {basis_name} (rank {basis_full.shape[1]})...", flush=True)
        ts = time.time()
        p_full = w @ basis_full
        c_full = w @ mu_b
        row_sq = np.sum(w * w, axis=1)
        print(f"  projection {time.time() - ts:.1f}s", flush=True)
        for rank in args.ranks:
            if rank > basis_full.shape[1]:
                continue
            basis_r = basis_full[:, :rank]
            p_r = p_full[:, :rank]
            for groups in args.groups:
                for kind in args.kinds:
                    label = f"{basis_name}/r{rank}/{kind}/g{groups}"
                    print(f"eval {label}", flush=True)
                    ts = time.time()
                    upper = build_upper_from_proj(
                        h_eval64, mu_b, basis_r, p_r, c_full, w, kind, groups, row_sq
                    )
                    bytes_info = shadow_bytes(vocab, hidden, rank, kind, groups)
                    # svd_w uses mu = 0, but shadow_bytes still charges a center
                    # vector. Drop it when mu is identically zero and c is not stored.
                    if basis_name == "svd_w":
                        bytes_info = dict(bytes_info)
                        bytes_info["shadow_total"] -= bytes_info["token_center"]
                        bytes_info["token_center"] = 0
                    stats = eval_config(
                        logits,
                        upper,
                        args.temperatures,
                        args.blocks,
                        args.replicates,
                        args.seed + rank + groups * 17,
                        bytes_info["shadow_total"],
                        bytes_info["dense_fp32"],
                        bytes_info["row_fp32"],
                        args.topk,
                    )
                    # Headline: block size 1 (tightest partition of this shadow), mean over T.
                    def _mean_ratio(mode: str) -> float | None:
                        vals = []
                        for t in args.temperatures:
                            cell = stats["temperature"][str(t)][mode].get("1")
                            if cell:
                                vals.append(cell["byte_ratio_mean"])
                        return float(np.mean(vals)) if vals else None

                    rec = {
                        "label": label,
                        "basis": basis_name,
                        "rank": rank,
                        "kind": kind,
                        "groups": groups,
                        "basis_meta": bmeta,
                        "bytes": bytes_info,
                        "seconds": time.time() - ts,
                        "mean_byte_ratio_block1_aligned": _mean_ratio("aligned"),
                        "mean_byte_ratio_block1_split": _mean_ratio("split"),
                        "stats": stats,
                    }
                    results["configs"].append(rec)
                    gap = stats["slack"]
                    print(
                        f"  slack p50={gap['p50']:.4f} p99={gap['p99']:.4f} min={gap['min']:.4e}"
                        f"  block1 aligned {rec['mean_byte_ratio_block1_aligned']:.4f}"
                        f"  split {rec['mean_byte_ratio_block1_split']:.4f}"
                        f"  ({rec['seconds']:.1f}s)",
                        flush=True,
                    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "gpt2_shadow.json"
    # Drop nothing: the file is the measurement record. It is large but has no weights.
    out_path.write_text(json.dumps(results))
    print(f"wrote {out_path} in {time.time() - t0:.1f}s", flush=True)

    # Console table: block-size 1 and 128, each temperature, sorted by mean ratio.
    print("\n== block size 1, aligned ceiling, byte ratio by temperature ==")
    ranked = sorted(
        results["configs"], key=lambda r: r["mean_byte_ratio_block1_aligned"] or 9e9
    )
    hdr = f"{'config':<40} {'shadow%':>8} " + " ".join(f"T={t:<4}" for t in args.temperatures)
    print(hdr)
    for rec in ranked:
        binfo = rec["bytes"]
        sh = 100.0 * binfo["shadow_total"] / binfo["dense_fp32"]
        cells = []
        for t in args.temperatures:
            cell = rec["stats"]["temperature"][str(t)]["aligned"].get("1")
            cells.append(f"{cell['byte_ratio_mean']:.3f}" if cell else "  -  ")
        print(f"{rec['label']:<40} {sh:8.2f} " + " ".join(f"{c:>6}" for c in cells))


if __name__ == "__main__":
    main()
