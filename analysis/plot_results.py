#!/usr/bin/env python3
"""Plots for the go/no-go sweep and, when present, the C++ bench JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "analysis" / "results"


def plot_temperature() -> None:
    quant = json.loads((RES / "gpt2_quant_row.json").read_text())
    shadow = json.loads((RES / "gpt2_shadow.json").read_text())
    temps = [float(t) for t in quant["temperatures"]]
    want = {
        "int8/q768/g1": "int8, 1 group",
        "int8/q768/g8": "int8, 8 groups",
        "int8/q768/g16": "int8, 16 groups",
        "fp16/q768/g1": "fp16 residual",
        "int4/q768/g1": "int4, 1 group",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for rec in quant["configs"]:
        if rec["label"] not in want:
            continue
        ys = [
            rec["stats"]["temperature"][str(t)]["aligned"]["1"]["byte_ratio_mean"] for t in temps
        ]
        ax.plot(temps, ys, marker="o", label=want[rec["label"]])
    for rec in shadow["configs"]:
        if rec["label"] != "affine_pca/r384/int8/g1":
            continue
        ys = [
            rec["stats"]["temperature"][str(t)]["aligned"]["1"]["byte_ratio_mean"]
            for t in shadow["temperatures"]
        ]
        ax.plot(
            shadow["temperatures"],
            ys,
            marker="s",
            linestyle="--",
            label="PCA int8 rank 384",
        )
    ax.axhline(1.0, color="black", linewidth=0.8, label="dense fp32")
    ax.set_xlabel("temperature")
    ax.set_ylabel("head bytes / dense fp32 bytes")
    ax.set_title("GPT-2 124M, WikiText-2 validation, block size 1")
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = RES / "byte_ratio_temperature.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path)


def plot_blocks() -> None:
    quant = json.loads((RES / "gpt2_quant_row.json").read_text())
    rec = next(r for r in quant["configs"] if r["label"] == "int8/q768/g8")
    blocks = [int(b) for b in quant["blocks"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mode, style in (("aligned", "-o"), ("split", "--s")):
        for t, alpha in ((0.5, 1.0), (1.0, 0.9), (1.2, 0.75)):
            ys = [rec["stats"]["temperature"][str(t)][mode][str(b)]["byte_ratio_mean"] for b in blocks]
            ax.plot(blocks, ys, style, alpha=alpha, label=f"{mode}, T={t}")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("block size")
    ax.set_ylabel("head bytes / dense fp32 bytes")
    ax.set_title("int8 shadow, 8 residual groups")
    ax.set_xticks(blocks)
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = RES / "byte_ratio_blocksize.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path)


def plot_bench() -> None:
    path = RES / "bench.json"
    if not path.exists():
        return
    bench = json.loads(path.read_text())
    runs = [r for r in bench["runs"] if r["kind"] == "partial" and r["top_k"] == 0 and r["block"] == 1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mode in ("aligned", "split"):
        for flush, name in ((False, "L3 resident"), (True, "cache flushed")):
            xs, ys = [], []
            for r in runs:
                if r["mode"] == mode and r["flush"] is flush:
                    xs.append(r["temperature"])
                    ys.append(r["tokens_per_sec"])
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot(
                    [xs[i] for i in order],
                    [ys[i] for i in order],
                    marker="o",
                    label=f"{mode}, {name}",
                )
    ax.set_xlabel("temperature")
    ax.set_ylabel("head samples / s")
    ax.set_title("GPT-2 head, block size 1, single thread")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RES / "tokens_per_sec.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    plot_temperature()
    plot_blocks()
    plot_bench()


if __name__ == "__main__":
    main()
