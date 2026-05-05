#!/usr/bin/env python3
"""Generate every figure for the final report from report/data/.

Inputs (produced by experiments/export_for_report.py):
    report/data/summary.csv
    report/data/curves/<run_id>.csv

Outputs:
    report/figures/*.png
    report/tables/*.tex   (auto-generated LaTeX tables for the results section)

Run from repo root:
    python report/scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report" / "data"
CURVES = DATA / "curves"
FIGS = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

BASELINE = 1.2244
BUDGET_BYTES = 16_000_000

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.25,
})


def save(fig, name: str):
    """Write both PDF (for LaTeX) and PNG (for quick preview)."""
    pdf = FIGS / f"{name}.pdf"
    png = FIGS / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"  wrote {pdf.relative_to(ROOT)} + {png.name}")

# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_summary() -> pd.DataFrame:
    df = pd.read_csv(DATA / "summary.csv")
    df = df.dropna(subset=["final_quant_bpb"])  # drop crashed/empty rows
    # drop dup with no data (e.g. orphan o2_rope_20k row)
    df = df.sort_values(["run_id", "max_step"]).drop_duplicates("run_id", keep="last")
    df["over_budget"] = df["total_submission_bytes"] > BUDGET_BYTES
    return df


def load_curve(run_id: str) -> pd.DataFrame:
    p = CURVES / f"{run_id}.csv"
    if not p.exists():
        return pd.DataFrame()
    c = pd.read_csv(p)
    if "step" in c.columns:
        c = c.dropna(subset=["step"])
    return c


def post_warmup(c: pd.DataFrame, min_step: int = 1000) -> pd.DataFrame:
    if "step" in c.columns:
        c = c[c["step"] >= min_step]
    return c.dropna(subset=["val/bpb"]) if "val/bpb" in c.columns else c


# ──────────────────────────────────────────────────────────────────────────────
# Curve plotting helper
# ──────────────────────────────────────────────────────────────────────────────

def plot_curves(runs: list[tuple[str, str, str]], title: str, name: str,
                ylim: tuple[float, float] | None = None,
                show_baseline: bool = True,
                figsize: tuple[float, float] = (9, 5.0)):
    """runs = list of (run_id, label, color). `name` is the basename (no ext)."""
    fig, ax = plt.subplots(figsize=figsize)
    if show_baseline:
        ax.axhline(BASELINE, color="black", ls="--", lw=1.2,
                   label=f"Reference ({BASELINE})")

    bpbs_for_ylim = []
    endpoints = []  # (y, label, color)
    max_x = 0
    for run_id, label, color in runs:
        c = post_warmup(load_curve(run_id))
        if c.empty:
            print(f"  [warn] no curve data for {run_id}")
            continue
        ax.plot(c["step"], c["val/bpb"], color=color, lw=2.2, label=label, alpha=0.95)
        bpbs_for_ylim.extend(c["val/bpb"].dropna().tolist())
        last = c.iloc[-1]
        endpoints.append((float(last["val/bpb"]), float(last["step"]), color))
        max_x = max(max_x, float(last["step"]))

    if ylim:
        ax.set_ylim(*ylim)
    elif bpbs_for_ylim:
        lo = min(bpbs_for_ylim + [BASELINE]) - 0.012
        hi = max(bpbs_for_ylim + [BASELINE]) + 0.025
        ax.set_ylim(lo, hi)

    # Stagger endpoint annotations vertically to avoid collisions.
    if endpoints and max_x > 0:
        endpoints.sort(key=lambda t: t[0])
        y_min, y_max = ax.get_ylim()
        min_sep = 0.025 * (y_max - y_min)
        prev_y = -1e9
        for y, x, color in endpoints:
            y_show = max(y, prev_y + min_sep)
            ax.annotate(
                f"{y:.3f}",
                xy=(x, y), xytext=(8, (y_show - y) * (200 / (y_max - y_min))),
                textcoords="offset points",
                fontsize=10, color=color, va="center", fontweight="bold",
            )
            prev_y = y_show
        ax.set_xlim(right=max_x * 1.07)

    ax.set_xlabel("Training step")
    ax.set_ylabel("val/bpb (in-memory bf16)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, ls="-", lw=0.5)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Per-axis curve figures
# ──────────────────────────────────────────────────────────────────────────────

def fig_o1_quant():
    plot_curves(
        [("o1_int8_20k", "INT8 PTQ (final 1.216)", "#1f78b4"),
         ("o1_int6_20k", "INT6 QAT+PTQ (final 1.349)", "#e31a1c")],
        "O1: Quantization precision",
        "curves_o1_quant",
    )


def fig_o2_posenc():
    plot_curves(
        [("o2_part_20k", "Partial RoPE (final 1.215)", "#d62728"),
         ("o2_rope_20k", "Full RoPE (final 1.217)", "#ff7f0e"),
         ("o2_learn_20k", "Learned absolute (final 1.235)", "#6a3d9a"),
         ("o2_none_20k", "No PE (final 1.269)", "#7f7f7f")],
        "O2: Positional encoding",
        "curves_o2_posenc",
    )


def fig_o3_optim():
    plot_curves(
        [("o3_muon_20k", "Muon (20k, final 1.217)", "#1f78b4"),
         ("o3_adamw_s1337", "AdamW (5k pilot, final 1.441)", "#7f7f7f"),
         ("o3_adaf_s1337", "Adafactor (5k pilot, final 1.490)", "#33a02c")],
        "O3: Optimizer",
        "curves_o3_optim",
    )


def fig_o4_weightparam():
    plot_curves(
        [("o1_int8_20k", "Dense INT8 (1.216)", "#1f78b4"),
         ("o4_lr128_20k", "Low-rank r=128 (1.301)", "#e6ab02"),
         ("o4_bd4_20k", "Block-diag g=4 (1.383)", "#a6761d"),
         ("o4_rp128_20k", "Random proj 9L (1.396)", "#7f7f7f"),
         ("o4_rp128_12L_20k", "Random proj 12L (1.360)", "#1b9e77")],
        "O4: Weight parameterizations",
        "curves_o4_weightparam",
    )


def fig_o5_resolved():
    plot_curves(
        [("o2_part_20k", "9L INT8 baseline (in-budget, 1.215)", "#1f78b4"),
         ("stacked_int8_safe_11L_mlp2_20k", "+11L INT8 (winner, 1.194)", "#2ca02c"),
         ("stacked_v1_pr_int6_11L_mlp3_20k", "+11L MLP=3 INT6 (fragile, 1.282)", "#e31a1c"),
         ("stacked_tq4_big_11L_mlp3_20k_v2", "+11L MLP=3 TurboQuant-INT4 (failed, 3.599)", "#6a3d9a")],
        "O5: Capacity expansion under three quantizers",
        "curves_o5_resolved",
    )


def fig_o6_mqa():
    plot_curves(
        [("o2_part_20k", "9L GQA-4 baseline (1.215)", "#1f78b4"),
         ("mqa_12L_mlp2_int8_20k", "12L + MQA (1.198)", "#2ca02c")],
        "O6: Multi-Query Attention enables 12-layer expansion",
        "curves_o6_mqa",
    )


def fig_o8_qkgain():
    plot_curves(
        [("o2_part_20k", "QK-gain init=1.5 (default, 1.215)", "#1f78b4"),
         ("o8_qk3p0_20k", "QK-gain init=3.0 (1.215)", "#33a02c"),
         ("o8_qk5p0_20k", "QK-gain init=5.0 (1.216)", "#ff7f0e")],
        "O7: QK-gain initialization (null effect)",
        "curves_o7_qkgain",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed bar chart
# ──────────────────────────────────────────────────────────────────────────────

def fig_multiseed():
    df = load_summary()
    groups = {
        "INT8 PTQ":       ["o1_int8_20k", "o1_int8_s42_20k", "o1_int8_s123_20k"],
        "Partial RoPE":   ["o2_part_20k", "o2_part_s42_20k", "o2_part_s123_20k"],
        "Muon optimizer": ["o3_muon_20k", "o3_muon_s42_20k", "o3_muon_s123_20k"],
    }

    means, stds, points = [], [], []
    for label, runs in groups.items():
        sub = df[df["run_id"].isin(runs)]["final_quant_bpb"].values
        means.append(sub.mean())
        stds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
        points.append(sub)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=stds, capsize=10,
                  color=["#1f78b4", "#d62728", "#33a02c"],
                  alpha=0.85, edgecolor="black", linewidth=1.0,
                  error_kw=dict(elinewidth=1.5, capthick=1.5))
    ax.axhline(BASELINE, color="black", ls="--", lw=1.2, label=f"Reference ({BASELINE})")

    for xi, pts in zip(x, points):
        ax.scatter([xi] * len(pts), pts, color="black", s=30, zorder=5, edgecolor="white", linewidth=0.8)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.001,
                f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylabel("Final bpb (post-quant zlib roundtrip)")
    ax.set_title("Multi-seed replication: mean ± std across seeds {1337, 42, 123}")
    ax.set_ylim(1.205, max(means) + 0.018)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "multiseed_bars")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# LR / warmup heatmap (null result)
# ──────────────────────────────────────────────────────────────────────────────

def fig_lr_warmup():
    df = load_summary()
    sw = df[df["group"] == "O10_hpsweep"].copy()

    # Run names look like o2_part_lr020_wu1000_20k
    def parse(rid: str) -> tuple[float, int]:
        parts = rid.split("_")
        lr_tag = [p for p in parts if p.startswith("lr")][0][2:]
        wu_tag = [p for p in parts if p.startswith("wu")][0][2:]
        return float(lr_tag) / 1000.0, int(wu_tag)

    sw["lr"], sw["warmup"] = zip(*sw["run_id"].map(parse))
    pivot = sw.pivot(index="lr", columns="warmup", values="final_quant_bpb").sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{w}" for w in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{lr}" for lr in pivot.index])
    ax.set_xlabel("Warmup steps")
    ax.set_ylabel("Matrix learning rate")
    ax.set_title("O8: LR/warmup sweep (null result, spread = 0.0019 bpb)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    color="white" if v > pivot.values.mean() else "black",
                    fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, label="Final bpb")
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    save(fig, "lr_warmup_heatmap")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Pareto frontier — the headline figure under the size-vs-quality framing
# ──────────────────────────────────────────────────────────────────────────────

def fig_pareto():
    df = load_summary().copy()

    # Drop pilots, the slide256 alias, catastrophic runs, and HP-sweep clutter from the headline plot.
    df = df[df["max_step"] >= 15000]
    df = df[df["final_quant_bpb"] < 1.5]
    df = df[~df["run_id"].str.contains("slide256")]
    df = df[~df["run_id"].str.endswith("_v2")]
    # The 9 LR/warmup cells cluster on top of o2_part_20k; suppress them in the headline figure.
    df = df[df["group"] != "O10_hpsweep"]

    pts = df[["total_submission_mb", "final_quant_bpb", "run_id", "group"]].copy()
    pts = pts.sort_values("total_submission_mb")
    on_frontier = []
    best_bpb = float("inf")
    for _, row in pts.iterrows():
        if row["final_quant_bpb"] < best_bpb:
            best_bpb = row["final_quant_bpb"]
            on_frontier.append(True)
        else:
            on_frontier.append(False)
    pts["frontier"] = on_frontier

    group_colors = {
        "O1_quant":       "#1f78b4",
        "O2_posenc":      "#d62728",
        "O3_optim":       "#33a02c",
        "O4_weightparam": "#a6761d",
        "O5_stacked":     "#6a3d9a",
        "O8_qkgain":      "#ff7f0e",
        "O9_attn":        "#e7298a",
    }
    group_labels = {
        "O1_quant":       "O1 Quantization",
        "O2_posenc":      "O2 Pos. encoding",
        "O3_optim":       "O3 Optimizer",
        "O4_weightparam": "O4 Weight param.",
        "O5_stacked":     "O5 Stacked",
        "O8_qkgain":      "O7 QK-gain",
        "O9_attn":        "O6 MQA",
    }

    fig, ax = plt.subplots(figsize=(12, 6.8))

    for group, sub in pts.groupby("group"):
        ax.scatter(sub["total_submission_mb"], sub["final_quant_bpb"],
                   s=95, color=group_colors.get(group, "#444"),
                   alpha=0.80, edgecolor="black", linewidth=0.7,
                   label=group_labels.get(group, group), zorder=3)

    # Frontier line
    front = pts[pts["frontier"]].sort_values("total_submission_mb")
    ax.plot(front["total_submission_mb"], front["final_quant_bpb"],
            color="black", lw=1.6, ls="-", marker="o", markersize=13,
            markerfacecolor="none", markeredgewidth=2.0,
            label="Pareto frontier", zorder=5)

    # Only annotate four critical frontier points; intermediate ones are visible from the line.
    KEY_LABELS = {
        "o4_rp128_20k":
            ("Smallest:\nRandProj 9L", (16, 14), "left"),
        "o1_int6_20k":
            ("Elbow:\nINT6 9L", (16, -22), "left"),
        "o2_part_20k":
            ("In-budget winner:\nPartial RoPE 9L\n(1.215 bpb, 15.87 MB)", (-22, 55), "right"),
        "stacked_int8_safe_11L_mlp2_20k":
            ("Over-budget winner:\nINT8 11L\n(1.194 bpb, 19.30 MB)", (-22, 55), "right"),
    }
    for _, row in front.iterrows():
        rid = row["run_id"]
        if rid not in KEY_LABELS:
            continue
        label, off, ha = KEY_LABELS[rid]
        is_winner = "winner" in label.lower()
        ax.annotate(
            label,
            xy=(row["total_submission_mb"], row["final_quant_bpb"]),
            xytext=off, textcoords="offset points",
            fontsize=10.5 if is_winner else 10,
            color="black", ha=ha, va="center",
            fontweight="bold" if is_winner else "normal",
            bbox=dict(boxstyle="round,pad=0.32",
                      fc="#fff8d6" if is_winner else "white",
                      ec="black",
                      lw=1.0 if is_winner else 0.5,
                      alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
            zorder=10,
        )

    # 16MB reference line + label
    ax.axvline(16.0, color="red", ls=":", lw=1.5, alpha=0.85,
               label="16 MB reference")
    ax.axhline(BASELINE, color="black", ls="--", lw=1.2, alpha=0.85,
               label=f"Reference bpb ({BASELINE})")

    # Extra right margin so the over-budget winner annotation is comfortable.
    x_min, x_max = pts["total_submission_mb"].min(), pts["total_submission_mb"].max()
    ax.set_xlim(x_min - 0.6, x_max + 1.4)

    ax.set_xlabel("Total submission size (MB, weights + code, post-zlib)")
    ax.set_ylabel("Final bpb (post-quant zlib roundtrip)")
    ax.set_title("Pareto frontier: size vs. quality across all 20k-step runs")
    ax.grid(True, alpha=0.25, ls="-", lw=0.5)
    # Push the legend below the plot so it never overlaps the scatter.
    ax.legend(fontsize=10, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=5,
              frameon=True, framealpha=0.95)
    fig.tight_layout()
    save(fig, "pareto_frontier")
    plt.close(fig)
    print(f"    frontier points: {front['run_id'].tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# Eval-protocol bars
# ──────────────────────────────────────────────────────────────────────────────

def fig_eval_protocol():
    df = load_summary()
    pairs = [
        ("stacked_int8_safe_11L_mlp2_20k", "Standard eval"),
        ("stacked_int8_safe_11L_mlp2_slide256_20k", "Sliding-window eval (stride=256)"),
    ]
    vals = [df[df["run_id"] == rid]["final_quant_bpb"].values[0] for rid, _ in pairs]
    labels = [lab for _, lab in pairs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=["#1f78b4", "#33a02c"],
                  edgecolor="black", linewidth=1.0, width=0.55)
    ax.axhline(BASELINE, color="black", ls="--", lw=1.2, label=f"Reference ({BASELINE})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Final bpb (post-quant zlib roundtrip)")
    ax.set_title("Eval protocol effect (same 11L INT8 weights)")
    ax.set_ylim(min(vals) - 0.025, BASELINE + 0.030)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    save(fig, "eval_protocol_bars")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX tables
# ──────────────────────────────────────────────────────────────────────────────

def write_results_table():
    """Master results table: every reportable 20k run as one row."""
    df = load_summary().copy()
    df = df[df["max_step"] >= 15000]
    df = df.sort_values(["group", "final_quant_bpb"])

    rows = []
    for _, r in df.iterrows():
        # Trim redundant suffixes — every reportable run is _20k by filter.
        run = r["run_id"]
        if run.endswith("_20k"):
            run = run[:-4]
        # Drop the leading axis tag (o1_, o2_, …) since the Group column already encodes it.
        if len(run) > 3 and run[0] == "o" and run[1].isdigit() and run[2] == "_":
            run = run[3:]
        rows.append({
            "group":  r["group"] or "",
            "run":    run.replace("_", r"\_"),
            "train":  f"{r['best_train_bpb']:.4f}" if not pd.isna(r["best_train_bpb"]) else "—",
            "final":  f"{r['final_quant_bpb']:.4f}",
            "size":   f"{r['total_submission_mb']:.2f}",
            "delta":  f"{r['final_quant_bpb'] - BASELINE:+.4f}",
            "budget": "OVER" if r["over_budget"] else "OK",
        })

    out = TABLES / "results_full.tex"
    with out.open("w", encoding="utf-8") as f:
        f.write("% auto-generated by report/scripts/make_figures.py\n")
        f.write("\\begin{tabular}{@{}llcccc c@{}}\n")
        f.write("\\toprule\n")
        f.write("Group & Run & Train bpb & Final bpb & Size (MB) & $\\Delta$ vs.\\ baseline & 16MB \\\\\n")
        f.write("\\midrule\n")
        last_group = None
        for r in rows:
            if r["group"] != last_group and last_group is not None:
                f.write("\\midrule\n")
            last_group = r["group"]
            f.write(f"{r['group'].replace('_', ' ')} & \\texttt{{{r['run']}}} & "
                    f"{r['train']} & {r['final']} & {r['size']} & {r['delta']} & {r['budget']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  wrote {out.relative_to(ROOT)}")


def write_multiseed_table():
    df = load_summary()
    groups = {
        "INT8 PTQ":       ["o1_int8_20k", "o1_int8_s42_20k", "o1_int8_s123_20k"],
        "Partial RoPE":   ["o2_part_20k", "o2_part_s42_20k", "o2_part_s123_20k"],
        "Muon optimizer": ["o3_muon_20k", "o3_muon_s42_20k", "o3_muon_s123_20k"],
    }
    out = TABLES / "multiseed.tex"
    with out.open("w", encoding="utf-8") as f:
        f.write("% auto-generated\n")
        f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("Condition & Seed 1337 & Seed 42 & Seed 123 & Mean $\\pm$ std \\\\\n\\midrule\n")
        for label, runs in groups.items():
            vals = [df[df["run_id"] == r]["final_quant_bpb"].values[0] for r in runs]
            mean = np.mean(vals); std = np.std(vals, ddof=1)
            f.write(f"{label} & {vals[0]:.4f} & {vals[1]:.4f} & {vals[2]:.4f} & "
                    f"{mean:.4f} $\\pm$ {std:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  wrote {out.relative_to(ROOT)}")


def write_lr_warmup_table():
    df = load_summary()
    sw = df[df["group"] == "O10_hpsweep"].copy()

    def parse(rid):
        parts = rid.split("_")
        lr = float([p for p in parts if p.startswith("lr")][0][2:]) / 1000.0
        wu = int([p for p in parts if p.startswith("wu")][0][2:])
        return lr, wu
    sw["lr"], sw["warmup"] = zip(*sw["run_id"].map(parse))
    pivot = sw.pivot(index="lr", columns="warmup", values="final_quant_bpb").sort_index().sort_index(axis=1)

    out = TABLES / "lr_warmup.tex"
    with out.open("w", encoding="utf-8") as f:
        f.write("% auto-generated\n")
        f.write("\\begin{tabular}{@{}l" + "c" * len(pivot.columns) + "@{}}\n\\toprule\n")
        f.write("Matrix LR & " + " & ".join(f"warmup={w}" for w in pivot.columns) + " \\\\\n\\midrule\n")
        for lr, row in pivot.iterrows():
            f.write(f"{lr} & " + " & ".join(f"{v:.4f}" for v in row.values) + " \\\\\n")
        f.write(f"\\bottomrule\n\\end{{tabular}}\n")
    print(f"  wrote {out.relative_to(ROOT)}")


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating figures …")
    fig_o1_quant()
    fig_o2_posenc()
    fig_o3_optim()
    fig_o4_weightparam()
    fig_o5_resolved()
    fig_o6_mqa()
    fig_o8_qkgain()
    fig_multiseed()
    fig_lr_warmup()
    fig_pareto()
    fig_eval_protocol()

    print("\nGenerating tables …")
    write_results_table()
    write_multiseed_table()
    write_lr_warmup_table()

    print("\nDone.")


if __name__ == "__main__":
    main()
