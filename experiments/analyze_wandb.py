#!/usr/bin/env python3
"""
analyze_wandb.py — Download W&B run histories and report facts only.

Usage:
    python experiments/analyze_wandb.py
    python experiments/analyze_wandb.py --project parameter-golf --entity spatadia
    python experiments/analyze_wandb.py --out-dir ./analysis

Outputs:
    <out-dir>/curves_<group>.png  val/bpb curves per group
    <out-dir>/summary.csv         per-run metric table
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# ──────────────────────────────────────────────────────────────────────────────
# Static config
# ──────────────────────────────────────────────────────────────────────────────

# Reference bpb from the public leaderboard / your confirmed baseline job
BASELINE_BPB = 1.2244

# ──────────────────────────────────────────────────────────────────────────────
# Condition registry
# Each key is the RUN_ID prefix we used in the sbatch commands.
# ──────────────────────────────────────────────────────────────────────────────
CONDITIONS = {
    # O1: Quantisation
    "o1_bf16": dict(
        group="O1_quant", label="BF16 (no quant)",
        script="experiments/train_quant.py",
        env={"QUANT_BITS": "16"},
    ),
    "o1_int8": dict(
        group="O1_quant", label="INT8 PTQ (baseline)",
        script="experiments/train_quant.py",
        env={"QUANT_BITS": "8"},
    ),
    "o1_int6": dict(
        group="O1_quant", label="INT6 QAT+PTQ",
        script="experiments/train_quant.py",
        env={"QUANT_BITS": "6"},
    ),
    # O2: Positional encoding
    "o2_rope": dict(
        group="O2_posenc", label="RoPE (full, baseline)",
        script="experiments/train_posenc.py",
        env={"POS_ENC": "rope"},
    ),
    "o2_part": dict(
        group="O2_posenc", label="Partial RoPE",
        script="experiments/train_posenc.py",
        env={"POS_ENC": "partial"},
    ),
    "o2_none": dict(
        group="O2_posenc", label="No PE",
        script="experiments/train_posenc.py",
        env={"POS_ENC": "none"},
    ),
    "o2_learn": dict(
        group="O2_posenc", label="Learned PE",
        script="experiments/train_posenc.py",
        env={"POS_ENC": "learned"},
    ),
    # O3: Optimiser
    "o3_muon": dict(
        group="O3_optim", label="Muon (baseline)",
        script="experiments/train_optim.py",
        env={"OPTIMIZER_TYPE": "muon"},
    ),
    "o3_adamw": dict(
        group="O3_optim", label="AdamW",
        script="experiments/train_optim.py",
        env={"OPTIMIZER_TYPE": "adamw"},
    ),
    "o3_adaf": dict(
        group="O3_optim", label="Adafactor",
        script="experiments/train_optim.py",
        env={"OPTIMIZER_TYPE": "adafactor"},
    ),
    # O4: Weight parameterisation
    "o4_dense": dict(
        group="O4_weightparam", label="Dense (O4 baseline)",
        script="experiments/train_weightparam.py",
        env={"PARAM_STYLE": "dense", "QUANT_BITS": "8"},
    ),
    "o4_lr128": dict(
        group="O4_weightparam", label="Low-rank r=128",
        script="experiments/train_weightparam.py",
        env={"PARAM_STYLE": "lowrank", "PARAM_RANK": "128", "QUANT_BITS": "8"},
    ),
    "o4_rp128": dict(
        group="O4_weightparam", label="RandProj k=128",
        script="experiments/train_weightparam.py",
        env={"PARAM_STYLE": "randproj", "PARAM_RANK": "128", "PARAM_SEED": "20260418", "QUANT_BITS": "8"},
    ),
    "o4_bd4": dict(
        group="O4_weightparam", label="Block-diag g=4",
        script="experiments/train_weightparam.py",
        env={"PARAM_STYLE": "blockdiag", "PARAM_GROUPS": "4", "QUANT_BITS": "8"},
    ),
    # O5: Stacked winners (Partial RoPE + INT6 + reallocated capacity)
    "stacked_v1_pr_int6_11L_mlp3": dict(
        group="O5_stacked", label="Partial RoPE + INT6 + 11L + MLP=3",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "6",
            "NUM_LAYERS": "11", "MLP_MULT": "3",
        },
    ),
    "stacked_v2_pr_int6_13L_mlp2": dict(
        group="O5_stacked", label="Partial RoPE + INT6 + 13L + MLP=2",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "6",
            "NUM_LAYERS": "13", "MLP_MULT": "2",
        },
    ),
    "stacked_v2_pr_int6_9L_mlp4": dict(
        group="O5_stacked", label="Partial RoPE + INT6 + 9L + MLP=4",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "6",
            "NUM_LAYERS": "9", "MLP_MULT": "4",
        },
    ),
    # O5b: TurboQuant-INT4 — addresses INT6 fragility via random rotation + Lloyd-Max
    "stacked_tq4_sanity_9L_mlp2": dict(
        group="O5_stacked", label="Partial RoPE + TQ4 (sanity, baseline arch)",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "4",
            "NUM_LAYERS": "9", "MLP_MULT": "2",
        },
    ),
    "stacked_tq4_big_11L_mlp3": dict(
        group="O5_stacked", label="Partial RoPE + TQ4 + 11L + MLP=3",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "4",
            "NUM_LAYERS": "11", "MLP_MULT": "3",
        },
    ),
    "stacked_int8_safe_11L_mlp2_slide256": dict(
        group="O5_stacked", label="Partial RoPE + INT8 + 11L + MLP=2 + slide256",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "8",
            "NUM_LAYERS": "11", "MLP_MULT": "2",
            "SLIDING_EVAL_STRIDE": "256",
        },
    ),
    "stacked_int8_safe_11L_mlp2": dict(
        group="O5_stacked", label="Partial RoPE + INT8 + 11L + MLP=2 (hedge)",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "8",
            "NUM_LAYERS": "11", "MLP_MULT": "2",
        },
    ),
    # O8: QK-Gain sweep (partial RoPE + INT8 + 9L/MLP=2 baseline arch)
    "o8_qk3p0": dict(
        group="O8_qkgain", label="QK-Gain=3.0",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "8",
            "NUM_LAYERS": "9", "MLP_MULT": "2",
            "QK_GAIN_INIT": "3.0",
        },
    ),
    "o8_qk5p0": dict(
        group="O8_qkgain", label="QK-Gain=5.0",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "8",
            "NUM_LAYERS": "9", "MLP_MULT": "2",
            "QK_GAIN_INIT": "5.0",
        },
    ),
    # O9: Attention-head allocation (multi-query / grouped-query)
    "mqa_12L_mlp2_int8": dict(
        group="O9_attn", label="MQA + 12L + MLP=2 + INT8",
        script="experiments/train_weightparam.py",
        env={
            "POS_ENC": "partial", "ROPE_DIMS": "32",
            "PARAM_STYLE": "dense", "QUANT_BITS": "8",
            "NUM_LAYERS": "12", "MLP_MULT": "2",
            "NUM_KV_HEADS": "1",
        },
    ),
}

# O10: LR / warmup sweep on the in-budget winner (partial RoPE 9L MLP=2 INT8).
# 3 LR x 3 warmup = 9 cells. Run-name format: o2_part_lr{NN}_wu{KKKK}_20k where
# NN encodes MATRIX_LR (020=0.02, 040=0.04, 080=0.08) and KKKK is WARMUP_STEPS.
_O2_PART_BASE_ENV = {
    "POS_ENC": "partial", "ROPE_DIMS": "32",
    "PARAM_STYLE": "dense", "QUANT_BITS": "8",
    "NUM_LAYERS": "9", "MLP_MULT": "2",
}
for _lr_tag, _lr_val in [("020", "0.02"), ("040", "0.04"), ("080", "0.08")]:
    for _wu in (1000, 2000, 3000):
        CONDITIONS[f"o2_part_lr{_lr_tag}_wu{_wu}"] = dict(
            group="O10_hpsweep",
            label=f"Partial RoPE (LR={_lr_val}, warmup={_wu})",
            script="experiments/train_weightparam.py",
            env={**_O2_PART_BASE_ENV, "MATRIX_LR": _lr_val, "WARMUP_STEPS": str(_wu)},
        )

# Colours per group (one per condition in that group)
GROUP_COLORS = {
    "O1_quant":       ["#e41a1c", "#377eb8", "#4daf4a"],
    "O2_posenc":      ["#984ea3", "#ff7f00", "#a65628", "#f781bf"],
    "O3_optim":       ["#999999", "#1f78b4", "#33a02c"],
    "O4_weightparam": ["#e6ab02", "#a6761d", "#666666", "#1b9e77"],
    "O5_stacked":     ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2"],
    "O8_qkgain":      ["#1f77b4", "#ff7f0e"],
    "O9_attn":        ["#2ca02c"],
    "O10_hpsweep":    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
                       "#e6ab02", "#a6761d", "#666666", "#1f77b4"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def condition_key(run_name: str) -> str | None:
    """Map a W&B run name to a CONDITIONS key, or None if not recognised.
    Returns the LONGEST matching prefix so e.g. 'stacked_int8_safe_11L_mlp2_slide256'
    is preferred over its shorter sibling 'stacked_int8_safe_11L_mlp2'."""
    best: str | None = None
    for key in CONDITIONS:
        if run_name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return best


def fetch_history(run) -> pd.DataFrame:
    """Download scalar history. W&B's scan_history(keys=...) returns only rows
    where ALL listed keys exist (intersection), so we scan twice and concat:
    once for per-step training-time val metrics, once for the single end-of-run
    'final/val_bpb' roundtrip metric."""
    train_keys = ("step", "val/bpb", "val/loss")
    final_keys = ("final/val_bpb", "final/val_loss")

    train_rows = []
    try:
        for row in run.scan_history(keys=list(train_keys), page_size=500):
            train_rows.append({k: row.get(k) for k in train_keys})
    except Exception:
        pass

    final_rows = []
    try:
        for row in run.scan_history(keys=list(final_keys), page_size=500):
            final_rows.append({k: row.get(k) for k in final_keys})
    except Exception:
        pass

    df_train = pd.DataFrame(train_rows)
    df_final = pd.DataFrame(final_rows)

    # Drop pure-empty training rows (rows where val/bpb and val/loss are both NaN).
    metric_cols = [k for k in ("val/bpb", "val/loss") if k in df_train.columns]
    if metric_cols:
        df_train = df_train.dropna(subset=metric_cols, how="all")
    if "step" in df_train.columns:
        df_train = df_train.sort_values("step").reset_index(drop=True)

    # Concat: training rows first, then a single final row with NaN for step/val/*.
    df = pd.concat([df_train, df_final], ignore_index=True, sort=False)
    return df


def is_diverged(bpb_series: pd.Series, max_final_bpb: float = 1.8) -> bool:
    """
    True only if the run genuinely failed:
      - no data points at all
      - NaN anywhere
      - final bpb is unreasonably high (model never converged)
      - bpb increased in the second half of training (actually diverging)
    Does NOT flag early-training high bpb, which is expected from a random init.
    """
    vals = bpb_series.dropna().values
    if len(vals) == 0:
        return True
    if np.any(np.isnan(vals)):
        return True
    # Final value still too high to be a real model
    if vals[-1] > max_final_bpb:
        return True
    # Bpb rising in the second half → actually diverging mid-run
    if len(vals) >= 4:
        mid = len(vals) // 2
        if vals[mid:].mean() > vals[:mid].mean() * 1.05:   # 5 % sustained increase
            return True
    return False


def summarise_run(run, history: pd.DataFrame) -> dict:
    # Crashed runs with no logged val rows produce an empty DataFrame with no
    # 'val/bpb' column at all — treat as zero data points.
    if "val/bpb" in history.columns:
        bpb = history["val/bpb"].dropna()
    else:
        bpb = pd.Series([], dtype=float)
    # final/val_bpb is logged once at end of run, AFTER quantize+zlib roundtrip.
    # This is the leaderboard-equivalent number; baseline 1.2244 is also a final number.
    if "final/val_bpb" in history.columns:
        fq = history["final/val_bpb"].dropna()
        final_quant_bpb = float(fq.iloc[-1]) if len(fq) else float("nan")
    else:
        final_quant_bpb = float("nan")
    summary = {
        "run_id":          run.name,
        "state":           run.state,
        "max_step":        int(bpb.index[-1]) if len(bpb) else 0,
        "n_val_pts":       len(bpb),
        "final_bpb":       float(bpb.iloc[-1]) if len(bpb) else float("nan"),
        "best_bpb":        float(bpb.min())    if len(bpb) else float("nan"),
        "final_quant_bpb": final_quant_bpb,
        "diverged":        is_diverged(bpb),
        "finished":        run.state == "finished",
    }
    # Actual step (from the step column if present)
    if "step" in history.columns and len(history):
        last_step = history["step"].dropna().max()
        if not math.isnan(last_step):
            summary["max_step"] = int(last_step)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_group(group_id: str, runs_data: list[dict], out_path: Path):
    """
    runs_data: list of dicts with keys: label, history (DataFrame), diverged, finished, run_id
    Plot only the post-warmup region (step >= 1000) so the random-init spike at
    bpb~=4 doesn't crush the interesting 1.20-1.30 band into one visual line.
    Skip 5k pilot runs when a 20k full run with the same label is present.
    """
    # Drop pilots whose label already has a 20k twin so the legend stays clean.
    twenty_k_labels = {rd["label"] for rd in runs_data
                       if not rd["history"].empty and rd["history"].get("step", pd.Series([0])).max() >= 15000}
    filtered = []
    for rd in runs_data:
        max_step = rd["history"]["step"].max() if "step" in rd["history"].columns and not rd["history"].empty else 0
        if max_step < 15000 and rd["label"] in twenty_k_labels:
            continue  # skip 5k pilot — its 20k twin will be plotted
        filtered.append(rd)
    runs_data = filtered

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = GROUP_COLORS.get(group_id, plt.rcParams["axes.prop_cycle"].by_key()["color"])

    ax.axhline(BASELINE_BPB, color="black", linestyle="--", linewidth=1.2,
               label=f"Leaderboard baseline ({BASELINE_BPB})")

    plotted_values = []  # all in-range bpb values, used for y-limit auto-fit
    plotted_endpoints = []  # (x, y, color, label) for endpoint annotations
    max_x_seen = 0
    for i, rd in enumerate(runs_data):
        hist = rd["history"]
        if hist.empty or "val/bpb" not in hist.columns:
            continue
        # Drop early steps so the y-axis can focus on the converged region.
        if "step" in hist.columns:
            hist = hist[hist["step"] >= 1000]
        if hist.empty:
            continue
        col = colors[i % len(colors)]
        style = "-" if not rd["diverged"] else ":"
        label = rd["label"]
        if rd["diverged"]:
            label += " [DIVERGED]"
        elif not rd["finished"]:
            label += " [running]"
        bpb_series = hist["val/bpb"].dropna()
        if not len(bpb_series):
            continue
        steps = hist["step"] if "step" in hist.columns else hist.index
        # Few-point pilot runs get markers so they're visible against long curves.
        marker = "o" if len(bpb_series) < 15 else None
        markersize = 5 if marker else 0
        ax.plot(
            steps, hist["val/bpb"],
            linestyle=style, color=col, linewidth=1.8, label=label,
            marker=marker, markersize=markersize, markeredgecolor=col, markerfacecolor=col,
        )
        plotted_values.extend(bpb_series.tolist())
        # Endpoint position (last non-NaN step/value).
        last_idx = bpb_series.index[-1]
        last_x = float(steps.loc[last_idx]) if hasattr(steps, "loc") else float(steps[last_idx])
        last_y = float(bpb_series.iloc[-1])
        plotted_endpoints.append((last_x, last_y, col, f"{last_y:.3f}"))
        max_x_seen = max(max_x_seen, last_x)

    # Auto y-limits over ALL plotted values (not just finals) so the descent is visible.
    if plotted_values:
        lo = min(plotted_values + [BASELINE_BPB]) - 0.02
        hi = max(plotted_values + [BASELINE_BPB]) + 0.04
        ax.set_ylim(lo, hi)

    # Annotate each curve's endpoint with its final bpb so values are readable
    # even when lines are short or close together. Pad x by ~3% of the axis range.
    if plotted_endpoints and max_x_seen > 0:
        x_pad = 0.012 * max_x_seen
        # Stagger labels vertically when two endpoints are very close in y.
        plotted_endpoints.sort(key=lambda t: t[1])
        prev_y = -1e9
        y_min_sep = 0.012 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for x, y, col, txt in plotted_endpoints:
            y_show = max(y, prev_y + y_min_sep)
            ax.annotate(
                txt, xy=(x, y), xytext=(x + x_pad, y_show),
                fontsize=8, color=col, va="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6) if y_show != y else None,
            )
            prev_y = y_show
        # Extend xlim so endpoint labels don't get clipped.
        ax.set_xlim(right=max_x_seen * 1.13)

    ax.set_xlabel("Step")
    ax.set_ylabel("val/bpb (training-time)")
    ax.set_title(group_id.replace("_", " — "))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse W&B pilot runs and recommend next steps.")
    parser.add_argument("--project",  default="parameter-golf",
                        help="W&B project name (default: parameter-golf)")
    parser.add_argument("--entity",   default=None,
                        help="W&B entity / username (default: your default entity)")
    parser.add_argument("--out-dir",  default="analysis", type=Path,
                        help="Directory for output files (default: ./analysis)")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Connect to W&B ────────────────────────────────────────────────────────
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    print(f"\nFetching runs from W&B project: {project_path}")

    try:
        all_runs = list(api.runs(project_path))
    except Exception as e:
        print(f"ERROR: Could not fetch runs — {e}")
        print("Make sure you have run `wandb login` and the project name is correct.")
        sys.exit(1)

    print(f"  Found {len(all_runs)} total runs in project.")

    # ── Filter to our ablation runs ───────────────────────────────────────────
    ablation_runs = [r for r in all_runs if condition_key(r.name) is not None]
    print(f"  Recognised {len(ablation_runs)} ablation runs.\n")

    if not ablation_runs:
        print("No runs matched known condition prefixes. Check --project / --entity.")
        print("Known prefixes:", list(CONDITIONS.keys()))
        sys.exit(1)

    # ── Download history for each run ─────────────────────────────────────────
    print("Downloading run histories …")
    run_records = {}   # run_name -> {summary, history}
    for run in ablation_runs:
        print(f"  {run.name:40s}  state={run.state}")
        hist = fetch_history(run)
        summ = summarise_run(run, hist)
        run_records[run.name] = {"summary": summ, "history": hist, "run": run}

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = [v["summary"] for v in run_records.values()]
    df_summary = pd.DataFrame(rows).sort_values("run_id")
    # delta_vs_baseline uses final_quant_bpb (post-quant zlib roundtrip — what the
    # leaderboard actually scores) when available, else falls back to best_bpb.
    df_summary["delta_vs_baseline"] = df_summary.apply(
        lambda r: (r["final_quant_bpb"] if not math.isnan(r["final_quant_bpb"]) else r["best_bpb"]) - BASELINE_BPB,
        axis=1,
    )
    summary_path = out_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)

    # Pretty print — show both training-time best_bpb and post-quant leaderboard score
    print("\n" + "=" * 110)
    print(f"{'RUN_ID':<32} {'STATE':<10} {'MAX_STEP':>8} {'BEST_BPB':>9} "
          f"{'FINAL_Q':>9} {'DELTA':>7}  NOTES")
    print("  (BEST_BPB = best in-training val/bpb; FINAL_Q = post-quant zlib roundtrip = leaderboard score)")
    print("  (DELTA   = (FINAL_Q if available else BEST_BPB) - baseline 1.2244)")
    print("-" * 110)
    for _, r in df_summary.iterrows():
        notes = []
        if r["diverged"]:   notes.append("diverged-or-incomplete")
        if not r["finished"] and r["state"] != "running": notes.append("did-not-finish")
        if r["state"] == "running": notes.append("still running")
        fq_str = f"{r['final_quant_bpb']:>9.4f}" if not math.isnan(r["final_quant_bpb"]) else f"{'—':>9}"
        print(
            f"{r['run_id']:<32} {r['state']:<10} {int(r['max_step']):>8} "
            f"{r['best_bpb']:>9.4f} {fq_str} "
            f"{r['delta_vs_baseline']:>+7.4f}  "
            + ", ".join(notes)
        )
    print("=" * 110)
    print(f"Baseline bpb: {BASELINE_BPB} (post-quant)")
    print(f"Summary saved to: {summary_path}\n")

    # ── Plots per group ───────────────────────────────────────────────────────
    print("Generating convergence plots …")
    groups = defaultdict(list)
    for rname, data in run_records.items():
        ckey = condition_key(rname)
        if ckey is None:
            continue
        meta = CONDITIONS[ckey]
        groups[meta["group"]].append({
            "label":    meta["label"],
            "history":  data["history"],
            "diverged": data["summary"]["diverged"],
            "finished": data["summary"]["finished"],
        })

    for group_id, runs_data in groups.items():
        plot_path = out_dir / f"curves_{group_id.lower()}.png"
        plot_group(group_id, runs_data, plot_path)
    print()


if __name__ == "__main__":
    main()
