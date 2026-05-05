#!/usr/bin/env python3
"""
export_for_report.py — Pull everything the final report needs from W&B in one shot.

For each recognised ablation run (see CONDITIONS in analyze_wandb.py):
  - downloads output.log via the W&B API
  - parses artifact sizes + final scores from the log lines:
        Serialized raw model: ... (N bytes)
        Serialized <method>: N bytes (payload:N raw_torch:N ratio:Rx)
        Total submission size <method>: N bytes
        16MB limit: 16000000 bytes — (OK|OVER)
        final_<method>_roundtrip val_loss:F val_bpb:F eval_time:Nms
        final_<method>_roundtrip_exact val_loss:F val_bpb:F
  - downloads training history (step, train_loss, val_loss, val_bpb)
  - writes one row per run to summary.csv with all metadata
  - writes one curves/<run>.csv per run

Run on the login node (or anywhere with WANDB_API_KEY set) once all jobs are done:

    python experiments/export_for_report.py --out-dir report_data

Then SCP the directory to the local PC for plotting:

    scp -r <user>@<cluster>:~/parameter-golf/parameter-golf/report_data ./report/data/
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import wandb

# Reuse the run-name → condition map already maintained for analyze_wandb.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_wandb import CONDITIONS, condition_key, fetch_history, BASELINE_BPB  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Log parsing
# ──────────────────────────────────────────────────────────────────────────────

RE_RAW = re.compile(r"Serialized raw model:\s*\S+\s*\((\d+)\s*bytes\)")
RE_QUANT = re.compile(
    r"Serialized\s+(\S+):\s*(\d+)\s*bytes\s*"
    r"\(payload:(\d+)\s+raw_torch:(\d+)\s+ratio:([\d.]+)x\)"
)
RE_TOTAL = re.compile(r"Total submission size\s+(\S+):\s*(\d+)\s*bytes")
# Match either ASCII "--" or unicode em-dash "—" between bytes and OK/OVER.
RE_LIMIT = re.compile(r"16MB limit:\s*(\d+)\s*bytes\s*[—\-]+\s*(OK|OVER)")
RE_FINAL_EXACT = re.compile(
    r"final_(\S+)_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"
)
RE_FINAL_FAST = re.compile(
    r"final_(\S+)_roundtrip\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+eval_time:(\d+)ms"
)
RE_PEAK_MEM = re.compile(r"peak memory allocated:\s*(\d+)\s*MiB\s*reserved:\s*(\d+)\s*MiB")


def parse_log(text: str) -> dict:
    """Extract artifact sizes, final scores, peak memory from a captured stdout log.

    A run may serialize multiple quant methods (e.g. int8 + int6); we record ALL
    matches as JSON-encoded lists, but also surface the LAST match per category
    as flat scalar fields, since that's typically the one the run reports as final.
    """
    raw_bytes = [int(m.group(1)) for m in RE_RAW.finditer(text)]
    quants = [
        dict(method=m.group(1), bytes=int(m.group(2)),
             payload=int(m.group(3)), raw_torch=int(m.group(4)), ratio=float(m.group(5)))
        for m in RE_QUANT.finditer(text)
    ]
    totals = [
        dict(method=m.group(1), bytes=int(m.group(2)))
        for m in RE_TOTAL.finditer(text)
    ]
    limits = [
        dict(limit_bytes=int(m.group(1)), status=m.group(2))
        for m in RE_LIMIT.finditer(text)
    ]
    finals_exact = [
        dict(method=m.group(1), val_loss=float(m.group(2)), val_bpb=float(m.group(3)))
        for m in RE_FINAL_EXACT.finditer(text)
    ]
    finals_fast = [
        dict(method=m.group(1), val_loss=float(m.group(2)), val_bpb=float(m.group(3)),
             eval_time_ms=int(m.group(4)))
        for m in RE_FINAL_FAST.finditer(text)
    ]
    peaks = [
        dict(allocated_mib=int(m.group(1)), reserved_mib=int(m.group(2)))
        for m in RE_PEAK_MEM.finditer(text)
    ]

    out: dict = {
        "log_raw_model_bytes_all":   json.dumps(raw_bytes),
        "log_serialized_quant_all":  json.dumps(quants),
        "log_total_submission_all":  json.dumps(totals),
        "log_limit_status_all":      json.dumps(limits),
        "log_final_exact_all":       json.dumps(finals_exact),
        "log_final_fast_all":        json.dumps(finals_fast),
        "log_peak_mem_all":          json.dumps(peaks),
    }
    # Flat scalars: take the LAST occurrence in each category (most runs only have one).
    if raw_bytes:
        out["raw_model_bytes"] = raw_bytes[-1]
    if quants:
        q = quants[-1]
        out["quant_method"] = q["method"]
        out["quant_bytes"] = q["bytes"]
        out["quant_payload_bytes"] = q["payload"]
        out["quant_raw_torch_bytes"] = q["raw_torch"]
        out["quant_ratio"] = q["ratio"]
    if totals:
        t = totals[-1]
        out["total_submission_method"] = t["method"]
        out["total_submission_bytes"] = t["bytes"]
        out["total_submission_mb"] = t["bytes"] / 1_000_000.0
    if limits:
        l = limits[-1]
        out["limit_bytes"] = l["limit_bytes"]
        out["limit_status"] = l["status"]
        out["over_budget"] = l["status"] == "OVER"
    if finals_exact:
        f = finals_exact[-1]
        out["final_method"] = f["method"]
        out["final_val_loss_exact"] = f["val_loss"]
        out["final_val_bpb_exact"] = f["val_bpb"]
    if finals_fast:
        f = finals_fast[-1]
        out["final_eval_time_ms"] = f["eval_time_ms"]
    if peaks:
        p = peaks[-1]
        out["peak_alloc_mib"] = p["allocated_mib"]
        out["peak_reserved_mib"] = p["reserved_mib"]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# W&B helpers
# ──────────────────────────────────────────────────────────────────────────────

def download_log(run, log_dir: Path) -> str | None:
    """Download the run's captured stdout to <log_dir>/<run.name>.log and return its text.

    W&B stores it as 'output.log' attached to the run; we rename on save so the
    filename is readable and won't collide between runs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"{run.name}.log"
    try:
        f = run.file("output.log")
    except Exception as e:
        print(f"    [warn] no output.log: {e}")
        return None
    try:
        # download() honours `replace=True`; save under a temp name then move.
        f.download(root=str(log_dir), replace=True)
    except Exception as e:
        print(f"    [warn] download failed: {e}")
        return None
    src = log_dir / "output.log"
    if src.exists():
        src.replace(target)
    if not target.exists():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"    [warn] read failed: {e}")
        return None


def fetch_full_history(run) -> pd.DataFrame:
    """Like analyze_wandb.fetch_history but also pulls train/loss + train_time
    so we can plot training curves and compute throughput in the report."""
    train_keys = ("step", "val/bpb", "val/loss", "train/loss", "train_time_ms")
    final_keys = ("final/val_bpb", "final/val_loss")

    train_rows = []
    try:
        for row in run.scan_history(keys=list(train_keys), page_size=500):
            train_rows.append({k: row.get(k) for k in train_keys})
    except Exception:
        pass
    if not train_rows:
        # Fall back to the lighter scan if the keyset above is too strict.
        return fetch_history(run)
    df_train = pd.DataFrame(train_rows)
    metric_cols = [c for c in ("val/bpb", "val/loss", "train/loss") if c in df_train.columns]
    if metric_cols:
        df_train = df_train.dropna(subset=metric_cols, how="all")
    if "step" in df_train.columns:
        df_train = df_train.sort_values("step").reset_index(drop=True)

    final_rows = []
    try:
        for row in run.scan_history(keys=list(final_keys), page_size=500):
            final_rows.append({k: row.get(k) for k in final_keys})
    except Exception:
        pass
    df_final = pd.DataFrame(final_rows)

    return pd.concat([df_train, df_final], ignore_index=True, sort=False)


def flat_config(run) -> dict:
    """Flatten run.config into scalar columns. Skip non-scalar / heavy entries."""
    cfg = dict(run.config) if run.config else {}
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[f"cfg.{k}"] = v
        else:
            try:
                out[f"cfg.{k}"] = json.dumps(v)
            except Exception:
                out[f"cfg.{k}"] = str(v)
    return out


def flat_summary(run) -> dict:
    """Pull final/val_bpb, best/val_bpb, etc. from W&B summary as flat scalars."""
    if run.summary is None:
        return {}
    out = {}
    for k, v in dict(run.summary).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[f"summary.{k}"] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-run summary
# ──────────────────────────────────────────────────────────────────────────────

def summarise(run, history: pd.DataFrame, log_text: str | None) -> dict:
    """One row per run: identity + history-derived + log-derived + W&B-summary fields."""
    bpb = history["val/bpb"].dropna() if "val/bpb" in history.columns else pd.Series([], dtype=float)
    final_q = history["final/val_bpb"].dropna() if "final/val_bpb" in history.columns else pd.Series([], dtype=float)

    ckey = condition_key(run.name)
    meta = CONDITIONS.get(ckey, {}) if ckey else {}

    row = {
        "run_id":         run.name,
        "condition_key":  ckey,
        "group":          meta.get("group"),
        "label":          meta.get("label"),
        "state":          run.state,
        "created_at":     getattr(run, "created_at", None),
        "n_val_pts":      int(len(bpb)),
        "max_step":       int(history["step"].dropna().max()) if "step" in history.columns and history["step"].notna().any() else 0,
        "best_train_bpb": float(bpb.min()) if len(bpb) else float("nan"),
        "last_train_bpb": float(bpb.iloc[-1]) if len(bpb) else float("nan"),
        "final_quant_bpb": float(final_q.iloc[-1]) if len(final_q) else float("nan"),
        "baseline_bpb":   BASELINE_BPB,
    }
    row["delta_vs_baseline"] = (
        (row["final_quant_bpb"] if not math.isnan(row["final_quant_bpb"]) else row["best_train_bpb"])
        - BASELINE_BPB
    )
    if log_text:
        row.update(parse_log(log_text))
    row.update(flat_summary(run))
    row.update(flat_config(run))
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="parameter-golf")
    p.add_argument("--entity", default=None)
    p.add_argument("--out-dir", default="report_data", type=Path)
    p.add_argument("--include-pattern", default=None,
                   help="Optional regex; only export runs whose name matches.")
    p.add_argument("--skip-logs", action="store_true",
                   help="Skip downloading output.log (faster; loses size info).")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    curves_dir = out_dir / "curves"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    print(f"Fetching runs from W&B project: {project_path}")
    all_runs = list(api.runs(project_path))
    print(f"  Found {len(all_runs)} total runs.")

    runs = [r for r in all_runs if condition_key(r.name) is not None]
    if args.include_pattern:
        rx = re.compile(args.include_pattern)
        runs = [r for r in runs if rx.search(r.name)]
    print(f"  Recognised {len(runs)} ablation runs to export.\n")

    rows: list[dict] = []
    for i, run in enumerate(runs, 1):
        print(f"[{i:>3}/{len(runs)}] {run.name}  state={run.state}")
        hist = fetch_full_history(run)
        # Per-run curve CSV
        curve_path = curves_dir / f"{run.name}.csv"
        try:
            hist.to_csv(curve_path, index=False)
        except Exception as e:
            print(f"    [warn] failed to write curve: {e}")

        # Log download + parse
        log_text = None if args.skip_logs else download_log(run, logs_dir)

        rows.append(summarise(run, hist, log_text))

    df = pd.DataFrame(rows)
    # Stable column ordering: identity + headline metrics first, then everything else.
    headline_cols = [
        "run_id", "condition_key", "group", "label", "state", "created_at",
        "max_step", "best_train_bpb", "last_train_bpb",
        "final_val_bpb_exact", "final_quant_bpb",
        "total_submission_bytes", "total_submission_mb", "limit_status", "over_budget",
        "quant_method", "quant_bytes", "raw_model_bytes",
        "delta_vs_baseline",
    ]
    cols = [c for c in headline_cols if c in df.columns] + [c for c in df.columns if c not in headline_cols]
    df = df[cols].sort_values("run_id")

    summary_path = out_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nWrote {len(df)} rows to {summary_path}")
    print(f"Per-run curves: {curves_dir} ({len(list(curves_dir.glob('*.csv')))} files)")
    print(f"Per-run logs:   {logs_dir}  ({len(list(logs_dir.glob('*.log')))} files)")

    # Sanity print: which runs are missing size data?
    missing_size = df[df["total_submission_bytes"].isna()]["run_id"].tolist() if "total_submission_bytes" in df.columns else []
    if missing_size:
        print(f"\n[!] {len(missing_size)} runs have no parsed total_submission_bytes:")
        for r in missing_size:
            print(f"      {r}")


if __name__ == "__main__":
    main()
