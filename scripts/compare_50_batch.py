#!/usr/bin/env python3
"""Compare final_scores across multiple WBIA runs on the same batch.

Reads the sv_on_true configuration from each run, extracts per-annot
csum (annot_score_list) and name scores, and produces pairwise comparison
metrics: daid-aware correlation, positional Spearman, name overlap.

Usage:
  python scripts/compare_50_batch.py [--runs-dir artifacts/wbia-oracle]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_oracle_arr(row, col, run_dir):
    """Load a npy-sidecar or inlined array from a parquet row."""
    val = row[col]
    if isinstance(val, np.ndarray):
        return val
    parsed = json.loads(val) if isinstance(val, str) else val
    if isinstance(parsed, dict) and "npy_path" in parsed:
        npy_path = parsed["npy_path"]
        if os.path.isabs(npy_path):
            full = Path(npy_path)
        else:
            full = run_dir / "final_scores" / npy_path
        return np.load(str(full))
    if isinstance(parsed, dict) and "values" in parsed:
        return np.array(parsed["values"])
    return np.array(parsed)


def _safe_name(label: str) -> str:
    parts = label.split("-")
    for i, p in enumerate(parts):
        if p == "wbia":
            return "-".join(parts[i + 1 : i + 3]) if len(parts) > i + 2 else label
    if label.startswith("hotspotter"):
        return "hotspotter"
    return label


def load_final_scores(run_dir: Path):
    """Load sv_on_true or default final_scores from a run directory."""
    score_dir = run_dir / "final_scores"
    files = sorted(score_dir.glob("sv_on_true_*.parquet"))
    if not files:
        files = sorted(score_dir.glob("default_*.parquet"))
    results = []
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            qaid = int(row["qaid"])
            qnid = int(row["qnid"])
            daids = _load_oracle_arr(row, "daid_list_array", run_dir).astype(int)
            csum = _load_oracle_arr(row, "annot_score_list_array", run_dir)
            nsum = _load_oracle_arr(row, "name_score_list_array", run_dir)
            scores = _load_oracle_arr(row, "score_list_array", run_dir)
            results.append(
                {
                    "qaid": qaid,
                    "qnid": qnid,
                    "daids": daids,
                    "csum": csum,
                    "nsum": nsum,
                    "scores": scores,
                }
            )
    return results


def daid_aware_rho(daids_a, csum_a, daids_b, csum_b):
    """Spearman ρ aligned by daid (not position)."""
    by_a = {int(d): float(c) for d, c in zip(daids_a, csum_a) if np.isfinite(c)}
    by_b = {int(d): float(c) for d, c in zip(daids_b, csum_b) if np.isfinite(c)}
    common = sorted(set(by_a) & set(by_b))
    if len(common) < 2:
        return float("nan"), 0
    va = [by_a[d] for d in common]
    vb = [by_b[d] for d in common]
    from scipy.stats import spearmanr

    rho, p = spearmanr(va, vb)
    return rho, len(common)


def positional_rho(a, b):
    """Positional Spearman ρ on common-length arrays."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), 0
    n = min(len(a), len(b))
    from scipy.stats import spearmanr

    rho, p = spearmanr(a[:n], b[:n])
    return rho, n


def top_k_overlap(daids_a, daids_b, k=5):
    """Fraction of top-k daids that match (unordered)."""
    top_a = set(daids_a[:k].tolist())
    top_b = set(daids_b[:k].tolist())
    return len(top_a & top_b) / k


def daid_jaccard(daids_a, daids_b):
    """Jaccard index of daid sets."""
    sa = set(daids_a.tolist())
    sb = set(daids_b.tolist())
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def _fmt_rho(rho, n=None):
    if np.isnan(rho):
        return "  N/A  "
    s = f"{rho:+.4f}"
    if n:
        s += f" (n={n})"
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "artifacts", "wbia-oracle"
        ),
    )
    parser.add_argument("--skip-nightly", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: runs dir not found: {runs_dir}", file=sys.stderr)
        return 1

    run_dirs = sorted(runs_dir.glob("*batch50-*"))
    if not run_dirs:
        print(f"No batch50 runs found in {runs_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(run_dirs)} WBIA batch50 runs:\n")
    runs = {}
    for rd in run_dirs:
        label = rd.name
        if args.skip_nightly and "nightly" in label:
            continue
        scores = load_final_scores(rd)
        print(
            f"  {label}: {len(scores)} results ({len(set(r['qaid'] for r in scores))} queries)"
        )
        runs[label] = scores

    labels = sorted(runs.keys())
    n = len(labels)
    if n < 2:
        print("\nNeed at least 2 runs to compare.")
        return 1

    print(f"\n{'='*80}")
    print("PAIRWISE COMPARISON (sv_on_true, daid-aware)")
    print(f"{'='*80}\n")

    short = {l: _safe_name(l) for l in labels}

    # Build matrix data
    for metric_name, metric_fn in [
        ("Daid Jaccard", lambda a, b: daid_jaccard(a["daids"], b["daids"])),
    ]:
        pass

    # Full pairwise comparison per query
    for qi in range(3):
        print(f"\n{'─'*70}")
        print(f"  Query {qi}")
        print(f"{'─'*70}")

        # Header
        print(f"  {'':>18}", end="")
        for b_label in labels:
            print(f"  {short[b_label]:>12}", end="")
        print(f"\n  {'─'*18}{'─'*(13*n)}")

        # Daid-aware csum Spearman ρ matrix
        print(f"\n  {'Daid-aware csum ρ':>18}")
        for a_label in labels:
            print(f"  {short[a_label]:>12}", end="")
            for b_label in labels:
                if a_label == b_label:
                    print(f"  {'───':>12}", end="")
                    continue
                a_scores = runs[a_label][qi]
                b_scores = runs[b_label][qi]
                rho, nc = daid_aware_rho(
                    a_scores["daids"],
                    a_scores["csum"],
                    b_scores["daids"],
                    b_scores["csum"],
                )
                print(f"  {_fmt_rho(rho, nc):>12}", end="")
            print()

        # Top-1 / Top-3 / Top-5 daid overlap
        for k in (1, 3, 5):
            print(f"\n  {'Top-'+str(k)+' overlap':>18}")
            for a_label in labels:
                print(f"  {short[a_label]:>12}", end="")
                for b_label in labels:
                    if a_label == b_label:
                        print(f"  {'───':>12}", end="")
                        continue
                    a_scores = runs[a_label][qi]
                    b_scores = runs[b_label][qi]
                    ov = top_k_overlap(a_scores["daids"], b_scores["daids"], k=k)
                    print(f"  {ov*100:>8.0f}%  ", end="")
                print()

        # Positional csum Spearman ρ
        print(f"\n  {'Positional csum ρ':>18}")
        for a_label in labels:
            print(f"  {short[a_label]:>12}", end="")
            for b_label in labels:
                if a_label == b_label:
                    print(f"  {'───':>12}", end="")
                    continue
                a_scores = runs[a_label][qi]
                b_scores = runs[b_label][qi]
                rho, nc = positional_rho(a_scores["csum"], b_scores["csum"])
                print(f"  {_fmt_rho(rho, nc):>12}", end="")
            print()

        # Daid Jaccard
        print(f"\n  {'Daid Jaccard':>18}")
        for a_label in labels:
            print(f"  {short[a_label]:>12}", end="")
            for b_label in labels:
                if a_label == b_label:
                    print(f"  {'───':>12}", end="")
                    continue
                a_scores = runs[a_label][qi]
                b_scores = runs[b_label][qi]
                jac = daid_jaccard(a_scores["daids"], b_scores["daids"])
                print(f"  {jac:.4f}  ", end="")
            print()

    # Overall summary (average across queries)
    print(f"\n\n{'='*80}")
    print("OVERALL SUMMARY (avg across 3 queries)")
    print(f"{'='*80}\n")

    print(f"  {'Daid-aware csum ρ':>30}", end="")
    for b_label in labels:
        print(f"  {short[b_label]:>12}", end="")
    print()

    for a_label in labels:
        print(f"  {short[a_label]:>30}", end="")
        for b_label in labels:
            if a_label == b_label:
                print(f"     ─────  ", end="")
                continue
            avg_rho = 0
            for qi in range(3):
                a_scores = runs[a_label][qi]
                b_scores = runs[b_label][qi]
                rho, _ = daid_aware_rho(
                    a_scores["daids"],
                    a_scores["csum"],
                    b_scores["daids"],
                    b_scores["csum"],
                )
                avg_rho += rho / 3
            print(f"  {avg_rho:+.3f}   ", end="")
        print()

    for k in (1, 3, 5):
        print(f"\n  {'Top-'+str(k)+' annot overlap':>30}", end="")
        for b_label in labels:
            print(f"  {short[b_label]:>12}", end="")
        print()

        for a_label in labels:
            print(f"  {short[a_label]:>30}", end="")
            for b_label in labels:
                if a_label == b_label:
                    print(f"     ────   ", end="")
                    continue
                avg_ov = 0
                for qi in range(3):
                    a_scores = runs[a_label][qi]
                    b_scores = runs[b_label][qi]
                    ov = top_k_overlap(a_scores["daids"], b_scores["daids"], k=k)
                    avg_ov += ov / 3
                print(f"  {avg_ov*100:>6.0f}%   ", end="")
            print()

    # Per-daid breakdown for one pair (first two non-nightly if possible)
    print(f"\n\n{'='*80}")
    print("PER-DAID CSUM BREAKDOWN (query 0, first pair)")
    print(f"{'='*80}\n")

    a_label = labels[0]
    b_label = labels[1] if len(labels) > 1 else a_label
    a_sc = runs[a_label][0]
    b_sc = runs[b_label][0]

    by_a = {
        int(d): float(c) for d, c in zip(a_sc["daids"], a_sc["csum"]) if np.isfinite(c)
    }
    by_b = {
        int(d): float(c) for d, c in zip(b_sc["daids"], b_sc["csum"]) if np.isfinite(c)
    }
    common = sorted(set(by_a) & set(by_b))

    print(
        f"  {'daid':>6}  {'A ('+short[a_label]+')':>12}  {'B ('+short[b_label]+')':>12}  {'Δ':>10}  {'Δ%':>8}"
    )
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*8}")
    for d in common[:20]:
        ca, cb = by_a[d], by_b[d]
        delta = ca - cb
        pct = delta / cb * 100 if cb != 0 else float("inf")
        print(f"  {d:>6}  {ca:>12.4f}  {cb:>12.4f}  {delta:>+10.4f}  {pct:>+7.1f}%")
    if len(common) > 20:
        print(f"  ... ({len(common) - 20} more)")

    rho_daid, _ = daid_aware_rho(
        a_sc["daids"], a_sc["csum"], b_sc["daids"], b_sc["csum"]
    )
    rho_pos, _ = positional_rho(a_sc["csum"], b_sc["csum"])
    print(f"\n  Daid-aware ρ = {rho_daid:+.4f}")
    print(f"  Positional  ρ = {rho_pos:+.4f}")
    print(f"  Common daids = {len(common)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
