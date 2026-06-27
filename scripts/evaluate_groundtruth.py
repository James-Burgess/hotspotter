#!/usr/bin/env python3
"""Evaluate a Hotspotter or WBIA trace against individual-identity ground truth.

Reads ``final_scores`` from a trace directory and the ``batch.json`` produced by
``prepare_wildlife_dataset.py``, then ranks each query's results by the trace's
``score_list`` and measures identification accuracy against the single-label
``individual_ids`` ground truth:

  * **Top-1 / Top-k accuracy** — fraction of queries whose rank-k results contain
    a same-identity annotation.
  * **mAP** — mean average precision (standard reid metric); the single most
    informative number for "which pipeline identifies animals better".

This is the metric that supports a "better than WBIA" claim: run it on both
traces (same batch) and compare. Both ``default`` (hotspotter) and
``sv_on_true`` (WBIA) traces are supported.

Usage:
    python scripts/evaluate_groundtruth.py \\
        --trace ../artifacts/wbia-oracle/<hs-run>  --config-label default  --batch ../batches/atrw.json
    python scripts/evaluate_groundtruth.py \\
        --trace ../artifacts/wbia-oracle/<wb-run>  --config-label sv_on_true --batch ../batches/atrw.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_arr(row, col, run_dir):
    """Resolve a parquet cell (ndarray / json-ref / npy-sidecar) to an ndarray."""
    val = row[col]
    if isinstance(val, np.ndarray):
        return val
    parsed = json.loads(val) if isinstance(val, str) else val
    if isinstance(parsed, dict) and "npy_path" in parsed:
        npy = parsed["npy_path"]
        # All paths in parquet cells are absolute as written by the trace
        # writer's container.  Use only the filename and reconstruct the
        # path relative to *run_dir* (the evaluator's mount may differ).
        fname = Path(npy).name
        full = run_dir / "final_scores" / "arrays" / fname
        return np.load(str(full))
    if isinstance(parsed, dict) and "values" in parsed:
        return np.array(parsed["values"])
    return np.array(parsed)


def load_results(trace_dir: Path, config_label: str):
    """Return list of {qaid, daids, scores} ranked by score descending."""
    score_dir = trace_dir / "final_scores"
    pattern = f"{config_label}_*.parquet"
    files = sorted(score_dir.glob(pattern))
    if not files:
        # fall back to whatever is present
        files = sorted(score_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet traces in {score_dir} ({pattern})")

    out = []
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            daids = _load_arr(row, "daid_list_array", trace_dir).astype(np.int64)
            scores = _load_arr(row, "score_list_array", trace_dir).astype(np.float64)
            # rank by score desc; stable so ties keep daid order
            order = np.argsort(-scores, kind="stable")
            out.append(
                {
                    "qaid": int(row["qaid"]),
                    "daids": daids[order],
                    "scores": scores[order],
                }
            )
    return out


def build_identity_map(batch_path: Path):
    """annot_id -> identity token (single label; first individual_id)."""
    batch = json.loads(batch_path.read_text())
    idmap = {}
    queries = []
    for a in batch["annotations"]:
        aid = int(a["annot_id"])
        ids = a.get("individual_ids") or []
        idmap[aid] = ids[0] if ids else None
        if a.get("is_query"):
            queries.append(aid)
    return idmap, set(queries)


def _is_match(query_identity, cand_identity):
    return query_identity is not None and cand_identity == query_identity


def average_precision(daids, query_identity, idmap):
    """Standard reid AP: (1/R) * sum(precision@k for each relevant rank)."""
    relevant_total = 0
    hits = 0
    ap = 0.0
    for rank, d in enumerate(daids, start=1):
        cand = idmap.get(int(d))
        if cand is None:
            continue
        if cand == query_identity:
            relevant_total += 1
            hits += 1
            ap += hits / rank
    if relevant_total == 0:
        return float("nan")  # query has no same-identity gallery (shouldn't happen)
    return ap / relevant_total


def top_k_hit(daids, query_identity, idmap, k):
    for d in daids[:k]:
        cand = idmap.get(int(d))
        if cand == query_identity:
            return True
    return False


def evaluate(trace_dir: Path, batch_path: Path, config_label: str, ks=(1, 5)):
    idmap, query_set = build_identity_map(batch_path)
    results = load_results(trace_dir, config_label)

    per_query = []
    for r in results:
        qaid = r["qaid"]
        qi = idmap.get(qaid)
        if qi is None:
            continue
        daids = r["daids"]
        ap = average_precision(daids, qi, idmap)
        hits = {k: top_k_hit(daids, qi, idmap, k) for k in ks}
        # rank-1 daid + its identity (for the per-query table)
        rank1_daid = int(daids[0]) if len(daids) else -1
        rank1_id = idmap.get(rank1_daid)
        per_query.append(
            {
                "qaid": qaid,
                "identity": qi,
                "rank1": rank1_daid,
                "rank1_identity": rank1_id,
                "top1": hits[1],
                **{f"top{k}": hits[k] for k in ks if k != 1},
                "ap": ap,
            }
        )
    return per_query


def fmt_report(label: str, per_query, ks=(1, 5)):
    n = len(per_query)
    lines = [f"\n=== {label} ===  (n={n} queries)"]
    lines.append(
        f"{'qaid':>6} {'identity':>12} {'rank1':>6} {'rank1_id':>12}"
        f" {'top1':>5} {'top5':>5} {'AP':>6}"
    )
    lines.append("-" * 60)
    for q in per_query[:25]:
        lines.append(
            f"{q['qaid']:>6} {str(q['identity']):>12} {q['rank1']:>6} "
            f"{str(q['rank1_identity']):>12} {str(q['top1']):>5} "
            f"{str(q.get('top5', '')):>5} {q['ap']:>6.3f}"
        )
    if n > 25:
        lines.append(f"... ({n - 25} more)")

    top1 = np.mean([q["top1"] for q in per_query]) if n else float("nan")
    top5 = np.mean([q.get("top5", False) for q in per_query]) if n else float("nan")
    aps = [q["ap"] for q in per_query if not np.isnan(q["ap"])]
    mAP = np.mean(aps) if aps else float("nan")
    lines.append("-" * 60)
    lines.append(f"Top-1 accuracy: {top1*100:.1f}%")
    lines.append(f"Top-5 accuracy: {top5*100:.1f}%")
    lines.append(f"mAP:            {mAP:.4f}  (n={len(aps)})")
    return "\n".join(lines), {"top1": top1, "top5": top5, "mAP": mAP, "n": n}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--trace", required=True, help="trace run directory")
    p.add_argument(
        "--batch", required=True, help="batch.json with ground-truth identities"
    )
    p.add_argument(
        "--config-label",
        default=None,
        help="trace config label (default: 'default' for HS, 'sv_on_true' for WBIA)",
    )
    p.add_argument("--label", default=None, help="display label for the report")
    p.add_argument("--json", default=None, help="write metrics json here")
    args = p.parse_args(argv)

    trace_dir = Path(args.trace).resolve()
    batch_path = Path(args.batch).resolve()
    label = args.label or trace_dir.name

    # infer config label if not given
    cfg = args.config_label
    if cfg is None:
        cfg = (
            "sv_on_true"
            if (trace_dir / "final_scores").glob("sv_on_true_*.parquet")
            else "default"
        )

    per_query = evaluate(trace_dir, batch_path, cfg)
    if not per_query:
        print(f"ERROR: no evaluable queries in {trace_dir}", file=sys.stderr)
        return 1
    text, metrics = fmt_report(label, per_query)
    print(text)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"label": label, "trace": str(trace_dir), **metrics}, indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
