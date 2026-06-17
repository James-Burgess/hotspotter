"""Cross-target result comparison and detailed analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _spearmanr(a: list[float], b: list[float]) -> float:
    """Pure-Python Spearman rank correlation (no scipy dependency)."""
    n = len(a)
    if n < 3:
        return 0.0
    rank_a = {v: i for i, v in enumerate(sorted(a, reverse=True))}
    rank_b = {v: i for i, v in enumerate(sorted(b, reverse=True))}
    d = sum((rank_a[ai] - rank_b[bi]) ** 2 for ai, bi in zip(a, b))
    return 1.0 - (6.0 * d) / (n * (n * n - 1.0))


def _top1_aids(
    results_dir: Path, target_names: list[str]
) -> dict[int, dict[str, str | None]]:
    """Build per-query map of target -> top-1 aid."""
    per_query: dict[int, dict[str, str | None]] = {}
    query_dirs_: set[int] = set()

    for name in target_names:
        target_dir = results_dir / f"target-{name}"
        if not target_dir.exists():
            continue
        for qdir in sorted(target_dir.glob("query_*/")):
            qnum = int(qdir.name.split("_")[1])
            query_dirs_.add(qnum)

    for qnum in sorted(query_dirs_):
        per_query[qnum] = {}
        for name in target_names:
            resp_path = (
                results_dir / f"target-{name}" / f"query_{qnum:03d}" / "response.json"
            )
            if not resp_path.exists():
                per_query[qnum][name] = None
                continue
            try:
                resp = json.loads(resp_path.read_text())
                scores = resp.get("response", {}).get("annot_scores", [])
                scores = sorted(scores, key=lambda x: x.get("score", 0), reverse=True)
                per_query[qnum][name] = scores[0]["aid"] if scores else None
            except Exception:
                per_query[qnum][name] = None

    return per_query


def _all_annot_scores(
    results_dir: Path, target_names: list[str]
) -> dict[int, dict[str, list[dict]]]:
    """Collect all annot_scores per query per target, sorted by score desc."""
    out: dict[int, dict[str, list[dict]]] = {}
    for name in target_names:
        target_dir = results_dir / f"target-{name}"
        if not target_dir.exists():
            continue
        for qdir in sorted(target_dir.glob("query_*/")):
            qnum = int(qdir.name.split("_")[1])
            if qnum not in out:
                out[qnum] = {}
            resp_path = qdir / "response.json"
            if not resp_path.exists():
                continue
            try:
                resp = json.loads(resp_path.read_text())
                scores = resp.get("response", {}).get("annot_scores", [])
                out[qnum][name] = sorted(
                    scores, key=lambda x: x.get("score", 0), reverse=True
                )
            except Exception:
                out[qnum][name] = []
    return out


def _get_target_names(results_dir: Path) -> list[str]:
    target_dirs = sorted(results_dir.glob("target-*/"))
    return [d.name[len("target-") :] for d in target_dirs]


# ---------------------------------------------------------------------------
# Detailed analysis helpers
# ---------------------------------------------------------------------------


def top_k_aids(results_dir: str | Path, k: int = 5) -> dict[int, dict[str, list[dict]]]:
    """For each query, return the top-K aids+scores for every target."""
    results_dir = Path(results_dir)
    target_names = _get_target_names(results_dir)
    all_scores = _all_annot_scores(results_dir, target_names)
    out: dict[int, dict[str, list[dict]]] = {}
    for qnum in sorted(all_scores):
        out[qnum] = {}
        for name in target_names:
            scores = all_scores.get(qnum, {}).get(name, [])
            out[qnum][name] = [
                {"aid": s["aid"], "score": round(s["score"], 4)} for s in scores[:k]
            ]
    return out


def top_k_overlap_matrix(
    results_dir: str | Path, k: int = 3
) -> dict[int, dict[str, dict[str, float]]]:
    """Per-query overlap: fraction of target A's top-K in target B's top-K."""
    results_dir = Path(results_dir)
    target_names = _get_target_names(results_dir)
    all_scores = _all_annot_scores(results_dir, target_names)
    out: dict[int, dict[str, dict[str, float]]] = {}

    for qnum in sorted(all_scores):
        out[qnum] = {}
        for a_name in target_names:
            out[qnum][a_name] = {}
            a_top = {s["aid"] for s in all_scores.get(qnum, {}).get(a_name, [])[:k]}
            for b_name in target_names:
                b_top = {s["aid"] for s in all_scores.get(qnum, {}).get(b_name, [])[:k]}
                overlap = len(a_top & b_top) / max(len(a_top), 1)
                out[qnum][a_name][b_name] = round(overlap, 4)
    return out


def score_distribution(
    results_dir: str | Path,
) -> dict[int, dict[str, dict[str, float | None]]]:
    """Per-query per-target score distribution stats."""
    results_dir = Path(results_dir)
    target_names = _get_target_names(results_dir)
    all_scores = _all_annot_scores(results_dir, target_names)
    out: dict[int, dict[str, dict[str, float | None]]] = {}

    for qnum in sorted(all_scores):
        out[qnum] = {}
        for name in target_names:
            vals = [s["score"] for s in all_scores.get(qnum, {}).get(name, [])]
            if not vals:
                out[qnum][name] = {"min": None, "max": None, "mean": None, "std": None}
            else:
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                out[qnum][name] = {
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                    "mean": round(mean, 4),
                    "std": round(variance**0.5, 4),
                    "n": len(vals),
                }
    return out


def aggregate_spearman(
    results_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Aggregate Spearman rho across all queries for each target pair."""
    results_dir = Path(results_dir)
    target_names = _get_target_names(results_dir)
    all_scores = _all_annot_scores(results_dir, target_names)

    pair_rhos: dict[tuple[str, str], list[float]] = {}

    for qnum in sorted(all_scores):
        names_present = [n for n in target_names if n in all_scores.get(qnum, {})]
        for i in range(len(names_present)):
            for j in range(i + 1, len(names_present)):
                a_name = names_present[i]
                b_name = names_present[j]
                a_map = {s["aid"]: s["score"] for s in all_scores[qnum].get(a_name, [])}
                b_map = {s["aid"]: s["score"] for s in all_scores[qnum].get(b_name, [])}
                common = sorted(set(a_map) & set(b_map))
                if len(common) >= 3:
                    a_vals = [a_map[aid] for aid in common]
                    b_vals = [b_map[aid] for aid in common]
                    rho = _spearmanr(a_vals, b_vals)
                    pair_rhos.setdefault((a_name, b_name), []).append(rho)

    out: dict[str, dict[str, Any]] = {}
    for (a, b), rhos in sorted(pair_rhos.items()):
        key = f"{a} vs {b}"
        out[key] = {
            "n_queries": len(rhos),
            "mean_rho": round(sum(rhos) / len(rhos), 4),
            "min_rho": round(min(rhos), 4),
            "max_rho": round(max(rhos), 4),
            "std_rho": round(
                (sum((r - sum(rhos) / len(rhos)) ** 2 for r in rhos) / len(rhos))
                ** 0.5,
                4,
            ),
            "all_rho": [round(r, 4) for r in rhos],
        }
    return out


# ---------------------------------------------------------------------------
# Replay fixture helpers
# ---------------------------------------------------------------------------


def load_fixture_scores(fixture_path: str | Path) -> dict[str, Any]:
    """Load a replay .npz fixture and extract ground-truth WBIA scores.

    Returns
    -------
    dict with keys:
        species, seed, query_idx, n_database,
        wbia_scores: {annot_uuid: score} sorted desc,
        wbia_top5: [annot_uuid, ...],
    """
    import numpy as np

    fixture_path = Path(fixture_path)
    fx = np.load(fixture_path, allow_pickle=True)

    species = str(
        fx.get("species", b"").item()
        if isinstance(fx.get("species"), bytes | bytearray)
        else fx.get("species", "")
    )
    seed = int(fx.get("seed", 0))
    query_idx = int(fx.get("query_idx", 0))
    annot_uuids = list(fx.get("annot_uuids", []))
    raw_result = (
        fx.get("raw_result", {}).item()
        if fx.get("raw_result", None) is not None
        else {}
    )

    # Parse cm_dict
    scores: dict[str, float] = {}
    json_result = raw_result.get("json_result", raw_result)
    cm_dict = json_result.get("cm_dict", {})
    if cm_dict:
        data = next(iter(cm_dict.values()))
        dannot_list = data.get("dannot_uuid_list", [])
        score_list = data.get("annot_score_list", [])
        for duuid, s in zip(dannot_list, score_list):
            uid = (
                duuid.get("__UUID__", str(duuid))
                if isinstance(duuid, dict)
                else str(duuid)
            )
            try:
                scores[uid] = float(s)
            except (ValueError, TypeError):
                pass

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "fixture": fixture_path.name,
        "species": species,
        "seed": seed,
        "query_idx": query_idx,
        "n_annotations": len(annot_uuids),
        "n_database": len(annot_uuids) - 1,
        "wbia_scores": dict(sorted_scores),
        "wbia_top5": [uid for uid, _ in sorted_scores[:5]],
    }


def load_fixture_images(fixture_path: str | Path) -> dict[str, Any]:
    """Load replay fixture and return images + metadata for running through targets.

    Returns
    -------
    dict with keys:
        species, query_idx,
        query_image (bytes), query_bbox,
        database: [{aid, image (bytes), bbox, species}, ...],
        wbia_result (full raw_result for comparison)
    """
    import cv2
    import numpy as np

    fixture_path = Path(fixture_path)
    fx = np.load(fixture_path, allow_pickle=True)

    annot_uuids = list(fx.get("annot_uuids", []))
    bboxes = list(fx.get("bboxes", []))
    image_bytes_list = list(fx.get("image_bytes", []))
    species_bytes = fx.get("species", b"")
    species = (
        species_bytes.item()
        if isinstance(species_bytes, bytes | bytearray)
        else str(species_bytes)
    )
    query_idx = int(fx.get("query_idx", 0))
    raw_result = fx.get("raw_result", None)
    if raw_result is not None:
        raw_result = raw_result.item()

    query_img_bytes = image_bytes_list[query_idx]
    query_bbox = list(bboxes[query_idx])

    # Extract annot UUIDs — they are stored as list of scalar UUID strings
    annot_uuid_strs = []
    for u in annot_uuids:
        if isinstance(u, bytes):
            annot_uuid_strs.append(u.decode("utf-8"))
        else:
            annot_uuid_strs.append(str(u))

    query_uuid = annot_uuid_strs[query_idx]

    database = []
    for i in range(len(annot_uuid_strs)):
        if i == query_idx:
            continue
        img_bytes = image_bytes_list[i]
        database.append(
            {
                "aid": annot_uuid_strs[i],
                "image": img_bytes,
                "bbox": list(bboxes[i]),
                "species": species,
            }
        )

    return {
        "fixture": fixture_path.name,
        "species": species,
        "query_idx": query_idx,
        "query_uuid": query_uuid,
        "query_image": query_img_bytes,
        "query_bbox": query_bbox,
        "database": database,
        "wbia_result": raw_result,
    }


# ---------------------------------------------------------------------------
# Main comparison entry point
# ---------------------------------------------------------------------------


def compare_results(results_dir: str | Path) -> dict:
    """Read all target result dirs, compute agreement metrics.

    Returns the summary dict.
    """
    results_dir = Path(results_dir)
    target_dirs = sorted(results_dir.glob("target-*/"))
    target_names = [d.name[len("target-") :] for d in target_dirs]

    config_path = results_dir / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())

    top1 = _top1_aids(results_dir, target_names)
    all_scores = _all_annot_scores(results_dir, target_names)

    global_top1_identical = True
    global_rankings_match = True
    global_max_score_delta = 0.0
    spearman_below_pairs: list[dict] = []

    per_query: list[dict] = []
    errors: list[dict] = []

    for qnum in sorted(all_scores.keys()):
        entry: dict[str, Any] = {
            "query_index": qnum,
            "top1_aids": top1.get(qnum, {}),
            "max_score_delta": 0.0,
            "spearman_pairs": [],
            "top3_overlap": {},
            "score_stats": {},
        }

        for name in target_names:
            resp_path = (
                results_dir / f"target-{name}" / f"query_{qnum:03d}" / "response.json"
            )
            if resp_path.exists():
                try:
                    resp = json.loads(resp_path.read_text())
                    if resp.get("error"):
                        errors.append(
                            {
                                "target": name,
                                "query_index": qnum,
                                "message": resp["error"],
                            }
                        )
                except Exception:
                    pass

        top1_aids = entry["top1_aids"]
        aids = [v for v in top1_aids.values() if v is not None]
        if len(set(aids)) > 1:
            global_top1_identical = False

        names_present = [n for n in target_names if n in all_scores.get(qnum, {})]
        if len(names_present) >= 2:
            aid_scores: dict[str, dict[str, float]] = {}
            for name in names_present:
                for item in all_scores[qnum].get(name, []):
                    aid = item["aid"]
                    if aid not in aid_scores:
                        aid_scores[aid] = {}
                    aid_scores[aid][name] = item["score"]

            for aid, scores_by_target in aid_scores.items():
                vals = list(scores_by_target.values())
                if len(vals) >= 2:
                    delta = max(vals) - min(vals)
                    if delta > global_max_score_delta:
                        global_max_score_delta = delta
                    if delta > entry["max_score_delta"]:
                        entry["max_score_delta"] = delta

            first_name = names_present[0]
            first_order = [item["aid"] for item in all_scores[qnum].get(first_name, [])]
            for name in names_present[1:]:
                order = [item["aid"] for item in all_scores[qnum].get(name, [])]
                if order != first_order:
                    global_rankings_match = False

            for i in range(len(names_present)):
                for j in range(i + 1, len(names_present)):
                    a_name = names_present[i]
                    b_name = names_present[j]
                    a_scores_map = {
                        item["aid"]: item["score"]
                        for item in all_scores[qnum].get(a_name, [])
                    }
                    b_scores_map = {
                        item["aid"]: item["score"]
                        for item in all_scores[qnum].get(b_name, [])
                    }
                    common_aids = sorted(set(a_scores_map) & set(b_scores_map))
                    if len(common_aids) >= 3:
                        a_vals = [a_scores_map[aid] for aid in common_aids]
                        b_vals = [b_scores_map[aid] for aid in common_aids]
                        rho = _spearmanr(a_vals, b_vals)
                    else:
                        rho = None

                    pair_entry = {"a": a_name, "b": b_name, "rho": rho}
                    entry["spearman_pairs"].append(pair_entry)

                    if rho is not None and rho < 1.0:
                        spearman_below_pairs.append(pair_entry)

            # Top-3 overlap
            for a_name in names_present:
                entry["top3_overlap"][a_name] = {}
                a_top3 = {s["aid"] for s in all_scores[qnum].get(a_name, [])[:3]}
                for b_name in names_present:
                    b_top3 = {s["aid"] for s in all_scores[qnum].get(b_name, [])[:3]}
                    overlap = len(a_top3 & b_top3) / max(len(a_top3), 1)
                    entry["top3_overlap"][a_name][b_name] = round(overlap, 4)

            # Score stats per target
            for name in names_present:
                vals = [s["score"] for s in all_scores[qnum].get(name, [])]
                if vals:
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean) ** 2 for v in vals) / len(vals)
                    entry["score_stats"][name] = {
                        "min": round(min(vals), 4),
                        "max": round(max(vals), 4),
                        "mean": round(mean, 4),
                        "std": round(var**0.5, 4),
                    }

        per_query.append(entry)

    summary: dict[str, Any] = {}
    run_id = results_dir.name.replace("test-run-results-", "")
    summary["run_id"] = run_id
    summary["config"] = config
    summary["targets"] = target_names
    summary["agreement"] = {
        "top1_identical": global_top1_identical,
        "all_rankings_match": global_rankings_match,
        "max_score_delta": global_max_score_delta,
        "spearman_below_pairs": spearman_below_pairs,
    }
    summary["per_query"] = per_query
    summary["errors"] = errors

    # Extra aggregate reports
    summary["top_k_aids"] = top_k_aids(results_dir, k=5)
    summary["top3_overall_overlap"] = _aggregate_overlap(results_dir)
    summary["aggregate_spearman"] = aggregate_spearman(results_dir)
    summary["score_distributions"] = score_distribution(results_dir)

    # Accuracy metrics (ground-truth individual_ids)
    accuracy = compute_accuracy(results_dir, target_names)
    summary["accuracy"] = accuracy

    return summary


def _aggregate_overlap(results_dir: Path) -> dict[str, float]:
    """Mean top-3 overlap across all queries for each target pair."""
    matrix = top_k_overlap_matrix(results_dir, k=3)
    pair_sums: dict[str, list[float]] = {}
    for qnum in matrix:
        for a_name in matrix[qnum]:
            for b_name, val in matrix[qnum][a_name].items():
                if a_name != b_name:
                    key = f"{a_name} -> {b_name}"
                    pair_sums.setdefault(key, []).append(val)
    out = {}
    for key, vals in sorted(pair_sums.items()):
        if vals:
            out[key] = round(sum(vals) / len(vals), 4)
    return out


# ---------------------------------------------------------------------------
# Accuracy metrics (ground-truth individual_ids)
# ---------------------------------------------------------------------------


def _load_ground_truth(
    results_dir: Path,
) -> dict[int, list[int]]:
    """Load annotations.json and build annot_id → individual_ids mapping."""
    annot_path = results_dir / "annotations.json"
    if not annot_path.exists():
        return {}
    annots = json.loads(annot_path.read_text())
    return {a["annot_id"]: a.get("individual_ids", []) for a in annots}


def _parse_aid(aid: str) -> int | None:
    if not aid or not aid.startswith("coco-annot-"):
        return None
    try:
        return int(aid.replace("coco-annot-", ""))
    except (ValueError, TypeError):
        return None


def compute_accuracy(
    results_dir: str | Path,
    target_names: list[str] | None = None,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute identification accuracy against ground-truth individual_ids.

    For each query + target, checks whether the top-ranked database annotation
    shares any ``individual_id`` with the query annotation.  Also computes MRR
    (Mean Reciprocal Rank) —  1 / rank of the first correct match.

    Returns
    -------
    dict with keys:
        per_target: {target_name: {top1, top3, top5, mrr, n_queries}}
        per_query: [{query_index, targets: {target_name: {top1_correct, mrr}}}]
    """
    results_dir = Path(results_dir)
    ground_truth = _load_ground_truth(results_dir)
    if not ground_truth:
        return {"error": "No ground truth found (annotations.json missing or empty)"}

    if target_names is None:
        target_dirs = sorted(results_dir.glob("target-*/"))
        target_names = [d.name[len("target-") :] for d in target_dirs]

    per_query: list[dict] = []
    per_target: dict[str, dict[str, Any]] = {}

    for name in target_names:
        per_target[name] = {
            "n_queries": 0,
            "n_correct": {k: 0 for k in k_values},
            "top1_correct": 0,
            "mrr_sum": 0.0,
        }

    all_scores = _all_annot_scores(results_dir, target_names)

    for qnum in sorted(all_scores.keys()):
        if qnum not in all_scores:
            continue
        entry: dict[str, Any] = {"query_index": qnum, "targets": {}}

        for name in target_names:
            scores = all_scores.get(qnum, {}).get(name, [])
            if not scores:
                continue

            per_target[name]["n_queries"] += 1

            # Parse the scored aids
            ranked_aids = [_parse_aid(s["aid"]) for s in scores]
            ranked_aids = [a for a in ranked_aids if a is not None]

            if not ranked_aids:
                continue

            # Query's individual_ids — look for the query annotation in ground truth
            # The query annotation is the one NOT in the database ranking (it's the
            # seeker), but it should have an entry in annotations.json with is_query=True
            query_ids: set[int] = set()
            for annot_id, indiv_ids in ground_truth.items():
                # The query annotation ID won't appear in the scored aids (it's not
                # in the database).  Find the query by checking annotations.json
                # for the first is_query entry matching the query index.
                pass

            # Rebuild: find query individual_ids from annotations.json
            annots = (
                json.loads((results_dir / "annotations.json").read_text())
                if (results_dir / "annotations.json").exists()
                else []
            )
            query_indiv: list[int] = []
            for a in annots:
                if a.get("is_query") and a.get("query_index") == qnum:
                    query_indiv = a.get("individual_ids", [])
                    break
            query_set = set(query_indiv)

            # Check top-k correctness
            first_correct_rank: int | None = None

            for rank, annot_id in enumerate(ranked_aids, start=1):
                db_ids = set(ground_truth.get(annot_id, []))
                is_correct = bool(query_set & db_ids)

                if is_correct:
                    if first_correct_rank is None:
                        first_correct_rank = rank

                    for k in k_values:
                        if rank <= k and name in per_target:
                            per_target[name]["n_correct"][k] += 1

            mrr = 1.0 / first_correct_rank if first_correct_rank else 0.0
            per_target[name]["mrr_sum"] += mrr

            if first_correct_rank is not None and first_correct_rank == 1:
                per_target[name]["top1_correct"] += 1

            entry["targets"][name] = {
                "top1_correct": first_correct_rank == 1,
                "mrr": round(mrr, 4),
                "rank": first_correct_rank,
            }

        per_query.append(entry)

    # Compute aggregate rates
    per_target_agg: dict[str, dict] = {}
    for name, stats in per_target.items():
        n = stats["n_queries"]
        per_target_agg[name] = {
            "top1_accuracy": round(stats["top1_correct"] / n, 4) if n else None,
            "mrr": round(stats["mrr_sum"] / n, 4) if n else None,
            "n_queries": n,
        }
        for k in k_values:
            per_target_agg[name][f"top{k}_accuracy"] = (
                round(stats["n_correct"][k] / n, 4) if n else None
            )

    return {
        "per_target": per_target_agg,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------


def print_detailed_report(results_dir: str | Path) -> None:
    """Print a human-readable analysis report."""
    results_dir = Path(results_dir)
    summary = compare_results(results_dir)
    ag = summary["agreement"]

    print(f"=== Benchmark Report: {summary['run_id']} ===\n")

    # Config
    cfg = summary.get("config", {})
    print(
        f"Config: {cfg.get('n_annots', '?')} annotations, "
        f"{len(summary.get('per_query', []))} queries, "
        f"seed={cfg.get('seed', '?')}, species={cfg.get('species', 'all')}"
    )
    print(f"Targets: {', '.join(summary['targets'])}")
    print(f"Errors: {len(summary['errors'])}\n")

    # Global agreement
    print(f"Global top-1 identical: {ag['top1_identical']}")
    print(f"Global rankings match: {ag['all_rankings_match']}")
    print(f"Global max score delta: {ag['max_score_delta']:.4f}")

    # Aggregate Spearman
    print("\n--- Aggregate Spearman ---")
    for pair_key, stats in summary.get("aggregate_spearman", {}).items():
        print(
            f"  {pair_key}: mean ρ={stats['mean_rho']:.4f} "
            f"(min={stats['min_rho']:.4f}, max={stats['max_rho']:.4f})"
        )

    # Accuracy
    acc = summary.get("accuracy", {})
    if acc.get("per_target"):
        print("\n--- Identification Accuracy (ground-truth individual_ids) ---")
        for name, stats in acc["per_target"].items():
            t1 = stats.get("top1_accuracy")
            t3 = stats.get("top3_accuracy")
            t5 = stats.get("top5_accuracy")
            mrr = stats.get("mrr")
            n = stats.get("n_queries", 0)
            parts = []
            if t1 is not None:
                parts.append(f"top-1={t1:.1%}")
            if t3 is not None:
                parts.append(f"top-3={t3:.1%}")
            if t5 is not None:
                parts.append(f"top-5={t5:.1%}")
            if mrr is not None:
                parts.append(f"MRR={mrr:.3f}")
            print(f"  {name}: {', '.join(parts)} (n={n})")

    # Top-3 overall overlap
    print("\n--- Mean Top-3 Overlap ---")
    for key, val in summary.get("top3_overall_overlap", {}).items():
        print(f"  {key}: {val:.3f}")

    # Per-query breakdown
    print("\n--- Per-Query Breakdown ---")
    for q in summary.get("per_query", []):
        qi = q["query_index"]
        print(f"\n  Query {qi}:")
        print(f"    Top-1: {q['top1_aids']}")
        print(f"    Max score delta: {q['max_score_delta']:.4f}")

        for pair in q.get("spearman_pairs", []):
            rho_str = f"{pair['rho']:.4f}" if pair["rho"] is not None else "N/A"
            print(f"    ρ({pair['a']}, {pair['b']}) = {rho_str}")

        # Top-3 aids per target
        tk = summary.get("top_k_aids", {}).get(qi, {})
        for name, items in tk.items():
            aids_str = ", ".join(f"{s['aid']}({s['score']})" for s in items)
            print(f"    {name} top-5: {aids_str}")

        # Score distribution
        sd = summary.get("score_distributions", {}).get(qi, {})
        for name, stats in sd.items():
            if stats.get("n"):
                print(
                    f"    {name} scores: μ={stats['mean']}, σ={stats['std']}, "
                    f"range=[{stats['min']}, {stats['max']}]"
                )

    # Errors
    if summary["errors"]:
        print("\n--- Errors ---")
        for e in summary["errors"]:
            print(f"  [{e['target']}] query {e['query_index']}: {e['message']}")

    print()
