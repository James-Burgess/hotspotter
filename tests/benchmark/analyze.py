#!/usr/bin/env python3
"""Analyze benchmark results and compare against replay fixture ground truth."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

from compare import (
    compare_results,
    load_fixture_images,
    load_fixture_scores,
    print_detailed_report,
    _spearmanr,
)

# ---------------------------------------------------------------------------
# Replay fixture → sidecar analysis
# ---------------------------------------------------------------------------


def _sidecar_identify(
    sidecar_url: str,
    query_image_b64: str,
    query_bbox: list[float],
    query_species: str,
    database: list[dict],
    timeout: int = 300,
) -> dict:
    """Send a single identification request to the wbia-core sidecar."""
    body = {
        "query_image_b64": query_image_b64,
        "query_bbox": query_bbox,
        "query_theta": 0.0,
        "query_species": query_species,
        "database": database,
        "config": {
            "pipeline_root": "vsmany",
            "K": 4,
            "Knorm": 1,
            "Kpad": 0,
            "fg_on": False,
            "sv_on": False,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{sidecar_url}/api/v1/identify/",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read())
    return raw


def _to_native(val):
    """Recursively convert numpy types to native Python types."""
    import numpy as np

    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return [_to_native(v) for v in val.tolist()]
    if isinstance(val, list):
        return [_to_native(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, bytes):
        return val.decode("utf-8") if val else ""
    return val


def _build_db_from_fixture(fixture_data: dict) -> list[dict]:
    """Build the database request entries from fixture data."""
    db = []
    for entry in fixture_data["database"]:
        db.append(
            {
                "aid": _to_native(entry["aid"]),
                "image_b64": base64.b64encode(entry["image"]).decode("utf-8"),
                "bbox": _to_native(entry["bbox"]),
                "theta": 0.0,
                "name_uuid": None,
                "species": _to_native(entry["species"]),
            }
        )
    return db


def analyze_fixture_sidecar(
    fixture_path: str | Path,
    sidecar_url: str = "http://localhost:5000",
) -> dict:
    """Run a replay fixture through the wbia-core sidecar and compare.

    Returns comparison metrics between sidecar output and fixture WBIA scores.
    """
    fixture_path = Path(fixture_path)
    fixture_data = load_fixture_images(fixture_path)
    fixture_scores = load_fixture_scores(fixture_path)

    query_b64 = base64.b64encode(fixture_data["query_image"]).decode("utf-8")
    query_bbox = _to_native(fixture_data["query_bbox"])
    query_species = _to_native(fixture_data["species"])
    database = _build_db_from_fixture(fixture_data)

    print(f"  Sending {fixture_path.name} to sidecar...", end=" ", flush=True)
    t0 = time.monotonic()
    try:
        raw = _sidecar_identify(
            sidecar_url,
            query_b64,
            query_bbox,
            query_species,
            database,
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return {"fixture": fixture_path.name, "error": str(exc)}
    elapsed = time.monotonic() - t0
    print(f"done ({elapsed:.1f}s)")

    if raw.get("status") != "completed":
        return {"fixture": fixture_path.name, "error": raw.get("message", "unknown")}

    sidecar_scores = raw.get("response", {}).get("annot_scores", [])
    sidecar_by_aid = {s["aid"]: s["score"] for s in sidecar_scores}

    # Align by rank position (order in annot_scores list = score-desc)
    sidecar_top5_aids = [s["aid"] for s in sidecar_scores[:5]]
    wbia_top5_aids = fixture_scores["wbia_top5"]

    # Top-5 overlap
    top5_overlap = len(set(sidecar_top5_aids) & set(wbia_top5_aids))

    # Spearman on common annotations
    common_aids = sorted(set(sidecar_by_aid) & set(fixture_scores["wbia_scores"]))
    if len(common_aids) >= 3:
        sidecar_vals = [sidecar_by_aid[aid] for aid in common_aids]
        wbia_vals = [fixture_scores["wbia_scores"][aid] for aid in common_aids]
        rho = _spearmanr(sidecar_vals, wbia_vals)
    else:
        rho = None

    result = {
        "fixture": fixture_path.name,
        "species": fixture_data["species"],
        "n_database": len(database),
        "sidecar_top5": sidecar_top5_aids,
        "fixture_wbia_top5": wbia_top5_aids,
        "top5_overlap": top5_overlap,
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "timing_s": round(elapsed, 1),
    }
    return result


def compare_fixtures_with_sidecar(
    fixture_dir: str | Path,
    sidecar_url: str = "http://localhost:5000",
) -> list[dict]:
    """Run all fixtures in a directory through the sidecar and report."""
    fixture_dir = Path(fixture_dir)
    npz_files = sorted(fixture_dir.glob("*.npz"))
    results = []
    for fp in npz_files:
        r = analyze_fixture_sidecar(fp, sidecar_url)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Sidecar → WBIA fixture cross-check (requires WBIA containers running)
# ---------------------------------------------------------------------------


def _post_json(url: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _unwrap(val):
    if isinstance(val, dict):
        return val.get("__UUID__", str(val))
    return str(val)


def _wbia_healthy(wbia_url: str, timeout: int = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = _post_json(f"{wbia_url}/api/test/heartbeat/", {})
            if resp.get("response") is True:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _wbia_identify(
    wbia_url: str,
    query_image_b64: str,
    query_bbox: list[float],
    query_species: str,
    database: list[dict],
    image_dir: Path,
    timeout: int = 300,
) -> dict:
    """Run fixture through full WBIA multi-step REST flow.

    This mirrors the WbiaTargetRunner logic but returns raw result + metadata.
    """
    import socket
    import socketserver
    import tempfile
    from http.server import SimpleHTTPRequestHandler
    from threading import Thread

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(image_dir), **kwargs)

        def log_message(self, fmt, *args):
            pass

    server = socketserver.TCPServer(("0.0.0.0", 0), _Handler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    img_port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    host_alias = "host.docker.internal"

    # Write images
    filenames = []
    query_bytes = base64.b64decode(query_image_b64)
    qf = "q.png"
    (image_dir / qf).write_bytes(query_bytes)
    filenames.append(qf)

    for i, db_entry in enumerate(database):
        img_bytes = base64.b64decode(db_entry["image_b64"])
        df = f"db_{i}.png"
        (image_dir / df).write_bytes(img_bytes)
        filenames.append(df)

    try:
        # Upload images
        uris = [f"http://{host_alias}:{img_port}/{fn}" for fn in filenames]
        image_uuids = _post_json(
            f"{wbia_url}/api/image/json/", {"image_uri_list": uris}
        )
        image_uuid_strs = [_unwrap(u) for u in image_uuids]

        # Create annotations
        bboxes = [query_bbox] + [e["bbox"] for e in database]
        annot_uuids = _post_json(
            f"{wbia_url}/api/annot/json/",
            {
                "image_uuid_list": [{"__UUID__": u} for u in image_uuid_strs],
                "annot_bbox_list": bboxes,
                "annot_theta_list": [0.0] * len(filenames),
                "annot_species_list": [query_species] * len(filenames),
            },
        )
        annot_uuid_strs = [_unwrap(u) for u in annot_uuids]

        # Query
        query_uuid = annot_uuid_strs[0]
        db_uuids = list(annot_uuid_strs[1:])

        jobid = _post_json(
            f"{wbia_url}/api/engine/query/graph/",
            {
                "query_annot_uuid_list": [{"__UUID__": query_uuid}],
                "database_annot_uuid_list": [{"__UUID__": u} for u in db_uuids],
                "query_config_dict": {
                    "pipeline": "vsmany",
                    "pipeline_root": "vsmany",
                },
            },
        )

        # Poll
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = _post_json(
                f"{wbia_url}/api/engine/job/result/",
                {"jobid": jobid},
            )
            if result.get("response", {}).get("status") == "completed":
                return result["response"]
            time.sleep(2)

        raise TimeoutError("WBIA job not complete")
    finally:
        server.shutdown()


def compare_fixture_with_wbia(
    fixture_path: str | Path,
    wbia_url: str,
    image_dir: Path,
) -> dict:
    """Run fixture through a WBIA container and compare against ground truth."""
    from compare import load_fixture_scores

    fixture_data = load_fixture_images(fixture_path)
    fixture_scores = load_fixture_scores(fixture_path)

    query_b64 = base64.b64encode(fixture_data["query_image"]).decode("utf-8")
    database_entries = _build_db_from_fixture(fixture_data)

    wbia_result = _wbia_identify(
        wbia_url,
        query_b64,
        fixture_data["query_bbox"],
        fixture_data["species"],
        database_entries,
        image_dir,
    )

    # Parse WBIA result
    json_result = wbia_result.get("json_result", wbia_result)
    cm_dict = json_result.get("cm_dict", {})
    parsed_scores = {}
    if cm_dict:
        data = next(iter(cm_dict.values()))
        duuid_list = data.get("dannot_uuid_list", [])
        score_list = data.get("annot_score_list", [])
        for du, sc in zip(duuid_list, score_list):
            uid = du.get("__UUID__", str(du)) if isinstance(du, dict) else str(du)
            try:
                parsed_scores[uid] = float(sc)
            except (ValueError, TypeError):
                pass

    wbia_run_top5 = sorted(parsed_scores, key=parsed_scores.get, reverse=True)[:5]
    fixture_top5 = fixture_scores["wbia_top5"]

    # Common UUIDs for Spearman
    common_uids = sorted(set(parsed_scores) & set(fixture_scores["wbia_scores"]))
    if len(common_uids) >= 3:
        run_vals = [parsed_scores[uid] for uid in common_uids]
        fx_vals = [fixture_scores["wbia_scores"][uid] for uid in common_uids]
        rho = _spearmanr(run_vals, fx_vals)
    else:
        rho = None

    top5_overlap = len(set(wbia_run_top5) & set(fixture_top5))

    return {
        "fixture": fixture_path.name,
        "species": fixture_data["species"],
        "wbia_run_top5": wbia_run_top5,
        "fixture_top5": fixture_top5,
        "top5_overlap": top5_overlap,
        "spearman_rho": round(rho, 4) if rho is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and compare against replay fixtures."
    )
    sub = parser.add_subparsers(dest="command")

    # Report benchmark results
    p_report = sub.add_parser("report", help="Print detailed benchmark report")
    p_report.add_argument("results_dir", type=str, help="Benchmark results directory")
    p_report.add_argument(
        "--json", action="store_true", help="Output JSON instead of text"
    )

    # Compare fixtures with sidecar
    p_fix = sub.add_parser("fixtures", help="Run replay fixtures through sidecar")
    p_fix.add_argument(
        "fixture_dir",
        type=str,
        default="tests/replay/testdata/fixtures",
        nargs="?",
        help="Replay fixture directory",
    )
    p_fix.add_argument("--sidecar-url", type=str, default="http://localhost:5000")

    # Compare benchmark against fixture scores
    p_check = sub.add_parser("check", help="Cross-check benchmark with fixture scores")
    p_check.add_argument(
        "fixture_dir",
        type=str,
        default="tests/replay/testdata/fixtures",
        nargs="?",
        help="Replay fixture directory",
    )
    p_check.add_argument("results_dir", type=str, help="Benchmark results directory")

    args = parser.parse_args()

    if args.command == "report":
        if args.json:
            summary = compare_results(args.results_dir)
            print(json.dumps(summary, indent=2))
        else:
            print_detailed_report(args.results_dir)

    elif args.command == "fixtures":
        results = compare_fixtures_with_sidecar(args.fixture_dir, args.sidecar_url)
        n_total = len(results)
        n_ok = sum(
            1
            for r in results
            if r.get("error") is None and r.get("top5_overlap", 0) > 0
        )
        print(f"\n=== Fixture vs Sidecar Results ===")
        print(f"Fixtures: {n_total}, with overlap: {n_ok}")
        print()
        for r in results:
            err = r.get("error")
            if err:
                print(f"  {r['fixture']}: ERROR {err}")
            else:
                overlap_desc = f"{r['top5_overlap']}/5 top-5 overlap"
                rho_str = (
                    f"ρ={r['spearman_rho']}"
                    if r["spearman_rho"] is not None
                    else "ρ=N/A"
                )
                print(f"  {r['fixture']}: {overlap_desc}, {rho_str}, {r['timing_s']}s")
        print()

    elif args.command == "check":
        fixture_dir = Path(args.fixture_dir)
        results_dir = Path(args.results_dir)

        # Load all fixtures and extract their scores
        npz_files = sorted(fixture_dir.glob("*.npz"))
        print(f"Loading {len(npz_files)} fixtures ...")
        fixture_scores = {}
        for fp in npz_files:
            fs = load_fixture_scores(fp)
            fixture_scores[fp.name] = fs

        # Load benchmark results
        print(f"Loading benchmark results from {results_dir} ...")
        bench = compare_results(results_dir)
        per_query = {q["query_index"]: q for q in bench.get("per_query", [])}

        # Cross-reference: fixture species → COCO species
        species_map = {
            "zebra_grevys": "zebra_plains",
            "giraffe_reticulated": "giraffe_masai",
        }

        print(f"\n=== Fixture vs Benchmark ===\n")
        for fx_name, fx in sorted(fixture_scores.items()):
            print(f"  {fx_name}: {fx['species']}, {fx['n_database']} db annots")
            wbia_top5 = fx["wbia_top5"]
            print(f"    fixture WBIA top-5: {wbia_top5}")

            # Compare against per-query benchmark targets
            for qi, qdata in per_query.items():
                for name, top1_aid in qdata.get("top1_aids", {}).items():
                    _ = top1_aid  # placeholder - can't directly compare UUIDs

        print()


if __name__ == "__main__":
    main()
