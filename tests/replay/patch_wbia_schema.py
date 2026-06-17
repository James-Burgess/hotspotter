"""Patch WBIA schema migration & fix job store dir creation.

Mounted into the WBIA container and run before the main process.
"""

import os
import pathlib
import sys

import wbia

wb_pkg = pathlib.Path(wbia.__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Create missing fix_annotation_orientation_issue module
# ---------------------------------------------------------------------------

missing_mod = wb_pkg / "scripts" / "fix_annotation_orientation_issue.py"
missing_mod.write_text("""\
def fix_annotation_orientation(ibs, is_tile=None):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("[fix_annotation_orientation] stub called (no-op)")
""")

# Ensure scripts/__init__.py exists
scripts_init = wb_pkg / "scripts" / "__init__.py"
if not scripts_init.exists():
    scripts_init.write_text("# patched\n")

# ---------------------------------------------------------------------------
# 2. Ensure engine_shelves directory exists (fixes JobStore crash)
# The cachedir is typically {dbdir}/_ibsdb/_ibeis_cache
# ---------------------------------------------------------------------------

for probe in ("/data/db", "/cache"):
    for base in (
        pathlib.Path(probe) / "_ibsdb" / "_ibeis_cache",
        pathlib.Path(probe),
    ):
        shelve_dir = base / "engine_shelves"
        shelve_dir.mkdir(parents=True, exist_ok=True)
        if (shelve_dir / "jobs.db").parent.exists():
            break

# ---------------------------------------------------------------------------
# 3. Monkey-patch queue_interrupted_jobs to handle missing jobcounter
# ---------------------------------------------------------------------------

import wbia.web.job_engine as _je

_original = _je.JobInterface.queue_interrupted_jobs


def _patched_queue_interrupted_jobs(jobiface):
    try:
        _original(jobiface)
    except KeyError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "[patch] queue_interrupted_jobs KeyError (%s) — skipped", exc
        )


_je.JobInterface.queue_interrupted_jobs = _patched_queue_interrupted_jobs
