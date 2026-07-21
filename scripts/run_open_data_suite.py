#!/usr/bin/env python3
"""Run every staged open dataset end to end, then validate the results.

This is the unattended driver. It walks the staged datasets under a root, picks the
right runner for each, and records what happened -- so a run that takes hours can be
left alone and inspected afterwards, and a single dataset blowing up does not cost the
other twenty.

Which runner, and why it is not one runner:

* CTC datasets are TIME SERIES. They go to ``run_volumetric_timelapse_barcode.py``,
  which assembles all timepoints onto one shared grid before analysing each. That
  matters for the barcode: cropping each timepoint to its own mask box would give every
  "fraction of volume" column a different denominator, so the columns would drift with
  the crop rather than with the specimen.
* Allen FOVs are INDEPENDENT SINGLE VOLUMES (OME SizeT=1). They go to
  ``run_volumetric_batch.py``, one row per field. Handing them to the time-lapse runner
  would invent a series out of unrelated fields and report differences between
  different cells as if they were dynamics.

Each dataset is run twice where masks exist: once with the published segmentation and
once without. The pair is the measurement of how much BARCODE's own thresholding can be
trusted on data where nobody has provided a mask -- which is most real data.

    python scripts/run_open_data_suite.py --root L:/FF/Hackathon/full_datasets
    python scripts/run_open_data_suite.py --root ... --flow --only ctc_Fluo-N3DH-CHO_01

Resumable: a dataset whose outputs already exist is skipped unless ``--force``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_open_data import discover, Staged

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


@dataclass
class Job:
    name: str
    argv: List[str]
    expect: str                    # a path that exists afterwards, for resume
    log: str = ""
    seconds: float = 0.0
    status: str = "pending"
    tail: List[str] = field(default_factory=list)


def timelapse_jobs(entry: Staged, flow: bool) -> List[Job]:
    script = os.path.join(HERE, "run_volumetric_timelapse_barcode.py")
    results = entry.results_dir
    jobs: List[Job] = []

    common = ["--component-stats"]
    if entry.dt_s:
        common += ["--frame-interval", str(entry.dt_s)]
    if flow:
        common += ["--flow"]

    if os.path.isdir(entry.mask_dir) and os.listdir(entry.mask_dir):
        jobs.append(Job(
            name=f"{entry.name}: masked",
            argv=[PYTHON, script, entry.data_dir, "--seg-root", entry.mask_dir] + common,
            expect=os.path.join(results, "timepoints_with_masks"),
        ))
    jobs.append(Job(
        name=f"{entry.name}: threshold only",
        argv=[PYTHON, script, entry.data_dir] + common,
        expect=os.path.join(results, "timepoints_no_masks"),
    ))
    return jobs


def batch_jobs(entry: Staged, flow: bool) -> List[Job]:
    script = os.path.join(HERE, "run_volumetric_batch.py")
    results = entry.results_dir
    jobs: List[Job] = []

    # xyzt on a single volume: the change and flow columns come back NaN, which is the
    # honest report for one timepoint, and every static 3D metric is still measured.
    # xyz would instead give per-slice 2D metrics, whose "Area" columns cannot be
    # compared against a volume -- so validation needs this mode, not that one.
    common = ["--mode", "xyzt", "--component-stats", "--packing", "--mask-intensity"]

    if os.path.isdir(entry.mask_dir) and os.listdir(entry.mask_dir):
        jobs.append(Job(
            name=f"{entry.name}: masked",
            argv=[PYTHON, script, entry.data_dir, "--seg-root", entry.mask_dir]
                 + common + ["--csv", os.path.join(results, "masked", "Summary.csv")],
            expect=os.path.join(results, "masked", "Summary.csv"),
        ))
    jobs.append(Job(
        name=f"{entry.name}: threshold only",
        argv=[PYTHON, script, entry.data_dir, "--mode", "xyzt", "--component-stats",
              "--csv", os.path.join(results, "threshold", "Summary.csv")],
        expect=os.path.join(results, "threshold", "Summary.csv"),
    ))
    return jobs


def run(job: Job, log_dir: str, timeout: float) -> Job:
    os.makedirs(log_dir, exist_ok=True)
    safe = job.name.replace(":", "").replace(" ", "_").replace("/", "_")
    job.log = os.path.join(log_dir, safe + ".log")
    started = time.time()

    parent = os.path.dirname(job.expect)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(job.log, "w", encoding="utf-8") as handle:
        handle.write(" ".join(job.argv) + "\n\n")
        handle.flush()
        try:
            completed = subprocess.run(
                job.argv, stdout=handle, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
            job.status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
        except subprocess.TimeoutExpired:
            job.status = f"timeout after {timeout:.0f}s"
        except Exception as error:
            job.status = f"{type(error).__name__}: {error}"

    job.seconds = time.time() - started
    try:
        with open(job.log, encoding="utf-8", errors="replace") as handle:
            job.tail = handle.read().splitlines()[-6:]
    except OSError:
        pass
    return job


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and validate every staged open dataset.")
    parser.add_argument("--root", default=r"L:\FF\Hackathon\full_datasets")
    parser.add_argument("--only", action="append", default=[],
                        help="staged folder name (repeatable); default is all of them")
    parser.add_argument("--flow", action="store_true",
                        help="also run the 3D optical flow branch. Much slower, and the "
                             "only way to fill the Speed columns -- which are the ones "
                             "the tracking ground truth can check")
    parser.add_argument("--force", action="store_true",
                        help="re-run datasets whose outputs already exist")
    parser.add_argument("--timeout", type=float, default=10800,
                        help="per-job timeout in seconds (default 3h)")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--include-other", action="store_true",
                        help="also run staged folders that this project did not create "
                             "(the lab's own datasets). Off by default -- they have "
                             "their own provenance and must not be overwritten")
    args = parser.parse_args()

    entries = discover(args.root)
    if args.only:
        entries = [e for e in entries if e.name in args.only]
    elif not args.include_other:
        # Only the folders this project staged. The same root holds the lab's own
        # datasets, and those have their own results folders, run parameters and
        # provenance -- an unattended sweep must not write into them.
        skipped = [e.name for e in entries if e.kind == "other"]
        entries = [e for e in entries if e.kind in ("ctc", "allen")]
        if skipped:
            print(f"not touching {len(skipped)} pre-existing dataset(s): "
                  f"{', '.join(skipped)}\n")
    if not entries:
        print(f"No staged datasets under {args.root}")
        return 1

    log_dir = os.path.join(args.root, "_suite_logs")
    jobs: List[Job] = []
    for entry in entries:
        maker = timelapse_jobs if entry.kind == "ctc" else batch_jobs
        jobs.extend(maker(entry, args.flow))

    print(f"{len(jobs)} job(s) over {len(entries)} dataset(s); logs in {log_dir}\n")
    done: List[Job] = []
    for n, job in enumerate(jobs, 1):
        if not args.force and os.path.exists(job.expect):
            job.status = "skipped (exists)"
            print(f"[{n}/{len(jobs)}] {job.name:<52} {job.status}")
            done.append(job)
            continue
        print(f"[{n}/{len(jobs)}] {job.name:<52} running...", flush=True)
        run(job, log_dir, args.timeout)
        print(f"[{n}/{len(jobs)}] {job.name:<52} {job.status} ({job.seconds:.0f}s)")
        if job.status != "ok":
            for line in job.tail:
                print(f"        | {line}")
        done.append(job)

    summary = os.path.join(log_dir, "suite_summary.json")
    os.makedirs(log_dir, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as handle:
        json.dump([{k: v for k, v in vars(j).items() if k != "tail"} for j in done],
                  handle, indent=2)

    ok = sum(1 for j in done if j.status in ("ok", "skipped (exists)"))
    print(f"\n{ok}/{len(done)} job(s) succeeded; summary {summary}")

    if not args.skip_validation:
        print("\n" + "=" * 78)
        subprocess.run([PYTHON, os.path.join(HERE, "validate_open_data.py"),
                        "--root", args.root], check=False)
    return 0 if ok == len(done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
