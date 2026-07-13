"""Budget-guarded Modal harness for experiment 64.

The default is a local dry run: it prints the fan-out and timeout-envelope cost
without starting a container. A substantive launch therefore needs an explicit
``--mode screen``, ``--mode focus``, or ``--mode all``. Put a USD 10 budget on a
dedicated Modal environment before launch; the in-code guard is an additional
estimate, not a substitute for Modal's account-side limit.

Examples from this directory:

    modal run app.py                         # dry run, no cloud spend
    modal run app.py --mode preflight --run-id july13
    modal run app.py --mode all --run-id july13

Pull combined results with:

    modal volume get exp64-flip-results july13 ./july13
"""

from pathlib import Path
import re
import modal

# Container path the experiment dir is mounted at (see the image build below).
# Must equal runtime EXP so the runner scripts resolve on Modal; the local EXP
# path only names the source for add_local_dir.
EXP_REMOTE = "/exp/experiments/64-flip-calibration-frontier"

if modal.is_local():
    HERE = Path(__file__).resolve()
    EXP = HERE.parents[1]
    EXPERIMENTS = EXP.parent
    MAGMAAN_RPKG = EXPERIMENTS.parent / "r-package"
    SUPPORT = EXPERIMENTS / "_support"
else:
    EXP = Path(EXP_REMOTE)
    EXPERIMENTS = Path("/exp/experiments")
    MAGMAAN_RPKG = Path("/build/magmaan")
    SUPPORT = Path("/exp/experiments/_support")

image = (
    modal.Image.from_registry("rocker/r-ver:4.5.1", add_python="3.11")
    .run_commands(
        "Rscript -e 'install.packages(c(\"Rcpp\",\"RcppEigen\",\"nloptr\",\"pkgload\"))'"
    )
    .add_local_dir(str(MAGMAAN_RPKG), "/build/magmaan", copy=True,
                   ignore=["**/*.o", "**/*.so", "**/*.a"])
    .run_commands("R CMD INSTALL /build/magmaan")
    .add_local_dir(str(SUPPORT), "/exp/experiments/_support", copy=True)
    .add_local_dir(str(EXP), EXP_REMOTE, copy=True,
                   ignore=["results/**", ".quarto/**", "report.html", "report_files/**"])
)

app = modal.App("exp64-flip-calibration-frontier")
vol = modal.Volume.from_name("exp64-flip-results", create_if_missing=True)

# One shard per design cell (128 screen, 32 focus). Even 1-cell shards keep every
# cell well under its timeout and avoid the cell_id-mod clustering that bundled
# all the heavy p=20 cells into one shard. Row-index sharding in the runner makes
# shard j == design row j.
SCREEN_SHARDS = 128
FOCUS_SHARDS = 32
SCREEN_TIMEOUT_H = 4
FOCUS_TIMEOUT_H = 8
CPU = 1.0
MEMORY_MIB = 2048
# Published Modal list rates used for the July 2026 planning calculation.
CPU_USD_PER_CORE_SECOND = 0.0000131
MEM_USD_PER_GIB_SECOND = 0.00000222
MAX_ESTIMATED_USD = 14.0

# Per-rep cost model fit to Modal timings (2026-07): a rep is one model fit plus
# `flips` cheap sign-flip re-evaluations of a per-rep influence matrix, so flip
# cost is ~independent of n and only the fit term scales with n. Measured at
# p=20, n_avg=50: 1.45 s/rep @199 flips and 5.29 s/rep @999 flips give
# flip20=0.0048 s, fit20(n50)=0.49 s; p=5 approximated at ~0.3x the p=20 terms.
FIT_N50 = {5: 0.15, 20: 0.49}       # seconds at n_avg=50
FLIP_SEC = {5: 0.0014, 20: 0.0048}  # seconds per flip
N_AVGS = {"screen": [50, 100, 200, 400], "focus": [50, 100]}
CELLS_PER_P_NAVG = {"screen": 16, "focus": 8}
DEFAULT_REPS = {"screen": 1000, "focus": 2000}
DEFAULT_FLIPS = {"screen": 199, "focus": 999}


def hourly_usd(cpu: float = CPU, memory_mib: int = MEMORY_MIB) -> float:
    return 3600 * (cpu * CPU_USD_PER_CORE_SECOND
                   + memory_mib / 1024 * MEM_USD_PER_GIB_SECOND)


def container_usd_per_second() -> float:
    return CPU * CPU_USD_PER_CORE_SECOND + MEMORY_MIB / 1024 * MEM_USD_PER_GIB_SECOND


def _profile_compute_seconds(profile: str, reps: int, flips: int) -> float:
    reps = reps or DEFAULT_REPS[profile]
    flips = flips or DEFAULT_FLIPS[profile]
    total = 0.0
    for p in (5, 20):
        for n_avg in N_AVGS[profile]:
            per_rep = FIT_N50[p] * (n_avg / 50.0) + flips * FLIP_SEC[p]
            total += CELLS_PER_P_NAVG[profile] * reps * per_rep
    return total


def expected_cost_usd(calibration: bool, screen: bool, focus: bool,
                      screen_reps: int = 0, screen_flips: int = 0,
                      focus_reps: int = 0, focus_flips: int = 0) -> float:
    """Realistic expected spend from the measured per-rep model (not worst case)."""
    per_sec = container_usd_per_second()
    total = 0.6 if calibration else 0.0  # ~40 min single-CPU power calibration
    if screen:
        total += _profile_compute_seconds("screen", screen_reps, screen_flips) * per_sec
    if focus:
        total += _profile_compute_seconds("focus", focus_reps, focus_flips) * per_sec
    return total


def timeout_envelope_usd(include_calibration: bool = True,
                         screen: bool = True, focus: bool = True) -> float:
    """Absolute worst case: every container runs to its full timeout (will not happen)."""
    hours = (2 if include_calibration else 0)
    hours += SCREEN_SHARDS * SCREEN_TIMEOUT_H if screen else 0
    hours += FOCUS_SHARDS * FOCUS_TIMEOUT_H if focus else 0
    hours += (1 if screen else 0) + (1 if focus else 0)
    return hours * hourly_usd()


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ValueError("run-id must be 1-64 safe path characters")


@app.function(image=image, volumes={"/vol": vol}, timeout=2 * 60 * 60,
              cpu=CPU, memory=MEMORY_MIB, retries=0)
def calibrate(run_id: str, smoke: bool = False) -> str:
    import subprocess
    output = f"/vol/{run_id}/calibration/power_calibration.csv"
    cmd = ["Rscript", str(EXP / "calibrate_power.R"), "--cores", "1",
           "--output", output]
    if smoke:
        cmd.append("--smoke")
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return output


def _slice_command(profile: str, shard: int, shards: int, run_id: str,
                   reps: int, flips: int, smoke: bool = False) -> list[str]:
    out = f"/vol/{run_id}/{profile}/slices/slice_{shard:03d}"
    cmd = ["Rscript", str(EXP / "run_experiment.R"), f"--{profile}",
           "--shard-index", str(shard), "--shard-count", str(shards),
           "--cores", "1", "--results-dir", out]
    if profile == "focus":
        cmd += ["--power-calibration-file",
                f"/vol/{run_id}/calibration/power_calibration.csv"]
    if reps:
        cmd += ["--reps", str(reps)]
    if flips:
        cmd += ["--flips", str(flips)]
    if smoke:
        cmd += ["--max-cells", "1", "--reps", "1", "--flips", "9"]
    return cmd


@app.function(image=image, volumes={"/vol": vol},
              timeout=SCREEN_TIMEOUT_H * 60 * 60,
              cpu=CPU, memory=MEMORY_MIB, retries=0)
def run_screen_slice(shard: int, run_id: str, reps: int = 0,
                     flips: int = 0, smoke: bool = False) -> int:
    import subprocess
    cmd = _slice_command("screen", shard, SCREEN_SHARDS, run_id,
                         reps, flips, smoke)
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return shard


@app.function(image=image, volumes={"/vol": vol},
              timeout=FOCUS_TIMEOUT_H * 60 * 60,
              cpu=CPU, memory=MEMORY_MIB, retries=0)
def run_focus_slice(shard: int, run_id: str, reps: int = 0,
                    flips: int = 0) -> int:
    import subprocess
    cmd = _slice_command("focus", shard, FOCUS_SHARDS, run_id, reps, flips)
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return shard


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60,
              cpu=CPU, memory=MEMORY_MIB, retries=0)
def combine(profile: str, run_id: str, expected_cells: int) -> str:
    import subprocess
    vol.reload()
    out = f"/vol/{run_id}/{profile}"
    subprocess.run(["Rscript", str(EXP / "modal" / "combine.R"),
                    "--out-dir", out, "--profile", profile,
                    "--expected-cells", str(expected_cells)], check=True)
    vol.commit()
    return out


def _run_phase(fn, args, shard_count, profile, run_id, expected_cells):
    """Fan a phase's slices out, tolerating per-slice failures.

    Returns the list of failed shard indices. Combine only runs when the phase
    is complete; a partial phase leaves its slices on the volume so the failed
    shards can be resumed and the combine re-run, rather than crashing the whole
    pipeline on one slow or broken cell.
    """
    print(f"DRIVER: launching {shard_count} {profile} slices...", flush=True)
    results = list(fn.starmap(args, return_exceptions=True))
    failed = [a[0] for a, r in zip(args, results) if isinstance(r, Exception)]
    for a, r in zip(args, results):
        if isinstance(r, Exception):
            print(f"DRIVER: {profile} shard {a[0]} FAILED: "
                  f"{type(r).__name__}: {r}", flush=True)
    print(f"DRIVER: {profile} slices ok {shard_count - len(failed)}/{shard_count}",
          flush=True)
    if failed:
        print(f"DRIVER: {profile} incomplete (shards {failed}); skipping combine. "
              f"Resume with --mode {profile} --skip-calibration, then it re-runs "
              f"only the unfinished shards and combines.", flush=True)
    else:
        print(f"DRIVER: {profile} combined:",
              combine.remote(profile, run_id, expected_cells), flush=True)
    return failed


@app.function(image=image, volumes={"/vol": vol}, timeout=24 * 60 * 60,
              cpu=0.25, memory=512, retries=0)
def drive(run_id: str, do_calibration: bool, do_screen: bool, do_focus: bool,
          screen_reps: int = 0, screen_flips: int = 0,
          focus_reps: int = 0, focus_flips: int = 0) -> dict:
    """Server-side orchestrator: run every phase from inside Modal.

    The local entrypoint spawns this and returns, so the multi-hour pipeline
    does not depend on the launching process staying alive. Slice functions fan
    out as separate containers; each phase blocks here until its slices finish.
    Screen and focus are independent (both only need calibration), so a failure
    in one does not block the other.
    """
    summary: dict = {}
    if do_calibration:
        print("DRIVER: calibrating power multiplier...", flush=True)
        calibrate.remote(run_id, False)
    if do_screen:
        args = [(j, run_id, screen_reps, screen_flips, False)
                for j in range(1, SCREEN_SHARDS + 1)]
        summary["screen_failed"] = _run_phase(
            run_screen_slice, args, SCREEN_SHARDS, "screen", run_id, 128)
    if do_focus:
        args = [(j, run_id, focus_reps, focus_flips)
                for j in range(1, FOCUS_SHARDS + 1)]
        summary["focus_failed"] = _run_phase(
            run_focus_slice, args, FOCUS_SHARDS, "focus", run_id, 32)
    print(f"DRIVER: done. summary={summary}. Pull: modal volume get "
          f"exp64-flip-results {run_id} ./{run_id}", flush=True)
    return summary


@app.local_entrypoint()
def main(mode: str = "dry-run", run_id: str = "full",
         screen_reps: int = 0, screen_flips: int = 0,
         focus_reps: int = 0, focus_flips: int = 0,
         skip_calibration: bool = False):
    validate_run_id(run_id)
    if mode not in {"dry-run", "preflight", "screen", "focus", "all"}:
        raise ValueError("mode must be dry-run, preflight, screen, focus, or all")
    wants_screen = mode in {"screen", "all", "dry-run"}
    wants_focus = mode in {"focus", "all", "dry-run"}
    wants_calibration = (mode in {"focus", "all", "dry-run"}
                         and not skip_calibration)
    if mode == "preflight":
        estimate = 4 * hourly_usd()
    else:
        estimate = expected_cost_usd(wants_calibration, wants_screen, wants_focus,
                                     screen_reps, screen_flips,
                                     focus_reps, focus_flips)
    worst = timeout_envelope_usd(wants_calibration, wants_screen, wants_focus)
    print(f"Expected cost: ${estimate:.2f}; guard: ${MAX_ESTIMATED_USD:.2f}")
    print(f"Absolute worst case (every container hits its timeout; will not "
          f"happen): ${worst:.2f}")
    print(f"Rate model: ${hourly_usd():.4f} per one-CPU/2-GiB container-hour")
    if estimate > MAX_ESTIMATED_USD:
        raise RuntimeError(
            f"expected cost ${estimate:.2f} exceeds the ${MAX_ESTIMATED_USD:.0f} guard")
    if mode == "dry-run":
        print(f"No containers launched. screen={SCREEN_SHARDS} slices; "
              f"focus={FOCUS_SHARDS} slices; retries=0.")
        return
    if mode == "preflight":
        path = calibrate.remote(run_id, True)
        print(f"Calibration preflight wrote {path}")
        done = run_screen_slice.remote(1, run_id, 1, 9, True)
        print(f"Screen preflight slice {done} complete; no substantive run launched.")
        return
    # Hand the whole pipeline to a server-side driver and return immediately, so
    # the run survives client disconnect. Launch with `modal run --detach` so the
    # ephemeral app persists after this entrypoint exits.
    handle = drive.spawn(run_id, wants_calibration, wants_screen, wants_focus,
                         screen_reps, screen_flips, focus_reps, focus_flips)
    print(f"Spawned server-side driver {handle.object_id} for run '{run_id}' "
          f"(calibration={wants_calibration}, screen={wants_screen}, "
          f"focus={wants_focus}).")
    print("Safe to disconnect. Watch: modal app logs <app-id>; "
          f"pull later: modal volume get exp64-flip-results {run_id} ./{run_id}")
