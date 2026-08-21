"""Modal harness for the experiment-77 FIML null/power continuation.

The default is a local dry run. A substantive launch requires ``--mode focus``,
``--mode stress``, or ``--mode all`` and should normally use ``--detach`` so the
server-side driver survives a disconnected client.

Examples from this directory:

    modal run app.py
    modal run app.py --mode preflight --run-id aug21
    modal run --detach app.py --mode all --run-id aug21

Pull results with:

    modal volume get exp77-gof-results aug21 ./aug21
"""

from pathlib import Path
import re
import modal

EXP_REMOTE = "/exp/experiments/77-fiml-global-gof-pilot"

if modal.is_local():
    HERE = Path(__file__).resolve()
    EXP = HERE.parents[1]
    EXPERIMENTS = EXP.parent
    MAGMAAN_RPKG = EXPERIMENTS.parent / "r-package"
    SUPPORT = EXPERIMENTS / "_support"
else:
    EXP = Path(EXP_REMOTE)
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
                   ignore=["results/**", ".quarto/**", "*.html"])
)

app = modal.App("exp77-fiml-gof-power")
vol = modal.Volume.from_name("exp77-gof-results", create_if_missing=True)

FOCUS_SHARDS = 165
STRESS_SHARDS = 60
TIMEOUT_H = 8
CPU = 1.0
MEMORY_MIB = 2048
CPU_USD_PER_CORE_SECOND = 0.0000131
MEM_USD_PER_GIB_SECOND = 0.00000222
MAX_ESTIMATED_USD = 15.0


def hourly_usd() -> float:
    return 3600 * (CPU * CPU_USD_PER_CORE_SECOND
                   + MEMORY_MIB / 1024 * MEMORY_USD_PER_GIB_SECOND)


def expected_cost_usd(focus: bool, stress: bool,
                      focus_reps: int, stress_reps: int,
                      flips: int) -> float:
    # Calibrated conservatively from the local paired null panel. The p=15
    # multiplier dominates; the three-corner sensitivity/weight ablation runs.
    flip_scale = flips / 199
    per_rep_199 = {5: .08, 6: .08, 10: .35, 12: .75, 15: 1.35}
    per_rep = {p: seconds * flip_scale for p, seconds in per_rep_199.items()}
    total_seconds = 60 * 20  # image-independent power calibration allowance
    if focus:
        # Eleven estimand-valid mechanisms and three truths per model.
        total_seconds += sum(11 * 3 * focus_reps * per_rep[p]
                             for p in (5, 6, 10, 12, 15))
    if stress:
        # Four nonnormal-MAR mechanisms and three truths per model.
        total_seconds += sum(4 * 3 * stress_reps * per_rep[p]
                             for p in (5, 6, 10, 12, 15))
    return total_seconds * (CPU_USD_PER_CORE_SECOND
                            + MEMORY_MIB / 1024 * MEM_USD_PER_GIB_SECOND)


def validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ValueError("run-id must be 1-64 safe path characters")


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60,
              cpu=CPU, memory=MEMORY_MIB, retries=0)
def calibrate(run_id: str) -> str:
    import subprocess
    output = f"/vol/{run_id}/calibration/power_calibration_n200.csv"
    subprocess.run([
        "Rscript", str(EXP / "calibrate_power.R"),
        "--n", "200", "--target", ".50", "--output", output
    ], check=True)
    vol.commit()
    return output


def slice_command(profile: str, shard: int, run_id: str,
                  reps: int, flips: int) -> list[str]:
    shards = FOCUS_SHARDS if profile == "focus" else STRESS_SHARDS
    region = "identified_null" if profile == "focus" else "nonnormal_mar_stress"
    output = f"/vol/{run_id}/{profile}/slices/slice_{shard:03d}"
    return [
        "Rscript", str(EXP / "run_sem_models.R"),
        "--reps", str(reps), "--n", "200", "--flips", str(flips),
        "--cores", "1", "--chunk-size", "25", "--estimators", "FIML",
        "--truth", "null,sparse,diffuse", "--regions", region,
        "--power-calibration-file",
        f"/vol/{run_id}/calibration/power_calibration_n200.csv",
        "--shard-index", str(shard), "--shard-count", str(shards),
        "--results-dir", output,
    ]


@app.function(image=image, volumes={"/vol": vol},
              timeout=TIMEOUT_H * 60 * 60,
              cpu=CPU, memory=MEMORY_MIB, retries=0)
def run_slice(profile: str, shard: int, run_id: str,
              reps: int, flips: int) -> int:
    import subprocess
    command = slice_command(profile, shard, run_id, reps, flips)
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    vol.commit()
    return shard


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60,
              cpu=CPU, memory=MEMORY_MIB, retries=0)
def combine(profile: str, run_id: str, expected_cells: int) -> str:
    import subprocess
    vol.reload()
    output = f"/vol/{run_id}/{profile}"
    subprocess.run([
        "Rscript", str(EXP / "modal" / "combine.R"),
        "--out-dir", output, "--profile", profile,
        "--expected-cells", str(expected_cells)
    ], check=True)
    vol.commit()
    return output


def run_phase(profile: str, run_id: str, reps: int, flips: int) -> list[int]:
    shards = FOCUS_SHARDS if profile == "focus" else STRESS_SHARDS
    arguments = [(profile, shard, run_id, reps, flips)
                 for shard in range(1, shards + 1)]
    results = list(run_slice.starmap(arguments, return_exceptions=True))
    failed = [args[1] for args, result in zip(arguments, results)
              if isinstance(result, Exception)]
    print(f"DRIVER: {profile} slices ok {shards - len(failed)}/{shards}",
          flush=True)
    if failed:
        print(f"DRIVER: {profile} incomplete; failed shards {failed}", flush=True)
    else:
        print("DRIVER: combined", combine.remote(profile, run_id, shards),
              flush=True)
    return failed


@app.function(image=image, volumes={"/vol": vol}, timeout=24 * 60 * 60,
              cpu=.25, memory=512, retries=0)
def drive(run_id: str, do_focus: bool, do_stress: bool,
          focus_reps: int, stress_reps: int, flips: int) -> dict:
    calibrate.remote(run_id)
    summary = {}
    if do_focus:
        summary["focus_failed"] = run_phase(
            "focus", run_id, focus_reps, flips)
    if do_stress:
        summary["stress_failed"] = run_phase(
            "stress", run_id, stress_reps, flips)
    print(f"DRIVER: done {summary}", flush=True)
    return summary


@app.local_entrypoint()
def main(mode: str = "dry-run", run_id: str = "full",
         focus_reps: int = 1000, stress_reps: int = 500,
         flips: int = 999):
    validate_run_id(run_id)
    if mode not in {"dry-run", "preflight", "focus", "stress", "all"}:
        raise ValueError("mode must be dry-run, preflight, focus, stress, or all")
    do_focus = mode in {"dry-run", "focus", "all"}
    do_stress = mode in {"dry-run", "stress", "all"}
    estimate = expected_cost_usd(
        do_focus, do_stress, focus_reps, stress_reps, flips)
    print(f"Expected compute cost: ${estimate:.2f}; guard ${MAX_ESTIMATED_USD:.2f}")
    print(f"One-CPU/2-GiB rate model: ${hourly_usd():.4f}/hour")
    if estimate > MAX_ESTIMATED_USD:
        raise RuntimeError("estimated cost exceeds the in-code guard")
    if mode == "dry-run":
        print(f"No containers launched. focus={FOCUS_SHARDS} cells, "
              f"stress={STRESS_SHARDS} cells, retries=0.")
        return
    if mode == "preflight":
        path = calibrate.remote(run_id)
        print("Calibration:", path)
        done = run_slice.remote("focus", 1, run_id, 2, 19)
        print("Preflight shard complete:", done)
        return
    handle = drive.spawn(run_id, do_focus, do_stress,
                         focus_reps, stress_reps, flips)
    print(f"Spawned server-side driver {handle.object_id} for '{run_id}'.")
    print("Launch with --detach for disconnect-safe execution. Pull later with "
          f"modal volume get exp77-gof-results {run_id} ./{run_id}")
