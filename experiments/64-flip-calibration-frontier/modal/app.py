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

SCREEN_SHARDS = 16
FOCUS_SHARDS = 8
CPU = 1.0
MEMORY_MIB = 2048
# Published Modal list rates used for the July 2026 planning calculation.
CPU_USD_PER_CORE_SECOND = 0.0000131
MEM_USD_PER_GIB_SECOND = 0.00000222
MAX_ESTIMATED_USD = 10.0


def hourly_usd(cpu: float = CPU, memory_mib: int = MEMORY_MIB) -> float:
    return 3600 * (cpu * CPU_USD_PER_CORE_SECOND
                   + memory_mib / 1024 * MEM_USD_PER_GIB_SECOND)


def timeout_envelope_usd(include_calibration: bool = True,
                         screen: bool = True, focus: bool = True) -> float:
    hours = (2 if include_calibration else 0)
    hours += SCREEN_SHARDS * 2 if screen else 0
    hours += FOCUS_SHARDS * 6 if focus else 0
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
    out = f"/vol/{run_id}/{profile}/slices/slice_{shard:02d}"
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


@app.function(image=image, volumes={"/vol": vol}, timeout=2 * 60 * 60,
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


@app.function(image=image, volumes={"/vol": vol}, timeout=6 * 60 * 60,
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


@app.function(image=image, volumes={"/vol": vol}, timeout=12 * 60 * 60,
              cpu=0.25, memory=512, retries=0)
def drive(run_id: str, do_calibration: bool, do_screen: bool, do_focus: bool,
          screen_reps: int = 0, screen_flips: int = 0,
          focus_reps: int = 0, focus_flips: int = 0) -> str:
    """Server-side orchestrator: run every phase from inside Modal.

    The local entrypoint spawns this and returns, so the multi-hour pipeline
    does not depend on the launching process staying alive. Slice functions fan
    out as separate containers; each phase blocks here until its slices finish.
    """
    if do_calibration:
        print("DRIVER: calibrating power multiplier...", flush=True)
        calibrate.remote(run_id, False)
    if do_screen:
        print("DRIVER: launching screen slices...", flush=True)
        args = [(j, run_id, screen_reps, screen_flips, False)
                for j in range(1, SCREEN_SHARDS + 1)]
        done = list(run_screen_slice.starmap(args))
        print(f"DRIVER: screen slices complete {len(done)}/{SCREEN_SHARDS}", flush=True)
        print("DRIVER: screen combined:", combine.remote("screen", run_id, 128),
              flush=True)
    if do_focus:
        print("DRIVER: launching focus slices...", flush=True)
        args = [(j, run_id, focus_reps, focus_flips)
                for j in range(1, FOCUS_SHARDS + 1)]
        done = list(run_focus_slice.starmap(args))
        print(f"DRIVER: focus slices complete {len(done)}/{FOCUS_SHARDS}", flush=True)
        print("DRIVER: focus combined:", combine.remote("focus", run_id, 32),
              flush=True)
    print(f"DRIVER: done. Pull: modal volume get exp64-flip-results {run_id} "
          f"./{run_id}", flush=True)
    return run_id


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
    estimate = (4 * hourly_usd() if mode == "preflight" else
                timeout_envelope_usd(wants_calibration, wants_screen, wants_focus))
    print(f"Estimated timeout envelope: ${estimate:.2f}; guard: ${MAX_ESTIMATED_USD:.2f}")
    print(f"Rate model: ${hourly_usd():.4f} per one-CPU/2-GiB container-hour")
    if estimate > MAX_ESTIMATED_USD:
        raise RuntimeError("planned timeout envelope exceeds the $10 guard")
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
