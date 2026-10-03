"""Modal fan-out for the DWLS policy calibration study (decisions/05).

One container per design cell runs `run_experiment.R --production --cell ID`
into a shared Volume; `combine` row-binds the cells' raw rows and runs the
study's own summaries and gates, so the frozen CSVs match a single-process
run (cells are seed-stable; see run_experiment.R).

    modal run app.py --mode smoke --run-id smoke-modal
    modal run app.py --mode production --run-id production-YYYY-MM-DD
    modal volume get magmaan-dwls-policy <run-id>/final ./<run-id>
"""
from pathlib import Path
import modal

STUDY = "experiments/decisions/05-dwls-policy-calibration"
N_CELLS = 146  # nrow(dwls_cells()); combine.R checks it

REPO = Path(__file__).resolve().parents[4] if modal.is_local() else Path("/repo")

image = (
    modal.Image.from_registry("rocker/r-ver:4.5.1", add_python="3.11")
    .run_commands("Rscript -e 'install.packages(c(\"Rcpp\", \"RcppEigen\", \"nloptr\", \"lavaan\"))'")
    .add_local_dir(str(REPO / "r-package"),
                   "/repo/r-package", copy=True,
                   ignore=["**/*.o", "**/*.so", "**/*.a", "build-rdev/**", "src/Makevars"])
    .run_commands("MAKEFLAGS=-j4 R CMD INSTALL /repo/r-package",
                  "mkdir -p /repo/cpp && touch /repo/cpp/CMakeLists.txt")
    .add_local_dir(str(REPO / "experiments/_support"),
                   "/repo/experiments/_support")
    .add_local_dir(str(REPO / STUDY),
                   f"/repo/{STUDY}", ignore=["results/**", "*.html", "report_files/**"])
)

app = modal.App("magmaan-dwls-policy")
vol = modal.Volume.from_name("magmaan-dwls-policy", create_if_missing=True)
RUNNER = f"/repo/{STUDY}/run_experiment.R"


@app.function(image=image, volumes={"/vol": vol}, timeout=3 * 60 * 60,
              cpu=4.0, memory=4096, retries=1)
def run_cell(cell_id: int, run_id: str, mode: str):
    import os, subprocess
    out = f"/vol/{run_id}/cells/cell_{cell_id:03d}"
    if os.path.exists(os.path.join(out, "raw.rds")):
        return cell_id  # already done (resumed launch)
    if os.path.exists(out):
        subprocess.run(["rm", "-rf", out], check=True)  # partial attempt
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(["Rscript", RUNNER, f"--{mode}", "--cell", str(cell_id),
                    "--workers", "4", "--out-dir", out], check=True, env=env)
    vol.commit()
    return cell_id


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60, cpu=2.0, memory=8192)
def combine(run_id: str, mode: str, git_head: str):
    import subprocess
    vol.reload()
    subprocess.run(["Rscript", f"/repo/{STUDY}/modal/combine.R", "--run-dir", f"/vol/{run_id}",
                    "--mode", mode, "--git-head", git_head], check=True)
    vol.commit()
    return f"/vol/{run_id}/final"


@app.local_entrypoint()
def main(mode: str = "smoke", run_id: str = "smoke-modal"):
    import subprocess
    if mode not in ("smoke", "pilot", "production", "explore", "confirm"):
        raise SystemExit("mode must be smoke, pilot, production, explore or confirm")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    ids = list(range(1, 53)) + list(range(127, 139)) if mode in ("explore", "confirm") else list(range(1, N_CELLS + 1))
    done = list(run_cell.starmap([(i, run_id, mode) for i in ids]))
    print(f"cells: {len(done)}/{len(ids)}")
    print(combine.remote(run_id, mode, head))
