"""Cell fan-out; launch production only after merger criteria registration."""
from pathlib import Path
import modal

STUDY = "experiments/decisions/06-ordinal-threshold-invariance"
REPO = Path(__file__).resolve().parents[4]
image = (
    modal.Image.from_registry("rocker/r-ver:4.5.1", add_python="3.11")
    .run_commands("Rscript -e 'install.packages(c(\"Rcpp\",\"RcppEigen\",\"nloptr\",\"lavaan\"))'")
    .add_local_dir(str(REPO / "r-package"), "/repo/r-package", copy=True,
                   ignore=["**/*.o", "**/*.so", "**/*.a", "build-rdev/**", "src/Makevars"])
    .run_commands("MAKEFLAGS=-j2 R CMD INSTALL /repo/r-package",
                  "mkdir -p /repo/cpp && touch /repo/cpp/CMakeLists.txt")
    .add_local_dir(str(REPO / "experiments/_support"), "/repo/experiments/_support")
    .add_local_dir(str(REPO / STUDY), f"/repo/{STUDY}", ignore=["results/**", "*.html"])
)
app = modal.App("magmaan-threshold-invariance")
volume = modal.Volume.from_name("magmaan-threshold-invariance", create_if_missing=True)


@app.function(image=image, volumes={"/vol": volume}, cpu=2, memory=8192,
              timeout=12 * 60 * 60)
def run_cell(cell_id: int, mode: str, run_id: str):
    import os
    import subprocess
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(["Rscript", f"/repo/{STUDY}/run_experiment.R", f"--{mode}",
                    "--cell", str(cell_id), "--workers", "2", "--out-dir",
                    f"/vol/{run_id}/cells/cell_{cell_id:03d}"], check=True, env=env)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, cpu=1, memory=8192, timeout=3600)
def combine(mode: str, run_id: str, cells: str):
    import subprocess
    volume.reload()
    subprocess.run(["Rscript", f"/repo/{STUDY}/modal/combine.R", "--mode", mode,
                    "--run-dir", f"/vol/{run_id}", "--cell", cells], check=True)
    volume.commit()


@app.local_entrypoint()
def main(mode: str = "smoke", run_id: str = "smoke-modal", cells: str = ""):
    if mode not in ("smoke", "pilot", "production"):
        raise ValueError("mode must be smoke, pilot or production")
    ids = [int(x) for x in cells.split(",")] if cells else list(range(1, 385))
    if len(set(ids)) != len(ids) or any(x < 1 or x > 384 for x in ids):
        raise ValueError("unique cell IDs 1..384 required")
    list(run_cell.starmap([(i, mode, run_id) for i in ids]))
    combine.remote(mode, run_id, ",".join(map(str, ids)))
