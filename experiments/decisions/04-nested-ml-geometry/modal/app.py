"""Cell fan-out; launch production only after merger criteria registration."""
from pathlib import Path
import modal

STUDY = "experiments/decisions/04-nested-ml-geometry"
# Inside the container this file is /root/app.py; only the local side needs the
# repository path to assemble the image.
REPO = Path(__file__).resolve().parents[4] if modal.is_local() else Path("/repo")
image = (
    modal.Image.from_registry("rocker/r-ver:4.5.1", add_python="3.11")
    .run_commands("Rscript -e 'install.packages(c(\"Rcpp\",\"RcppEigen\",\"nloptr\",\"lavaan\"))'")
    .add_local_dir(str(REPO / "r-package"), "/repo/r-package", copy=True,
                   ignore=["**/*.o", "**/*.so", "**/*.a", "build-rdev/**", "src/Makevars"])
    .run_commands("MAKEFLAGS=-j2 R CMD INSTALL /repo/r-package",
                  "mkdir -p /repo/cpp && touch /repo/cpp/CMakeLists.txt")
    .add_local_dir(str(REPO / "r-magmaan"), "/repo/r-magmaan", copy=True,
                   ignore=["**/*.tar.gz", "**/*.Rcheck/**"])
    .run_commands("R CMD INSTALL /repo/r-magmaan")
    .add_local_dir(str(REPO / "experiments/_support"), "/repo/experiments/_support")
    .add_local_dir(str(REPO / STUDY), f"/repo/{STUDY}", ignore=["results/**", "*.html", "*.pdf", "report_files/**", ".quarto/**"])
)
app = modal.App("magmaan-nested-ml-geometry")
volume = modal.Volume.from_name("magmaan-nested-ml-geometry", create_if_missing=True)


@app.function(image=image, volumes={"/vol": volume}, cpu=2, memory=8192,
              timeout=12 * 60 * 60)
def run_cell(cell_id: int, lane: str, mode: str, run_id: str, git_head: str):
    import os
    import subprocess
    # The container has no git; the local entrypoint passes the commit.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MAGMAAN_GIT_HEAD=git_head)
    subprocess.run(["Rscript", f"/repo/{STUDY}/run_experiment.R", f"--{mode}",
                    "--lane", lane, "--cell", str(cell_id), "--workers", "2", "--out-dir",
                    f"/vol/{run_id}/cells/cell_{cell_id:03d}"], check=True, env=env)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, cpu=1, memory=8192, timeout=3600)
def combine(lane: str, mode: str, run_id: str, cells: str):
    import subprocess
    volume.reload()
    subprocess.run(["Rscript", f"/repo/{STUDY}/modal/combine.R", "--mode", mode,
                    "--lane", lane, "--run-dir", f"/vol/{run_id}", "--cell", cells], check=True)
    volume.commit()


@app.local_entrypoint()
def main(lane: str = "structured-mean", mode: str = "smoke", run_id: str = "", cells: str = ""):
    import re
    import subprocess
    import uuid
    if lane not in ("structured-mean", "nested-geometry"):
        raise ValueError("unknown lane")
    if mode not in ("smoke", "pilot", "production"):
        raise ValueError("mode must be smoke, pilot or production")
    # A fresh suffix prevents reuse even when the caller repeats a label.
    label = run_id or f"{lane}-{mode}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("invalid run ID")
    run_id = f"{label}-{uuid.uuid4().hex}"
    count = 72 if lane == "structured-mean" else 48
    ids = [int(x) for x in cells.split(",")] if cells else list(range(1, count + 1))
    if len(set(ids)) != len(ids) or any(x < 1 or x > count for x in ids):
        raise ValueError(f"unique cell IDs 1..{count} required")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    print(f"Run ID: {run_id}")
    list(run_cell.starmap([(i, lane, mode, run_id, head) for i in ids]))
    combine.remote(lane, mode, run_id, ",".join(map(str, ids)))
