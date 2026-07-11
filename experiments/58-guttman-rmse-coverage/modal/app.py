"""Modal harness for experiment 58 (Guttman RMSE / coverage / admissibility).

Fans ONE container per (generator, factors, indicators, n, strength, rho) cell.
Each runs run_experiment.R over the remaining conditions (scale x population) at
that coordinate and writes its per-cell summary CSVs to a shared Volume under
cells/<slug>/. The runner seeds each design point from its coordinates
(cell_code, not a running counter), so a cell reproduces exactly its slice of a
single-process full sweep; combine.R then row-binds the cells into the final
result-shaped CSVs the report reads. Modal retries any cell that dies wholesale.

Usage (from this directory, with Modal creds in ~/.modal.toml):

    modal run app.py --reps 20 --run-id smoke     # cheap smoke: forces the image build
    modal run app.py --reps 1000 --run-id pilot
    modal run app.py --reps 2000                  # paper run, run-id "full"

Pull results (result-shaped CSVs land at the Volume root under <run-id>/):

    modal volume get exp58-guttman-results <run-id>/whole_joint_summary.csv ./
    modal volume get exp58-guttman-results <run-id> ./<run-id>            # everything

The image source-builds the vendored, self-contained magmaan r-package (NLopt via
the nloptr CRAN package); it is cached across runs until the sources change.
"""

from pathlib import Path

import modal

# These paths matter only when BUILDING the image (locally). The spec is still
# evaluated at import inside the container, so the else branch defines the same
# names (dummy remote paths) to avoid NameError.
if modal.is_local():
    HERE = Path(__file__).resolve()
    EXP = HERE.parents[1]                       # experiments/58-guttman-rmse-coverage/
    EXPERIMENTS = EXP.parent                    # experiments/
    MAGMAAN_RPKG = EXPERIMENTS.parent / "r-package"   # magmaan/r-package (vendored)
    SUPPORT = EXPERIMENTS / "_support"
else:
    EXP = Path("/exp/experiments/58-guttman-rmse-coverage")

# rocker/r-ver ships a pinned R with the Posit binary repo, so Rcpp/RcppEigen/
# nloptr/pkgload install as binaries. magmaan is compiled from source in-image
# (src/Makevars sets CXX_STD = CXX23 for std::expected), so the base MUST carry
# GCC 13+; a noble-based rocker (R 4.5.x) does, jammy (GCC 11) does not. magmaan
# resolves NLopt from the nloptr package (no system NLopt module needed). Host
# build artifacts are excluded so the in-image toolchain recompiles cleanly. The
# runner sources experiments/_support/R/helpers.R via a "../_support" relative
# path, so the repo's experiments/ layout is mirrored under /exp/experiments/.
image = (
    modal.Image.from_registry("rocker/r-ver:4.5.1", add_python="3.11")
    .run_commands(
        "Rscript -e 'install.packages(c(\"Rcpp\",\"RcppEigen\",\"nloptr\",\"pkgload\"))'"
    )
    .add_local_dir(str(MAGMAAN_RPKG), "/build/magmaan", copy=True,
                   ignore=["**/*.o", "**/*.so", "**/*.a"])
    .run_commands("R CMD INSTALL /build/magmaan")
    .add_local_dir(str(SUPPORT), "/exp/experiments/_support", copy=True)
    .add_local_file(str(EXP / "run_experiment.R"),
                    "/exp/experiments/58-guttman-rmse-coverage/run_experiment.R")
    .add_local_file(str(EXP / "modal" / "combine.R"), "/exp/modal/combine.R")
)

app = modal.App("exp58-guttman-sim")
vol = modal.Volume.from_name("exp58-guttman-results", create_if_missing=True)

# Fan coordinates mirror the --full grid dimensions in run_experiment.R. Each
# container expands the remaining dims (scale x population regime) internally.
# `Rscript run_experiment.R --full --help` and condition_grid() are the truth.
GENERATORS = ["normal", "ig", "ordinal"]
FACTORS = [2, 3, 5]
INDICATORS = ["3", "5", "u"]
NS = [50, 100, 300, 800]
STRENGTH = ["moderate", "weak"]
RHO = [0.0, 0.35, 0.6, 0.8]


def cells():
    """Enumerate the fanned coordinates (864 at the --full defaults)."""
    for gen in GENERATORS:
        for fac in FACTORS:
            for ind in INDICATORS:
                for n in NS:
                    for strength in STRENGTH:
                        for rho in RHO:
                            yield (gen, fac, ind, n, strength, rho)


@app.function(image=image, volumes={"/vol": vol}, timeout=3 * 60 * 60,
              cpu=2.0, memory=2048, retries=3)
def run_cell(gen: str, fac: int, ind: str, n: int, strength: str, rho: float,
             run_id: str, reps: int) -> str:
    """One fanned coordinate: scale x population regime at these dims x reps."""
    import subprocess

    slug = f"{gen}_f{fac}_i{ind}_n{n}_{strength}_r{rho}"
    out_dir = f"/vol/{run_id}/cells/{slug}"
    cmd = ["Rscript", "/exp/experiments/58-guttman-rmse-coverage/run_experiment.R",
           "--full",
           "--generators", gen, "--factors", str(fac), "--indicators", ind,
           "--n", str(n), "--strength", strength, "--rho", str(rho),
           "--cores", "2", "--results-dir", out_dir]
    if reps:                               # reps=0 -> the --full default (50)
        cmd += ["--reps", str(reps)]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return slug


@app.function(image=image, volumes={"/vol": vol}, timeout=60 * 60,
              cpu=2.0, memory=8192)
def combine(run_id: str) -> str:
    """Reduce all cells into the final result-shaped CSVs at /vol/<run_id>/."""
    import subprocess

    vol.reload()
    out_dir = f"/vol/{run_id}"
    subprocess.run(["Rscript", "/exp/modal/combine.R", "--out-dir", out_dir], check=True)
    vol.commit()
    return out_dir


@app.local_entrypoint()
def main(reps: int = 0, run_id: str = "full"):
    args = [(*c, run_id, reps) for c in cells()]
    print(f"Launching {len(args)} cells, "
          f"reps={'--full default (50)' if reps == 0 else reps}, run_id={run_id}")
    done = list(run_cell.starmap(args))     # blocks until every cell finishes
    print(f"Cells complete: {len(done)}/{len(args)}")
    out_dir = combine.remote(run_id)
    print(f"Combined into volume exp58-guttman-results at {out_dir}.")
    print("Pull results with:")
    print(f"  modal volume get exp58-guttman-results {run_id}/whole_joint_summary.csv ./")
    print(f"  modal volume get exp58-guttman-results {run_id} ./{run_id}")
