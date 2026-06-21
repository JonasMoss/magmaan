"""
Survival-curve smoke for the nested GLS profile-Hessian spectrum.

The plot compares the finite-sample distribution of

    2N { f_B(S) - f_A(S) }

against two central asymptotic references:

  * profile-Hessian weighted chi-square mixture;
  * ordinary chi-square using the exact-fit U spectrum.

No Monte Carlo is used for the reference laws. The empirical curves are the
finite-sample simulation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import tempfile

import numpy as np

import gls_nested_profile_mc as smoke
import gls_nested_profile_scan as scan


def covariance_direction(base, p: int, i: int, j: int) -> np.ndarray:
    E = np.zeros((p, p))
    E[i, j] = E[j, i] = 1.0
    return base.vech(E, p)


def model_jacobians(base, p: int, model: str):
    if model == "cov-equality":
        return base.jacobians(p)
    if model == "one-cov":
        Delta_B = base.vech(np.eye(p), p)[:, None]
        Delta_A = np.column_stack([Delta_B[:, 0], covariance_direction(base, p, 1, 0)])
        return Delta_A, Delta_B
    raise ValueError(model)


def sigma0_from_args(p: int, eps: float, pat: str, diag: str | None) -> np.ndarray:
    if diag is None:
        return np.eye(p) + eps * np.diag(scan.pattern(pat, p))
    vals = np.array([float(x) for x in diag.split(",") if x], dtype=float)
    if vals.size != p:
        raise ValueError(f"--diag has {vals.size} entries but --p is {p}")
    return np.diag(vals)


def spectrum_setup(p: int, eps: float, pat: str, diag: str | None, model: str):
    base = smoke.load_base()
    Dp = base.dup(p)
    Dplus = np.linalg.inv(Dp.T @ Dp) @ Dp.T
    Delta_A, Delta_B = model_jacobians(base, p, model)
    df_diff = Delta_A.shape[1] - Delta_B.shape[1]

    Sig0 = sigma0_from_args(p, eps, pat, diag)
    u0 = base.vech(Sig0, p)
    Gamma = base.gamma_nt(Sig0, Dplus)
    W = np.linalg.inv(Gamma)

    Q_profile = base.score_hessian(Delta_B, u0, p, Dplus) - base.score_hessian(
        Delta_A, u0, p, Dplus
    )
    Q_classic = base.fixed_weight_q(Delta_B, W) - base.fixed_weight_q(Delta_A, W)

    lam_profile, _ = base.spectrum(Q_profile, Gamma)
    lam_classic, _ = base.spectrum(Q_classic, Gamma)
    full_lam = smoke.positive_lambdas(lam_profile)
    classic_lam = smoke.positive_lambdas(lam_classic)
    chi_lam = np.ones(df_diff)
    return base, Sig0, Delta_A, Delta_B, Dplus, full_lam, classic_lam, chi_lam


def mixture_survival_grid(lam: np.ndarray, xs: np.ndarray) -> np.ndarray:
    pos = smoke.positive_lambdas(lam)
    if pos.size == 0:
        return np.zeros_like(xs)
    args = [
        str(smoke.mixture_cli()),
        "upper_grid",
        f"{xs[0]:.17g}",
        f"{xs[-1]:.17g}",
        str(xs.size),
    ]
    args.extend(f"{x:.17g}" for x in pos)
    out = subprocess.check_output(args, text=True)
    vals = []
    for line in out.splitlines():
        vals.append(float(line.split()[1]))
    return np.array(vals, dtype=float)


def empirical_survival_grid(stats: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.array([np.mean(stats > x) for x in xs], dtype=float)


def write_csv(path: Path, xs: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        names = ["x", *columns.keys()]
        writer.writerow(names)
        for i, x in enumerate(xs):
            writer.writerow(
                [f"{x:.12g}", *[f"{columns[name][i]:.12g}" for name in columns]]
            )


def write_plot(csv_path: Path, pdf_path: Path, title: str, df_diff: int) -> None:
    r_code = r'''
args <- commandArgs(TRUE)
csv_path <- args[[1]]
pdf_path <- args[[2]]
title <- args[[3]]
df_diff <- as.integer(args[[4]])
d <- read.csv(csv_path, check.names = FALSE)
emp_cols <- grep("^emp_N", names(d), value = TRUE)
cols <- c(profile = "#111111", chi = "#777777",
          "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")
pdf(pdf_path, width = 7.2, height = 4.8, useDingbats = FALSE)
par(mar = c(4.2, 4.4, 2.6, 1.0))
plot(d$x, d$profile, type = "l", log = "y", ylim = c(1e-3, 1),
     xlab = "test statistic t", ylab = "survival Pr(T > t)",
     main = title, lwd = 2.2, col = cols[["profile"]])
grid(nx = NA, ny = NULL, col = "gray88", lty = "dotted")
lines(d$x, d$chi, lwd = 2.0, col = cols[["chi"]], lty = 2)
for (i in seq_along(emp_cols)) {
  lines(d$x, d[[emp_cols[[i]]]], lwd = 1.45, col = cols[[2 + i]])
}
abline(h = 0.05, col = "gray75", lty = 3)
chi_label <- bquote(chi[.(df_diff)]^2)
legend_labels <- c("profile mixture", as.expression(chi_label),
                   sub("^emp_N", "empirical N=", emp_cols))
legend_cols <- c(cols[["profile"]], cols[["chi"]], cols[seq_len(length(emp_cols)) + 2])
legend_lty <- c(1, 2, rep(1, length(emp_cols)))
legend("topright", legend = legend_labels, col = legend_cols, lty = legend_lty,
       lwd = c(2.2, 2.0, rep(1.45, length(emp_cols))), bg = "white",
       box.col = "gray80", cex = 0.82)
dev.off()
'''
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(r_code)
        r_path = Path(fh.name)
    try:
        subprocess.run(
            ["Rscript", str(r_path), str(csv_path), str(pdf_path), title, str(df_diff)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    finally:
        r_path.unlink(missing_ok=True)


def run(
    p: int,
    eps: float,
    pat: str,
    diag: str | None,
    model: str,
    Ns: tuple[int, ...],
    M: int,
    grid_n: int,
    out_prefix: str,
) -> None:
    base, Sig0, Delta_A, Delta_B, Dplus, full_lam, classic_lam, chi_lam = (
        spectrum_setup(p, eps, pat, diag, model)
    )
    df_diff = Delta_A.shape[1] - Delta_B.shape[1]
    q995 = max(
        smoke.weighted_chisq_quantile(full_lam, 0.995),
        smoke.weighted_chisq_quantile(chi_lam, 0.995),
    )
    xs = np.linspace(0.0, 1.04 * q995, grid_n)

    columns: dict[str, np.ndarray] = {
        "profile": mixture_survival_grid(full_lam, xs),
        "classic": mixture_survival_grid(classic_lam, xs),
        "chi": mixture_survival_grid(chi_lam, xs),
    }
    profile_q95 = smoke.weighted_chisq_quantile(full_lam, 0.95)
    chi_q95 = smoke.weighted_chisq_quantile(chi_lam, 0.95)

    design = f"diag={diag}" if diag is not None else f"pattern={pat}, eps={eps}"
    print(f"survival smoke: p={p}, model={model}, {design}, Ns={list(Ns)}, M={M}")
    print(f"profile lambdas={full_lam}")
    print(f"profile q95={profile_q95:.4f}, chi q95={chi_q95:.4f}")
    print("N        tail_prof95  tail_chi95  supdiff_profile  supdiff_chi")
    for idx, N in enumerate(Ns):
        stats = smoke.simulate_stats(
            base, Sig0, Delta_A, Delta_B, p, Dplus, N, M, seed=20260630 + idx
        )
        emp = empirical_survival_grid(stats, xs)
        columns[f"emp_N{N}"] = emp
        sup_profile = np.max(np.abs(emp - columns["profile"]))
        sup_chi = np.max(np.abs(emp - columns["chi"]))
        print(
            f"{N:5d}       {np.mean(stats > profile_q95):8.4f}"
            f"    {np.mean(stats > chi_q95):8.4f}"
            f"        {sup_profile:8.4f}     {sup_chi:8.4f}"
        )

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / f"{out_prefix}.csv"
    pdf_path = out_dir / f"{out_prefix}.pdf"
    write_csv(csv_path, xs, columns)
    write_plot(csv_path, pdf_path, f"Nested GLS survival: {model}, {design}", df_diff)
    print(f"wrote {csv_path}")
    print(f"wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--pattern", type=str, default="alternating")
    parser.add_argument("--diag", type=str, default=None)
    parser.add_argument("--model", type=str, default="cov-equality")
    parser.add_argument("--Ns", type=str, default="150,600,3000")
    parser.add_argument("--M", type=int, default=12000)
    parser.add_argument("--grid", type=int, default=220)
    parser.add_argument("--out-prefix", type=str, default="gls_nested_profile_survival")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Ns = tuple(int(x) for x in args.Ns.split(",") if x)
    run(
        args.p,
        args.eps,
        args.pattern,
        args.diag,
        args.model,
        Ns,
        args.M,
        args.grid,
        args.out_prefix,
    )
