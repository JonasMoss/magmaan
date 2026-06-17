# Installing magmaan on Saga (Sigma2)

magmaan's R package (`r-package/`) is **self-contained**: the C++ core plus the
vendored PORT and QUADPACK routines are compiled from sources under
`r-package/src/`. The one external numeric dependency is **NLopt**, and you do
**not** need a system NLopt module — the package picks it up from the **`nloptr`
CRAN package**, which bundles and self-builds NLopt. So on a compute node you
just need a C++23 toolchain, R, and `nloptr` installed.

## 1. Load a toolchain

magmaan uses `std::expected`, so it needs **g++ >= 13**; `CXX_STD = CXX23` needs
**R >= 4.3**.

```sh
module avail GCC R          # find the exact versions available
module load <foss/GCC toolchain with g++ >= 13>
module load <R >= 4.3>
```

## 2. Install NLopt via nloptr (no system module needed)

```sh
Rscript -e 'install.packages("nloptr", repos="https://cloud.r-project.org")'
```

`nloptr` finds a system NLopt if one is available, otherwise it builds the
bundled NLopt from source (via CMake) into its own package. magmaan then links
that automatically: `nlopt.h` comes through `LinkingTo: nloptr`, and the
`nlopt_*` symbols from `nloptr.so` are linked by full path + rpath. (If you
*do* have a system NLopt — an `nlopt.pc` on `PKG_CONFIG_PATH`, e.g. from a
`module load NLopt` — that is preferred automatically.)

## 3. Get the package onto Saga

The repo is private, so either copy the working tree or clone with a PAT. Only
`r-package/` plus `dev/saga/Makevars` are needed, but the whole tree is handy:

```sh
rsync -a --exclude='build' --exclude='build-*' --exclude='.git' \
    ./magmaan/ saga:~/magmaan/
```

## 4. Install with the tuned native flags

Run this **inside a Slurm job on the partition you will run jobs on**, so
`-march=native` matches the compute node (see the caveat in `dev/saga/Makevars`):

```sh
cd ~/magmaan
R_MAKEVARS_USER=$PWD/dev/saga/Makevars \
    MAKEFLAGS="-j$SLURM_CPUS_PER_TASK" \
    R CMD INSTALL r-package
```

Or, equivalently, copy `dev/saga/Makevars` to `~/.R/Makevars` and run a plain
`R CMD INSTALL r-package` / `devtools::install_local("r-package")`.

## Troubleshooting

- **`there is no package called 'nloptr'`**: install it first (step 2). It is an
  `Imports` + `LinkingTo` dependency.
- **`nlopt.h: No such file` / `cannot find -lnlopt`**: `nloptr` is not installed,
  or you want to force a specific NLopt. Pass it explicitly:
  `NLOPT_CFLAGS="-I<nlopt>/include" NLOPT_LIBS="-L<nlopt>/lib -lnlopt" R CMD INSTALL r-package`.
- **`std::expected` not found / C++23 errors**: the loaded g++ is < 13. Load a
  newer compiler module.
- **Illegal instruction at runtime**: the build node's microarch was newer than
  the run node's. Rebuild on the run partition, or use `-march=x86-64-v3`.
