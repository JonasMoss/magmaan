# Installing magmaan on Saga (Sigma2)

magmaan's R package (`r-package/`) is **self-contained**: the C++ core plus the
vendored PORT and QUADPACK routines are compiled from sources under
`r-package/src/`, linking a **system NLopt**. No CMake, no prebuilt library. So
on an HPC node you just need a C++23 toolchain, R, and NLopt.

## 1. Load a toolchain

magmaan uses `std::expected`, so it needs **g++ >= 13**; `CXX_STD = CXX23` needs
**R >= 4.3**. NLopt is a link dependency.

```sh
module avail GCC R NLopt        # find the exact versions available
module load <foss/GCC toolchain with g++ >= 13>
module load <R >= 4.3>
module load <NLopt>
```

The NLopt module should put `nlopt.pc` on `PKG_CONFIG_PATH` (EasyBuild modules
do). Check with `pkg-config --modversion nlopt`.

## 2. Get the package onto Saga

The repo is private, so either copy the working tree or clone with a PAT. Only
`r-package/` plus `dev/saga/Makevars` are needed, but the whole tree is handy:

```sh
rsync -a --exclude='build' --exclude='build-*' --exclude='.git' \
    ./magmaan/ saga:~/magmaan/
```

## 3. Install with the tuned native flags

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

- **`nlopt.h: No such file` / `cannot find -lnlopt`**: the NLopt module isn't
  loaded or doesn't expose pkg-config. Either load it, or pass the paths
  explicitly:
  `NLOPT_CFLAGS="-I<nlopt>/include" NLOPT_LIBS="-L<nlopt>/lib -lnlopt" R CMD INSTALL r-package`.
- **`std::expected` not found / C++23 errors**: the loaded g++ is < 13. Load a
  newer compiler module.
- **Illegal instruction at runtime**: the build node's microarch was newer than
  the run node's. Rebuild on the run partition, or use `-march=x86-64-v3`.
