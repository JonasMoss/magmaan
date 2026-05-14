# magmaan <img src="docs/figures/logo_compact.png" align="right" height="170" /></a>

> "This world ever was, and is, and shall be, an ever-living Fire."
> — Heraclitus

`magmaan` is the subterranean cousin of `lavaan`. It has a modular, simple design, and is made for methods researchers.

**Status:** Proto-type
**Language:** C++17
**Scope:** Non-Bayesian structural equation modeling with linear constraints
**Philosophy:** Make everything explicit and modular, don't prioritize user-friendliness, avoid object-oriented programing, don't change state, stay functional, work with composition, strict separation of concerns

## Roadmap

1. Complete data normal theory with robustness
2. Generalized least squars (AFD, "GLS", ULS)
3. C++17 without OO, structs only, no concepts, minimal abstractions, no exceptions
4. Threshold model
5. Incomplete data normal theory (FIML)
6. Variety of optimizers, RAM and LISREL representations

## In scope

1. Any kind of weird estimation of SEM models, such as t-distributed residuals or pairwise likelihood.
2. Any kind of weird 

## Not in scope

1. IRT models.
2. Other kinds of factor analysis such as EFA.