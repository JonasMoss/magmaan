# Iteration 04: 1,000-rep representative-SEM null run

## Design and runtime

The iteration-03 design was run unchanged at 1,000 replications per cell:
four SEMs, five native normal/VM/IG generators, complete versus 30% MCAR, and
`n = 120`. This is 40 null cells and 40,000 attempted FIML fits. It remains a
null-only comparison of ordinary LR, Yuan--Bentler/Mplus MLR, and FMG
corrections; the general latent-SEM multiplier statistic is not implemented.

The run took 541.1 seconds (9.0 minutes) on four workers, versus the 6.0-minute
projection from ten reps per cell. The discrepancy is mostly slow-tail load
imbalance: bifactor plus severe nonnormal MCAR cells took 0.15--0.35 seconds per
replication, while the compact complete-data cells were often below 0.02.
Future large runs should split or cost-balance cells instead of scheduling one
whole cell per worker. The observed run projects a doubled null-plus-one-power
panel at 1,000 reps/cell to about 18 minutes, or about 36 minutes at 2,000
reps/cell, before adding multiplier work.

## Run health

FIML converged in 39,985/40,000 attempts. All 15 failures were optimizer
nonconvergence in MCAR cells: 14 in the two-factor model and one in the
one-factor model. MLR was finite in 39,894 attempts; after excluding fit
failures, 91 additional p-values were non-finite.

FMG returned results in 39,156 attempts. After excluding fit failures, 829
calls hit the same strict projected `U Gamma` rank diagnostic seen in the
smoke run. The bifactor accounted for 797, the two-factor model for 27, and the
one-factor model for five. The growth model had no FMG failures. These failures
are reported separately and are not counted as non-rejections.

## Null calibration

Across the 40 cells, Yuan--Bentler MLR had mean rejection .366, mean absolute
size error .316, and range .062--.932. Only one cell was in the descriptive
.025--.075 band. Pooled rejection over its 39,894 finite values was .365.
The worst cell was the severe-VM bifactor with MCAR (.932); the complete-data
version was also .819. Thus the failure is not created solely by missingness,
although MCAR generally amplifies it.

The FMG corrections improved substantially but did not solve the grid:

- `all`: mean cell rejection .066, mean absolute error .024, range .022--.168;
- SS: mean .073, mean absolute error .028, range .027--.183;
- pEBA(4): mean .135, range .057--.381;
- SB: mean .163, range .060--.450.

FMG `all` and SS each put 26/40 cells inside .025--.075. Their remaining
liberality was concentrated in severe nonnormal MCAR cells. For `all`, the
largest rates were .168 for bifactor VM2+MCAR, .161 for one-factor VM2+MCAR,
and .123 for bifactor VM1+MCAR. Complete-data `all` pooled to .047, versus .085
under MCAR; MLR pooled to .298 and .433 respectively.

## Consequence for the next method

This run strengthens the negative Yuan--Bentler result across ordinary latent
models, but it also removes the easy conclusion that an existing FMG tail
transform is uniformly adequate. The missing general multiplier comparator is
therefore central rather than decorative. It should test the residual
`df`-dimensional complement of each fitted SEM's tangent using casewise
observed-data contributions, without fitting or inverting a saturated model on
each multiplier draw. Its first validation must separate the direct saturated
null-score formulation from the first-order-equivalent saturated-EM
moment-influence formulation, reduce exactly to the complete-data projected
residual experiment, and report the bifactor rank/conditioning diagnostics
rather than silently dropping them.
