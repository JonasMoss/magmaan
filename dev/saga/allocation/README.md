# Sigma2 / NRIS resource application (umbrella magmaan project)

Tracked working area for the magmaan group's national e-infrastructure
allocation on Saga (Sigma2 / NRIS). The motivation is the whole magmaan
simulation pipeline: many methods papers, each validated by large embarrassingly
parallel Monte-Carlo studies (Type-I, power, coverage across generator ×
sample-size × model-size × invariance-level × replicate grids). We are currently
bottlenecked on laptops and paid cloud (Modal); Saga removes that ceiling and the
"can't run my own project on a coauthor's allocation" problem.

## Files

- `proposal-draft.md` - the application narrative (summary, science, methods,
  resource justification, software/storage, track record). Paste/adapt into MAS.
- `compute-budget.md` - bottom-up core-hour estimate, anchored on the one grid
  we have actually timed (experiment 61). Refine the per-paper rows with the
  paper leads before submitting.

## Process (verify current details on the Sigma2 site before submitting)

- Allocations are 6-month periods: **1 Apr - 30 Sep** and **1 Oct - 31 Mar**.
- Apply through **MAS** (NRIS Administration System) at applications.sigma2.no.
- The Resource Allocation Committee judges scientific excellence + efficient,
  justified need. A modest, well-justified first ask that scales beats an
  inflated one.
- **Keep it short.** Sigma2 national access is a MAS form plus a tight project
  description, not a ten-page grant. Aim for ~1 page of prose and a clear
  resource-justification table. Cite only the key methods references. Author
  credibility is PI eligibility + funding source, not a narrative.
- **Funding source is mandatory** in MAS. Have it ready.
- **Fast path / top-up:** extra allocations mid-period need only a short
  justification and can be submitted any time. So a modest base ask now can be
  grown without waiting for the next round.

## Deadlines / status

| Item | Value |
| --- | --- |
| Next deadline | **~26 Aug 2026** (verify on sigma2.no) |
| Period it funds | 1 Oct 2026 - 31 Mar 2027 |
| Headline ask (draft) | **120-150 kCPU-h** (see `compute-budget.md`) |
| Flagships | estimated-weight-se, fiml-fmg, ordinal-fmg, guttman-inference |
| PI / institution | TODO - Sigma2 requires a lead at a Norwegian institution |
| Funding source (mandatory) | TODO |
| MAS project created | TODO |
| Storage (NIRD) needed? | small; results are CSV/JSON. TODO decide |
| Status | DRAFT |

## Bridge before 1 Oct

For immediate FIML-MI compute that belongs to the coauthored project, use that
project's existing allocation. This umbrella application is for magmaan-owned
work and for scaling the whole pipeline going forward.

Sources: <https://www.sigma2.no/apply-e-infrastructure-resources>,
<https://documentation.sigma2.no/getting_started/applying_resources.html>
