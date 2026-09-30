---
name: magmaan-experiment
description: Create, extend, run or report a focused magmaan repository experiment, with reproducible runners and result-driven Quarto reports. Use for showcases, replications, research, engineering checks and default-setting decision studies under experiments; not independent paper/private-note workflows or component fixture generation.
---

# magmaan experiment

Read `experiments/AGENTS.md`, its index and any study-local instructions. For an
existing study, read its report/design and current runner before editing. Start
from the question, requested operation, chosen category and evidence needed;
do not expand a focused question into an inference-policy project.

## Choose the operation

- **Create/extend:** locate the existing leaf or next category-local number.
  Create the required QMD, runner and ignore files; update the collection index.
  Read [references/runners.md](references/runners.md) before implementing the
  compute path and [references/reports.md](references/reports.md) before writing
  the narrative. Keep SEM primitives in core/bindings and harness mechanics in
  `_support`; never import a sibling study or paper.
- **Run:** inspect help and existing checkpoints/results, select a cheap smoke
  or pilot, then execute the requested scope. Read the runner reference. Record
  cost/progress and preserve failed attempts. Do not overwrite frozen evidence
  or describe a planned run as observed evidence.
- **Report/render:** read local result metadata and summaries; read the report
  reference. Rendering reads results only, never runs estimation or simulation.
  Missing results need a clear reproduction command, not invented findings.
- **Default decision:** additionally read
  [references/decisions.md](references/decisions.md) before touching criteria,
  launching a decision run or claiming a default is justified. Its safeguards
  supplement the normal runner/report workflows.

Classification alone does not authorize archiving/deleting a study or changing
a library default. Preserve paper-supporting designs/results/reproduction paths.

## Deliver and verify

The outputs are one question's runner, standalone report, local results with
provenance, and updated index as applicable. Run runner help and a meaningful
smoke for changed compute paths; verify deterministic replicate IDs/seeds,
failure reporting and required metadata. Render when results are available and
check that the answer, uncertainty, cost and scope agree with them. Frozen
summaries need their documented reason/provenance.

Run `just check-layering` and `just check-tracked` for new artifacts; manually
inspect report result paths. C++/binding changes also need their focused checks
and vendor refresh. Update the roadmap/backlog when findings change contracts,
state or concrete remaining work. Commit explicit completed paths on the current
branch without absorbing unrelated work.
