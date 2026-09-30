# Default-setting decision studies

Read the existing lane report, committed `criteria/<lane>.md`, current library
default and applicable `experiments/AGENTS.md` before a decision operation.
One study can cover multiple estimation lanes, each with its own criteria.
Engineering explorations inform a decision but do not substitute for its
preregistered confirming evidence.

Before the decision run, commit the decision, candidate configurations, problems,
judge, acceptance rules and the outcome they imply. Do not rewrite the criteria
after seeing outcomes; a later amendment is dated and identified as post-results.
The baseline is the library default called without options. Use library
`fit$converged` as the common judge, including for other engines at their estimates.

Separate held-out problems from candidate-development populations. Development
controls are labeled and reported, never gating. Use fresh seed bases, distinct
from smoke and previous decision runs; draws prompting fixes become development
data, so confirmation after a fix needs fresh draws. Report every family
separately, every regression against baseline, and the result implied by the
precommitted rule. Do not retrospectively pool away a loss.

Commit per-run summary CSVs and `metadata.csv` under
`results/<lane>/<run-id>/`; raw per-fit outputs stay local. Preserve seed,
command, source/package provenance and the criteria version needed to recover
the decision. The report opens with a register of each covered default, value,
settled/provisional/open status and supporting evidence, followed by the concise
finding and caveats. An inconclusive result remains open/provisional.

Do not change the default merely because a benchmark wins. Apply the stated
rule and the authorized implementation scope, retain losses as regressions, and
update the maintained implementation contracts/backlog if the decision changes
what users get.
