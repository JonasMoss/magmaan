# Report workflow


A report is a short talk, not a lab notebook. Write it for a colleague who has
sixty seconds with you on a Zoom call. They should leave knowing three things:
the question, the answer, and how much to trust it. Everything else is optional
reading.

Keep the opening concise and scientifically complete:

- **The first screen is the whole story.** The report opens with the question
  and the answer, in prose, before any table or figure. A reader who stops after
  the first screen still gets the finding. If the answer only emerges by reading
  a table, the report has failed.
- **Lead with the conclusion, never the setup.** No metadata dump, threshold
  table, or design grid before the answer. Setup, if shown at all, comes after
  the finding; most of it belongs in `results/`.
- **One summary artifact per finding.** Each genuine outcome gets at most one
  table or one plot, showing a summary slice, not the full design grid. A
  replication may carry several outcomes; that is fine. A report's length should
  track the number of real findings, never the size of the design grid. If a
  table has a row per design cell, it belongs in `results/`, not the report.
- **Explain conventions that affect interpretation.** Information, sandwich,
  scaling and reference-distribution choices belong in the report when they
  distinguish the methods being studied. Keep the opening readable; put detailed
  methods after it. Code locations and implementation bookkeeping belong in
  code comments or metadata. Caveats explain uncertainty and scope to the reader.

Ordinary reports use this opening order:

1. **Question** - one or two sentences. What did we want to know, and why does
   it matter?
2. **Short answer** - the finding in prose, naming the numbers that matter. The
   sentence you would say out loud. For a multi-outcome replication, one sentence
   per outcome.
3. **Evidence** - one summary table or plot per finding, rounded for reading;
   optionally a few sentences on the key slice.
4. **Caveats** - what this run can and cannot establish. Reader-facing only.
5. **Reproduce** - the commands.

Detailed design/method sections may follow these. Default-decision reports
instead open with the default register required by `experiments/AGENTS.md`,
then give the question and short answer without burying them in setup.

Self-check before committing a report: read only the Question and Short answer
aloud. If that is not a complete, honest account of the result, fix those two
sections before touching anything else.

Recommended Quarto defaults:

```yaml
format:
  html:
    toc: true
    number-sections: false
execute:
  echo: false
  warning: false
  message: false
```

Reports should read from `results/`, not from hidden local state. If the
required result files are absent, stop with a command the reader can run.

## Tables And Figures

Tables in the report are presentation objects, not data dumps.

- Use properly capitalized, human-readable headers such as `Median Time (ms)`,
  not raw names like `median_elapsed_ms`.
- Keep tables short enough to fit on one screen. Use summaries, top-N rows, or
  grouped slices instead of printing a full design grid or all replicates.
- Put long tables in `results/*.csv` and mention the file in prose.
- Round numbers for interpretation. Keep raw precision in the CSV.
- Prefer one clear plot over several near-duplicates.

## Style

Use the repo's methods-developer voice: direct, specific, and modest about
claims. Say what the experiment can and cannot establish. Keep comparisons
cost-aware when optimizer or simulation work is involved, and separate
statistical conclusions from engineering diagnostics.

When a finding changes implementation state, validation expectations, or the
active backlog, update `project/architecture/roadmap.md` or
`project/backlog/todo.md` as appropriate.
