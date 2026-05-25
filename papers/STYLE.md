# Style

How magmaan papers look and read. This complements `papers/AGENTS.md`
(which defines structure and layout) with the visual and prose conventions
every paper inherits. Paper-specific direction lives in each paper's own
AGENTS.md.

When you draft text or build a table or figure, this file is the reference.

## Audience Profiles

Each paper's AGENTS.md declares a `Profile:` line. Three options:

| Profile | Target journals | Math | Theorems | Default body (pages) | Brief slot |
|---|---|---|---|---|---|
| **Math** | Psychometrika, BJMSP, JEBS | dense | yes, in body | **17–22** | Short Note, <10 pp |
| **Applied** | SEM, MBR, Psychological Methods (also EPM for measurement work) | moderate, in displays | rare | **10–14** | Notes & Comments, <10 pp |
| **Tool** | BRM | light, tutorial OK | almost never | **11–14** | Short Report, <10 pp |

Pages are total published pages (manuscript including refs), from a
2020–2025 OpenAlex pull of articles ≥ 5 pages across the listed journals.
Targets are the **lower quartile** of each journal cluster, on the
default-aim-short rule. Real medians: Psychometrika 26, BJMSP 25, JEBS 29
(the longest cluster), MBR 19, Psychological Methods 19, EPM 25, SEM 14,
BRM 18. A paper may overshoot when there is content to justify it; it
should not overshoot by default.

What each profile expects beyond length:

- **Math.** Theorem/Proof environments fine. Symbol density expected.
  Figures sparing; one or two carefully chosen plots. Discussion short.
  Prose closer to a clean mathematical exposition than to a tutorial.
- **Applied.** Theorems only as math displays plus paragraph statement,
  never as `Theorem 1` environments. Path diagrams welcome (TikZ).
  Simulation study expected. Discussion in plain prose. Reproducibility
  against lavaan / Mplus is central.
- **Tool.** Tutorial framing OK. Software description with usage examples.
  Validation studies stand in for theory. Code and data effectively
  required at submission.

A `Brief slot` paper drops to roughly half the default length and
typically carries one of: a closed-form result, a parity correction, a
small software release, or a single-finding empirical note. The
manuscript should be submitted explicitly to the journal's brief
category, not as a short regular article.

## Tables

APA shape, `booktabs` always, no exceptions:

- **Rules.** `\toprule`, `\midrule`, `\bottomrule` only. Never `\hline`,
  never `|` between columns.
- **Caption above the tabular**, ending in a period, self-contained. Lead
  with the verb. Name sample size, design dimensions, and the headline
  finding when there is one. A reader who skims the captions should still
  follow the paper.
- **Spanner headers** via `\cmidrule(lr){a-b}` instead of repeating
  column-name prefixes. If three columns are all `Median time (ms)` for
  three optimizers, put `Median time (ms)` once over a `\cmidrule`-spanned
  group and let the three sub-columns name the optimizers.
- **Decimal alignment** via `siunitx` `S` columns. One number format per
  column, never per row. Round for interpretation; keep raw precision in
  the corresponding CSV under `results/`.
- **Headers in human Title Case** (`Median Time (ms)`), never raw
  identifiers (`median_elapsed_ms`).
- **At most ~12 columns.** Above that, transpose or split. Spanner groups
  count once.
- **Group repeated row labels** with an extra `\midrule` and a group label;
  do not repeat the same string down the leftmost column.
- **Notes below the tabular** in a `\parbox` matching table width.
  General Note first, one short sentence on what is shown. Symbol
  footnotes second, alphabetic superscripts in order of appearance. Notes
  describe *what*, not *how*; method details belong in the body text.

Worked example:

```latex
\begin{table}[t]
\caption{Per-backend wall time for the Geiser-corpus benchmark.
$n = 28$ models, 1000 replications; SNLLS is 1.4$\times$ faster on
average.}
\label{tab:geiser-time}
\centering
\begin{tabular}{l S[table-format=2.1] S[table-format=2.1] S[table-format=1.2]}
\toprule
        & \multicolumn{2}{c}{Median Time (ms)} & {Ratio} \\
\cmidrule(lr){2-3}
Model   & {L-BFGS} & {PORT} & {PORT / L-BFGS} \\
\midrule
Geiser 1 & 12.4    & 18.1   & 1.46 \\
Geiser 2 &  8.9    & 13.2   & 1.48 \\
\bottomrule
\end{tabular}
\par\smallskip
\footnotesize\textit{Note.} Median across 1000 replications per cell.
\end{table}
```

What to refuse:

- Vertical rules and `\hline`.
- Footnote bushes longer than the table itself.
- A "results" column with three rows in different units.
- Captions that say only `Table 3: Timings.`
- A 30-row design grid pasted whole; summarise with top-N or grouped
  slices and refer the reader to the CSV.

## Figures

APA shape, monochrome-safe, paper-consistent:

- **One theme** across every figure in every paper. `theme_classic()` base
  with the modifications listed below. A future `papers/_support/` package
  will export `theme_paper()` so it is one call.
- **Fixed palette per dimension**, reused across figures and across papers.
  Methods are always the same colour wherever they appear, optimizers are
  always the same colour, weights are always the same colour. Use the
  Okabe-Ito palette below as the default; pick a stable per-dimension
  assignment per paper and document it once at the top of the figure
  script.
- **Linetypes vary alongside colour.** A black-and-white print of any
  figure must still be readable. The "headline" series is solid; comparators
  are dashed, dotted, dot-dash.
- **Captions** follow the table rules: above-the-figure in APA, but in
  practice LaTeX places them below; lead with the verb, self-contained,
  headline finding inline when there is one.
- **Vector formats only.** PDF for the manuscript, no PNG for plots.
- **Standard widths.** Single-column `3.5"` (~89 mm), two-column `7"`
  (~178 mm). Default to single-column. Heights chosen for a roughly 3:2
  or 4:3 aspect ratio, never wider than 7" or taller than 5".
- **No chartjunk.** No 3D, no shadows, no gradients, no chart background
  shading, no per-bar texture. Gridlines only where they aid reading
  (use sparing horizontal gridlines for value comparison; never both
  axes).
- **Axes** labelled with units, sensible breaks, no superfluous tick
  marks. Numbers respect the table number-format rules.
- **Legends** inside the plot if there is room and they do not overlap
  data; otherwise to the right. Title-Case labels, never raw identifiers.

The Okabe-Ito palette (colourblind-safe, eight colours):

```
black   #000000
orange  #E69F00
sky     #56B4E9
green   #009E73
yellow  #F0E442
blue    #0072B2
red     #D55E00
purple  #CC79A7
```

A typical per-dimension assignment for one paper:

```r
paper_palette <- list(
  method   = c(full  = "#000000", snlls = "#0072B2"),
  backend  = c(lbfgsb = "#0072B2", nlminb = "#D55E00",
               port   = "#009E73", port_nls = "#CC79A7"),
  weight   = c(uls = "#000000", gls = "#0072B2", adf = "#D55E00")
)
paper_linetype <- list(
  method   = c(full = "solid", snlls = "dashed"),
  backend  = c(lbfgsb = "solid", nlminb = "dashed",
               port = "dotted", port_nls = "dotdash")
)
```

Pick the assignment once per paper, write it at the top of the figure
script (or in the paper's R helper package), and reuse.

## Citations

APA throughout. `natbib` with `apalike` is the magmaan default and is
what the existing manuscripts use. Stay with it.

- `\citet{key}` for textual citations: `Author (Year) showed...`
- `\citep{key}` for parenthetical: `...(Author, Year).`
- `\citep[][p.~123]{key}` for page references.
- `\citep[e.g.,][]{key}` for "e.g." prefixes.
- `\citep{a,b,c}` for multi-citation; let `apalike` sort them.

`.bib` hygiene:

- Author names in proper case (`Bollen, K. A.`), never all-caps. Use
  protective braces (`{B}ollen`) only where the bibstyle would otherwise
  lowercase wrongly.
- Journal names full or consistently abbreviated within a paper. Pick
  one style for the whole `.bib` and stick to it.
- Include DOIs when available; URLs only for materials without a DOI.
- Distinguish preprint and published versions explicitly; do not silently
  cite the preprint when the journal version exists.
- One `.bib` per paper folder. Do not share `.bib` files across papers;
  copy the entries you need.

## Math

A short notation contract that holds across all magmaan papers unless the
per-paper AGENTS.md overrides for a documented reason:

- Italic lowercase for scalars and vectors: `$x$`, `$\theta$`, `$y$`.
- Italic uppercase for matrices: `$X$`, `$\Sigma$`, `$\Lambda$`.
- Bold reserved for explicit collections when the distinction matters:
  `$\boldsymbol{\theta}$` for the full parameter vector when individual
  `$\theta_j$` also appear and need disambiguation. Do not bold by default.
- Multi-letter operators via `\operatorname{}`: `\operatorname{vec}`,
  `\operatorname{tr}`, `\operatorname{diag}`, `\operatorname*{arg\,min}`.
  Never write `vec(X)` as upright text; always as an operator.
- Equation numbers only on equations the prose references. Unnumbered
  displays for transitional algebra.
- Do not display a single-symbol expression. `$x$` stays inline.
- Use `\,` for thin spaces in integrals and product names (`\,dx`,
  `\arg\,min`).

**Symbol glossary.** Every paper carries a short Symbol Glossary section
near the front of the manuscript, listing each symbol once with its
meaning. Prevents the "p means both p-value and number of indicators"
drift. Update the glossary whenever notation changes.

Theorems by profile:

- **Math.** `Theorem`, `Lemma`, `Proposition`, `Corollary`, `Proof`
  environments fine. Proofs in the body when short, in a short appendix
  when long. State assumptions explicitly above each statement.
- **Applied.** No `Theorem` environments. State the result as a numbered
  display plus a short paragraph naming assumptions and consequence.
  Citations carry the rigour.
- **Tool.** No formal mathematical statements unless they are immediate
  consequences of definitions.

## Prose

The voice we use, the words we do not.

### Voice

Pick "we" or impersonal per paper and stick with it. Recommended:

- **Math, Tool.** Impersonal: "The estimator solves...", "The
  simulation reports..."
- **Applied.** "We" is fine: "We fit the model with...", "We benchmark
  against lavaan."

Past tense for what was done (methods, results). Present tense for what
holds (definitions, theorems, ongoing implications). Do not slip into
future tense ("we will show that") outside a paragraph that genuinely
forecasts later material.

### Structural rules

These hold across every profile:

- **No em-dashes.** Use a comma, a pair of parentheses, or two sentences.
  En-dashes in compound names (Golub-Pereyra) and numeric ranges are
  correct and stay.
- **No semicolons.** Split into two sentences, or use a comma.
- **Few colons.** A colon may introduce a genuine list. Do not use one
  for a dramatic pause before a punchline.
- **No antithesis.** Sentences that negate a foil and then assert the
  real point ("It is not X, it is Y", "not one decision but several",
  "this is not a failure but a demarcation") almost never carry
  information the positive statement does not. State the point directly,
  or drop the sentence.
- **No short blogger assertions.** Punchy standalone sentences that
  announce or characterise the text instead of advancing it
  ("The recommendation is short.", "This paper is an engineering
  study.") belong in the trash. Open paragraphs with substance.
- **Methods-developer voice.** Direct, specific, modest about claims.
  Say what the work can and cannot establish. Separate statistical
  conclusions from engineering diagnostics.

### Words and phrases to refuse

LLM-isms and stock academic filler that should never appear in a magmaan
paper:

- **Empty meta-comments.** "It is worth noting that...", "It is
  important to note that...", "Notably,", "Importantly,",
  "Interestingly,", "It is interesting to note that...". The reader can
  decide what is interesting. Say the thing.
- **Filler verbs.** "delve into", "dive into", "explore", "leverage",
  "utilize", "facilitate", "demonstrate". Use "show", "use", "let".
- **Filler adjectives.** "robust", "powerful", "elegant", "intuitive",
  "comprehensive", "novel" when they describe the work being presented.
  Show the property; do not assert it.
- **Vague quantifiers.** "various", "numerous", "myriad", "plethora",
  "a wide range of". Name the number or the cases.
- **Empty connectives at sentence start.** "Moreover,", "Furthermore,",
  "Additionally,", "In addition,". Either the sentence connects on its
  own or it should be merged with the previous one.
- **Bookend phrases.** "In this section we will...", "Having discussed
  X, we now turn to Y", "In conclusion,". The section structure is its
  own signal.
- **Hedge stacks.** "may potentially", "could possibly", "appears to
  perhaps", "tends to generally". Pick one hedge or none.
- **Triadic padding.** "X, Y, and Z" structures repeated across many
  consecutive sentences. Vary the rhythm. If you have two items, write
  two.
- **Clause chains.** "X is critical for Y, ensuring Z and providing W."
  Break into two sentences or drop the trailing clauses.
- **False precision.** "approximately 78.34%", "roughly 1.245$\times$".
  Round when hedging; do not hedge a precise number.
- **"Notably" stand-ins.** "Of particular interest", "It bears
  mentioning", "Of note", "Remarkably". Same problem, slightly different
  costume.

Mostly these compound: a sentence with two of them is wrong twice. When
in doubt, delete the sentence and re-read; if nothing breaks, the
sentence was filler.

### What good looks like

- A clear claim per sentence, with the verb early.
- Numbers come from the macros file, never typed by hand.
- Caveats stated in the same paragraph as the claim, not deferred.
- Method paragraphs end with what the choice buys, not with what it is.

## Build Hygiene for Style

Style consistency that the pipeline must enforce, not the author:

- **Every cited number is generated.** Prose numbers come from a single
  `tables/<slug>_stats.tex` file written by the paper's table script.
  Caption numbers come from the same file. The manuscript carries `??`
  fallbacks for unbuilt macros so it compiles before the tables exist.
- **Figure and table file slugs match in-text `\label{}` slugs.**
  `figures/geiser_profile_time_ratio.pdf` paired with
  `\label{fig:geiser-profile-time-ratio}` (kebab-case in labels,
  snake-case in filenames is fine, but the stem must match).
- **Tables and figures prefixed with the paper slug.**
  `tables/snlls_geiser_corpus_summary.tex`, not `tables/summary.tex`.
  Prevents cross-paper collisions and makes grep useful.
- **Symbol glossary updated when notation changes.** A grep for `$p$`
  should agree with what the glossary says `p` means.
- **One R script writes one rectangular CSV.** A script that produces
  three artifacts produces three CSVs. Tables and figures read CSVs,
  never call simulation code.

## Notes on This File

These conventions hold by default. A paper may override a rule when the
paper-specific AGENTS.md explains why. Conventions that get overridden
in two or more papers should migrate back here.

The published-page numbers in the audience-profile table come from a
2020–2025 OpenAlex query (`type:article`, ≥5 pages, filtered by journal
ISSN). The query script can be re-run when the targets need refreshing;
the methodology note is preserved in the commit message that introduced
this file.
