# Paper Corpus Scouting Notes

Reconnaissance date: 2026-05-22.

Status update: the first tracked scout now lives in the ignored nested
`external/paper_corpus` repository. Its export,
`tests/fixtures/paper_corpus/scout_manifest.json`, inventories the seed OSF
nodes below, scans only small code files for lavaan-shaped signals, and keeps
raw downloads out of magmaan history.

Promotion update: `zxqvn` is now the first promoted paper-corpus fixture.
`external/paper_corpus/scripts/export_magmaan.R` reads tracked minimal derived
files from the nested corpus repo and writes
`external/paper_corpus/exports/magmaan/zxqvn_reference.json`; magmaan's bridge
script copies that export to `tests/fixtures/paper_corpus/zxqvn_reference.json`.
The fixture freezes the core complete-data ML point-estimate surface. The
source script's clustered-SE option is recorded as catalogued but outside this
first parity surface.

The goal is a future `paper_corpus` alongside the textbook corpus: empirical or
methods-paper SEM examples with reusable model syntax and data, preferably
discovered automatically from public repositories. The first implementation
should be a scout/index, not a fixture generator: find candidate projects,
inventory files, inspect small scripts for lavaan usage, and only then decide
which data files are worth downloading and promoting into lavaan-backed
fixtures.

## Discovery Architecture

- Use OSF/web search for candidate nodes, but assume search mostly covers
  metadata and filenames rather than full file contents.
- OSF references used during scouting:
  <https://help.osf.io/article/588-getting-started-with-osf-search> and
  <https://developer.osf.io/>.
- Use the OSF API to recurse through public node files and child components:
  `https://api.osf.io/v2/nodes/{node_id}/files/`,
  `https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/`, and
  `https://api.osf.io/v2/nodes/{node_id}/children/`.
- Download small code files first (`.R`, `.Rmd`, `.qmd`, `.html`, `.sps`,
  maybe `.txt`) and scan for `lavaan`, `sem(`, `cfa(`, `growth(`, model string
  assignments, and likely data-loading calls.
- Defer large data downloads until a code file looks promising. Track file
  sizes and hashes from OSF metadata before downloading.
- Keep raw downloaded material in ignored storage (`resources/` or
  `external/`). Only commit derived manifests, extracted model/data summaries,
  and lavaan oracle fixtures whose redistribution status is acceptable.

Suggested first tracked artifact:

```text
tests/fixtures/paper_corpus/scout_manifest.json
```

Useful fields: source platform, node id, title, URL, file inventory, candidate
code files, candidate data files, detected lavaan calls, model count, data
format, size class, license/redistribution note, and promotion status.

## Candidate Seeds

### OSF `gduy4` Family

- Root: <https://osf.io/gduy4/>
- API: <https://api.osf.io/v2/nodes/gduy4/>
- Title: "Perfectionism, Negative Motives for Drinking, and Alcohol-Related
  Problems: A 21-day Diary Study"
- Why promising: public project with child components for data and tutorials.
  A Journal of Open Psychology Data article says the dataset has been used for
  SEM tutorials, including measurement invariance and cross-lagged panel
  modeling with lavaan.
- Article: <https://openpsychologydata.metajnl.com/articles/10.5334/jopd.44>

Important child components found:

- `hwkem`: <https://osf.io/hwkem/>, "Measurement Invariance Tutorial Paper"
  - Files verified through API:
    - `CLPM_D6.v4.Rmd`, 22 KB, direct download
      <https://osf.io/download/jyz2u/>
    - `Abridged Comp Data.csv`, 54 KB, direct download
      <https://osf.io/download/achzs/>
    - `Total Dataset.sav`, 3.8 MB
    - `my_CLPM_work_space.v4.RData`, 9.2 MB
  - Code scan confirmed many lavaan models: configural, metric, scalar,
    residual CFA, structural SEM, and random-intercepts cross-lagged panel
    model. This is the strongest first empirical seed.
- `m9wdn`: <https://osf.io/m9wdn/>, "Diary Study Data"
  - Files include `Total Dataset.csv`, `Total Dataset.sav`, codebook PDFs, and
    SPSS scale-score syntax.
  - Useful as a shared data source for `hwkem`, but less directly model-rich.
- `yez2b`: <https://osf.io/yez2b/>, "Psychometric Study Data"
  - Files include `Psychometric Data.csv`, `Psychometric Data v2.sav`, and
    `Psychometric Syntax.sps`.
  - More psychometric preprocessing than lavaan, but likely useful for
    measurement examples or data overlap checks.

### OSF `zxqvn`

- Node: <https://osf.io/zxqvn/>
- API file listing showed:
  - `analysis_code.R`, 3.6 KB, direct download
    <https://osf.io/download/9zxmr/>
  - `data.rdata`, 42 KB, direct download <https://osf.io/download/3z8t2/>
  - `Supplementary material.pdf`
- Code scan confirmed a compact lavaan mediation/SEM:

```r
affect =~ positive + negative
PEBT2 ~ intention + affect + PEBT1
intention ~ affect + PEBT1
affect ~ PEBT1
```

The script fits it with `sem(model1, data=data, cluster = "participant_id")`.
This is a strong small candidate, although clustered SE behavior may need to be
catalogued separately from the linear SEM core parity surface.

### OSF `rntmc`

- Node: <https://osf.io/rntmc/>
- API file listing showed:
  - `stress and bp shared codes.Rmd`, 13 KB
  - Several large CSV files, roughly 20 MB to 164 MB
  - `Supplemental materials.pdf`
- Status: candidate only. It has code plus data, but the data files are large.
  Scan the Rmd before any full data download.

### OSF `d6hs7`

- Node: <https://osf.io/d6hs7/>
- Related paper:
  <https://www.tandfonline.com/doi/full/10.1080/10705511.2024.2398034>
- API file listing showed many R scripts and a `Data/` folder:
  `simulation.R`, `do_sim.R`, `do_sim_reestimation.R`, `sim_VAR.R`,
  `step1.R`, `step2.R`, and support scripts.
- Code scan confirms heavy lavaan use, but the project looks simulation and
  methods oriented rather than empirical-data-first. Treat it as a methods-paper
  candidate, not the first paper-corpus seed.

## Proposed First Pass

1. Done: build a read-only OSF scout script under `external/paper_corpus/scripts/`.
2. Done: seed it with the nodes above plus a small manually maintained query result
   list.
3. Done: write a tracked scout manifest with file inventories and lavaan-detection
   summaries, but do not commit raw OSF downloads to magmaan.
4. Partly done: promote `hwkem` and `zxqvn` first:
   - `hwkem` for realistic longitudinal CFA/SEM/RI-CLPM complexity.
   - Done: `zxqvn` for a compact empirical mediation SEM with small data.
5. Next: generate lavaan oracle fixtures only after each promoted case has a clear
   license/data handling note and a supported magmaan surface classification.
