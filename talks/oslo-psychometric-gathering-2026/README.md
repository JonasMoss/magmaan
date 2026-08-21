# Oslo Psychometric Gathering talk

Quarto reveal.js presentation about building magmaan with coding agents and the
architectural and validation choices that made repository-scale collaboration
possible.

## Files

- `talk.qmd` — presentation source.
- `outline.md` — narrative, timing, cuts, and source notes.
- `theme.scss` — local reveal.js theme.
- `tools/repo_metrics.py` — dependency-free metrics generator.
- `generated/repo_metrics.csv` — ignored, regenerated before every render.

## Build

From this directory:

```sh
just render
```

Or from the repository root:

```sh
just talk-oslo
```

The Quarto pre-render hook runs the metrics tool automatically. The standalone
presentation is written to `_site/talk.html` with resources embedded.

To inspect the metric definitions without rendering:

```sh
python3 tools/repo_metrics.py --repo ../..
```
