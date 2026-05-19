# Documentation Proposal

This proposal sketches the documentation system magmaan should grow next. It
is intentionally a proposal, not a Quarto scaffold yet: the first task is to
settle audience, vocabulary, API status, and page shape before creating a tree
of placeholder files.

The main decision is to document two different surfaces:

- a C++ compositional methods manual for the public primitive layer below
  `magmaan::api`;
- a user-facing API manual for the staged workflow exposed by C++, R, and a
  future Python binding, adjusted for each language's conventions.

The old phrase "core API manual" is ambiguous after the core/frontier status
split. In the status vocabulary from [ideas.md](ideas.md), **core** is a
source-compatibility promise, not a synonym for "low-level C++." The manual
for the non-`api` C++ layer should therefore be named something like
**C++ Compositional Methods Manual**. It will include core, frontier, and
compatibility surfaces, but every documented entry must say which status it has
and what evidence supports it.

## Documentation Principles

- Document magmaan's ontology first. Lavaan remains the oracle where
  compatibility is claimed, but the manual should describe magmaan model
  construction, magmaan data objects, magmaan moment vectors, and magmaan
  post-fit primitives. Use lavaan terminology only when discussing compatibility
  projection, oracle fixtures, or user-facing lavaan-style syntax.
- Avoid historical lavaan-derived build verbs in user-facing docs. Prefer
  "build a magmaan model", "model construction", "partable projection", or
  "compatibility projection", depending on the actual operation.
- Keep statistical status separate from API status. A method can be
  lavaan-compatible but still have an unsettled C++ shape; a non-lavaan method
  can become core if its software contract and evidence are settled.
- Document only public surfaces. Internal `src/` helpers, uninstalled detail
  headers, and `detail` namespaces should not appear as callable documentation
  targets, though they may be mentioned in design notes when needed.
- Prefer topic pages over namespace dumps. A methodologist should find
  "Robust tests based on U-Gamma" or "RMSEA and close-fit tests", not a page
  whose only organizing idea is `robust.hpp`.
- Every methods page should combine notation, assumptions, scaling
  conventions, public entry points, examples, validation status, and references.
- Unsupported combinations should be documented as part of the contract when
  they are likely to be attempted.
- Quarto is the right authoring system because the docs need equations,
  citations, code examples, HTML output, and eventually PDF-friendly manuals.

## API Status In Documentation

The manuals should use the status vocabulary from [ideas.md](ideas.md):

- **Core**: settled source-level contract. Names, signatures, result shapes,
  scaling conventions, and failure modes are documented; breaking changes go
  through deprecation.
- **Frontier**: public, tested methods-development surface with no
  deprecation-cycle promise yet. Frontier is not a dumping ground and not a
  waiting room; it is the permanent home for many research primitives.
- **Compat**: compatibility projection and naming helpers, currently centered
  on `compat::lavaan`. These match lavaan-shaped surfaces but do not define
  magmaan's internal ontology.
- **Rejected**: combinations that fail explicitly because the contract is not
  supported.
- **Undesigned**: plausible future work with no current public contract.

Each documented function or workflow should also state its **statistical
evidence**: lavaan fixtures, independent oracle, source-paper replication,
simulation, finite-difference invariant, shape test, or exploratory use.

The long-run documentation contract should be a generated methods catalog:
one row per model/data/estimator/post-fit slice with API status, evidence,
entry point, header, tests, and explicit rejected combinations. Until that
catalog exists, topic pages should carry this information locally.

## Manual 1: C++ Compositional Methods Manual

Audience: methods developers, SEM researchers, and contributors who want to
compose the primitive C++ layer directly.

Scope: installed public headers under `include/magmaan/`, excluding the
friendly staged facade in `include/magmaan/api/`. The manual may reference
`magmaan::api` only to explain what the lower layer feeds. It should cover
domain namespaces such as `parse`, `spec`, `model`, `data`, `estimate`,
`inference`, `robust`, `measures`, `optim`, and `compat::lavaan`, including
their nested `frontier` surfaces.

Tone: mathematical and explicit, but still runnable. Each major topic should
define terms, write down the relevant formulas or transformations, cite the
literature, and show a minimal code path.

Suggested structure:

```text
C++ Compositional Methods Manual
+-- Orientation
|   +-- What magmaan is and is not
|   +-- API status vs statistical evidence
|   +-- Public surface layers
|   +-- Error values and source-level stability
|   +-- Reading examples by estimator/data slice
|
+-- Model Construction
|   +-- Lavaan-style syntax as input syntax
|   +-- Parsed partables and magmaan model construction
|   +-- LatentStructure, LatentNames, and Starts
|   +-- Matrix representation and implied moments
|   +-- Names, labels, constraints, and starts
|   +-- Compatibility projection to LavaanParTable
|   +-- Composite models
|
+-- Data And Moments
|   +-- Raw continuous data
|   +-- Complete-data sample statistics
|   +-- Ordinal sample statistics
|   +-- Mixed continuous/ordinal statistics
|   +-- Moment vectors, weights, Gamma, and block conventions
|   +-- Frontier moment builders
|
+-- Estimation
|   +-- Complete-data normal-theory ML
|   +-- Continuous FIML
|   +-- ULS, GLS, WLS, and explicit moment quadratics
|   +-- Ordinal and mixed categorical DWLS/WLS
|   +-- Bounds and constrained estimation
|   +-- Separable nonlinear least squares
|   +-- Optimizer concepts
|   +-- Convergence diagnostics
|
+-- Inference
|   +-- Expected and observed information
|   +-- Standard errors and vcov
|   +-- Wald and z tests
|   +-- Score tests and modification indices
|   +-- Defined parameters
|   +-- Nested model tests
|
+-- Robust Corrections
|   +-- U-Gamma machinery
|   +-- Sandwich covariance estimators
|   +-- Weighted chi-square mixture reducers
|   +-- Satorra-Bentler-family compatibility wrappers
|   +-- FIML MLR-style reporting
|   +-- Robust LS reporting
|   +-- FMG eigenvalue tests
|
+-- Fit Measures And Residuals
|   +-- Chi-square, df, and baseline models
|   +-- RMSEA family and close-fit tests
|   +-- CFI and TLI
|   +-- SRMR and residual summaries
|   +-- Information criteria
|   +-- Standardized parameters
|   +-- Factor scores and effects
|
+-- Ordinal And Pairwise Methods
|   +-- Thresholds and response scales
|   +-- Polychoric estimators
|   +-- Polyserial estimators
|   +-- Pairwise influence functions
|   +-- H-score frontier estimators
|   +-- DPD frontier estimators
|   +-- Robust correlation repair and shrinkage
|
+-- Reference
    +-- Methods catalog
    +-- Function index
    +-- Header index
    +-- Status glossary
    +-- Bibliography
```

### Topic Page Template

Each substantial topic page should use a predictable shape:

```text
Title
+-- Status summary
|   +-- API status: core/frontier/compat/rejected/undesigned
|   +-- Statistical evidence
|   +-- Public headers
|   +-- Main entry points
+-- Purpose
+-- Notation
+-- Assumptions and supported slices
+-- Computation
+-- Scaling and block conventions
+-- Failure modes and rejected combinations
+-- Minimal C++ example
+-- Relationship to compatibility labels, if any
+-- Validation tests and fixtures
+-- References
```

### First Exemplar Pages

The first docs should be chosen to establish style and expose real naming
pressure:

- **The magmaan model contract**: `LatentStructure`, `LatentNames`, `Starts`,
  matrix representation, and compatibility projection.
- **Robust tests based on U-Gamma**: moment Jacobians, Gamma, U matrices,
  group/block stacking, total-N scaling, weighted chi-square reducers, robust
  SEs, and where lavaan/Mplus labels enter only as compatibility presets.
- **RMSEA family and close-fit tests**: noncentrality, truncation, df,
  standard vs robust/scaled forms, baseline dependencies, close-fit p-values,
  and estimator-specific availability.
- **Polychoric and polyserial estimators**: default compatibility paths,
  frontier h-score and DPD variants, influence/Gamma construction, and what is
  SEM-facing versus pair-level diagnostic machinery.

## Manual 2: User-Facing API Manual

Audience: people fitting SEMs through the staged friendly API, whether from
C++, R, or a future Python binding.

Scope: the friendly workflow, not the primitive graph. This manual should be
language-neutral in organization and language-specific in examples. It should
explain the same conceptual steps across bindings: construct a model, construct
data or moments, fit point estimates, then explicitly request inference,
robust corrections, fit measures, defined parameters, or summaries.

The user-facing API shape is still unsettled. The proposal should therefore
document the intended workflow and not overcommit to class names beyond what
already exists. Current names such as `magmaan::api::Model`, `Data`, `Fit`,
`Analysis`, and `Summary` can appear as the C++ snapshot, while the manual
leaves room for R and Python to choose idiomatic object syntax.

Suggested structure:

```text
User-Facing API Manual
+-- Orientation
|   +-- Staged workflow
|   +-- Estimate first, post-fit explicitly
|   +-- What is computed on demand
|   +-- How C++, R, and Python examples correspond
|
+-- Fit A Model
|   +-- Specify a model
|   +-- Provide data or moments
|   +-- Choose an estimator
|   +-- Fit point estimates
|   +-- Inspect convergence
|
+-- Add Inference
|   +-- Information matrices
|   +-- Vcov and standard errors
|   +-- Wald and z tests
|   +-- Robust corrections
|   +-- Modification indices
|   +-- Defined parameters
|
+-- Compare Models
|   +-- Standard nested tests
|   +-- Robust nested tests
|   +-- Compatibility methods
|   +-- Required model relationships
|
+-- Report Fit
|   +-- Chi-square and df
|   +-- Fit measures
|   +-- Residuals
|   +-- Standardized results
|   +-- Factor scores and effects
|
+-- Data Types
|   +-- Complete continuous data
|   +-- Missing continuous data with FIML
|   +-- Ordinal data
|   +-- Mixed continuous/ordinal data
|   +-- Multi-group data
|
+-- Recipes
    +-- CFA
    +-- Structural regression
    +-- Growth model
    +-- Multi-group model
    +-- Equality constraints
    +-- Composite model
    +-- Robust reporting
```

## Quarto Layout

A minimal Quarto layout can wait until the first exemplar pages are ready, but
the likely structure is:

```text
docs/
+-- manuals/
|   +-- compositional/
|   |   +-- _quarto.yml
|   |   +-- index.qmd
|   |   +-- model/
|   |   +-- data/
|   |   +-- estimation/
|   |   +-- inference/
|   |   +-- robust/
|   |   +-- measures/
|   |   +-- ordinal/
|   |   +-- reference/
|   +-- user-api/
|       +-- _quarto.yml
|       +-- index.qmd
|       +-- workflows/
|       +-- inference/
|       +-- model-comparison/
|       +-- reporting/
|       +-- recipes/
+-- bibliography/
    +-- sem.bib
```

The manuals should share a bibliography and status glossary. Cross-links
should point from user-facing workflow pages into compositional methods pages
only when the extra detail helps methods users; ordinary users should not need
to read primitive pages to understand a standard workflow.

## Near-Term Plan

1. Keep this proposal as the planning document for the docs pass.
2. Create a small methods catalog prototype, probably from YAML, before the
   manuals try to be exhaustive.
3. Write the first three compositional pages:
   - the magmaan model contract;
   - robust tests based on U-Gamma;
   - RMSEA family and close-fit tests.
4. Let those pages define the Quarto style, equation conventions, citation
   style, status boxes, and code-example style.
5. Add the first user-facing workflow page only after the friendly API shape
   stabilizes enough that the page will not become a fossil immediately.
