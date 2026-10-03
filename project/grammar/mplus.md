# Mplus input frontend (planned)

Milestone 0.3.0. Increments 1–2 and 5 are accepted; increment 3 implementation awaits
its printed-SE meaning gate. Development
starts now, alongside 0.2.0 release work, and is not a 0.2.0 exit criterion.
This document owns the target, input boundary, increments, evidence and the
stability bar. The [backlog](../backlog/todo.md#mplus-input-frontend) owns task
state. The [source inventory](mplus_source_inventory.md) owns manual evidence,
page references and the Demo probe list, as the
[EQS inventory](eqs_source_inventory.md) does for EQS. The normative grammar
is [mplus_grammar.ebnf](mplus_grammar.ebnf).

## Target

magmaan reads an Mplus input file and lowers its linear SEM model into the
existing model triple, reproducing Mplus's model defaults exactly. Like the
[EQS frontend](eqs.md), it is a model-language adapter. It adds no fitting or
inference capability, does not run Mplus jobs and does not reproduce Mplus's
estimator conventions. An accepted input fits through the existing estimators
within [scope](../scope.md). Anything else is rejected with a classified
diagnostic at its source span.

The purpose is migration and replication. Mplus is the most widely used
commercial SEM program in the fields magmaan serves, and published `.inp` files
are plentiful. Hand translation of the syntax is easy but often gets the
defaults wrong. lavaan's `mplus2lavaan()` is not a usable reference. The
textbook-corpus Mplus audit records that it expands `y21-y31` numerically
instead of in NAMES order, drops label lists, ignores Mplus's x-variable, mean
and growth defaults, and cannot express PWITH, MODEL CONSTRAINT or
group-specific MODEL sections.

Stability is a requirement from the first increment, not a later phase; see
the [stability bar](#stability-bar).

## Input boundary

The frontend reads the whole input file. An Mplus input never contains its
data: DATA names a separate data file, which only the data increment reads.
MODEL cannot be read alone, because its meaning depends on other commands:

- `y21-y31` expands in VARIABLE NAMES order, not numerically.
- CATEGORICAL adds thresholds and scale factors (delta) or fixed residual
  variances (theta), and removes intercepts.
- GROUPING supplies the labels that group-specific `MODEL g:` sections use.
- ANALYSIS `MODEL = NOCOVARIANCES` and `NOMEANSTRUCTURE`, and PARAMETERIZATION,
  change the defaults.
- ANALYSIS TYPE decides whether the input is in scope at all.

Every command and option belongs to exactly one class. Nothing is silently
skipped; an option outside the inventory is rejected. The inventory's
classification table (CL rules) is the complete per-option list.

| Class | Members | Treatment |
| --- | --- | --- |
| Schema | VARIABLE: NAMES, USEVARIABLES, CATEGORICAL, GROUPING. ANALYSIS: TYPE, MODEL, PARAMETERIZATION | Imported into the model triple and schema |
| Data description | DATA: FILE, FORMAT, TYPE, NOBSERVATIONS, LISTWISE. VARIABLE: MISSING | Parsed into a data plan, used only by the data reader |
| Execution | TITLE. ANALYSIS: ESTIMATOR, INFORMATION, iteration and convergence controls, BOOTSTRAP. OUTPUT, SAVEDATA, PLOT. MODEL TEST. LOOP/PLOT in MODEL CONSTRAINT | Recognized and reported as not imported |
| Rejected | DEFINE, USEOBSERVATIONS, SUBPOPULATION, weights, and the model families under [not planned](#not-planned) | Diagnostic with source span and reason class |

ESTIMATOR is not imported but does screen scope. With CATEGORICAL, an ML-family
estimator denotes a full-information link model fitted by numerical
integration, not the limited-information model. Such an input is rejected
rather than reinterpreted as DWLS.

DEFINE is Mplus's data-transformation command: it creates or changes variables
before the analysis (products, centering, standardizing, logs, conditional
recodes, CUT). USEOBSERVATIONS and SUBPOPULATION select cases. Ignoring them
would silently change the analysis sample, so they are rejected; users perform
them in R. A future subset is a
[trigger-register entry](../backlog/speculative.md#mplus-define-and-case-selection).

## Language increments

Each increment changes the normative grammar first, then the C++ lowering,
then fixtures and gates, then the lab adapter. Defaults listed below are those
the corpus translator reproduces against shipped Mplus output and, where noted,
the Mplus 9.1 Demo. The inventory confirms each with manual pages before
implementation.

1. **Single group, continuous outcomes.** Input-file structure and command
   classification; BY, ON, WITH, PWITH, PON; means and intercepts `[ ]`;
   variances; `@` fixing and `*` freeing/starts; labels per statement and label
   lists; ranges in NAMES order; latent ranges; `!` comments; IS/ARE/=
   forms and documented abbreviations. Defaults: first loading of a factor's
   first BY statement fixed at one; x variables conditioned on; free observed
   intercepts; free residual covariances among all final dependent variables
   that are not factor indicators, observed or latent; free covariances
   among exogenous factors unless NOCOVARIANCES; an unmentioned USEVARIABLES
   variable gets a free mean and variance and no covariances. Two Mplus
   behaviors are rejected rather than reproduced: mentioning an x
   variable's variance or mean, which brings that one variable into the
   model with no covariances (mixed conditioning), and NOMEANSTRUCTURE
   without an explicit `INFORMATION = EXPECTED`, which Mplus ignores. The
   Demo probes settled these rules; where Mplus 9.1 departs from the guide,
   the frontend follows 9.1.
2. **Multiple groups.** GROUPING labels and integer codes; group-specific MODEL
   sections; the invariance defaults (equal loadings and indicator intercepts,
   factor means fixed at zero in the first group and free in the others);
   frees and overrides by mention in a group-specific section. With one data
   set the first (reference) group is the one with the lowest grouping value,
   not the first label declared; codes that GROUPING does not list leave the
   analysis. A single `ANALYSIS: MODEL = CONFIGURAL`, `METRIC` or `SCALAR`
   setting is accepted; a list of them is rejected until its result shape is
   decided.
3. **Categorical outcomes.** CATEGORICAL thresholds `[u$k]` and threshold
   ranges; delta (scale factors) and theta parameterizations; threshold
   invariance and free scale factors or residual variances in later groups.
   Lowering is complete for every categorical model; fitting is limited to
   supported routes. All-ordinal DWLS fits; continuous plus categorical
   outcomes wait for the 0.3.0 mixed workflows; categorical outcomes regressed
   on observed covariates need
   [conditional categorical moments](../backlog/speculative.md#conditional-categorical-moments-with-observed-covariates).
   Unsupported routes return an explicit unsupported-fit result. Binary and
   ordinal CONFIGURAL/SCALAR shortcuts are supported; METRIC is rejected
   according to Mplus 9.1. DELTA `{u}` rows retain free, fixed and equality
   restrictions directly; their latent-response residuals are derived.
   Data-driven completion preserves the frontend defaults and does not
   invoke lavaan group-equality releases.
4. **Growth and derived parameters.** `|` statements with time scores and
   their defaults (outcome intercepts fixed at zero, free growth means; the
   intercept-factor mean is fixed when outcomes are latent or categorical).
   MODEL CONSTRAINT: NEW, defining equations and equalities, onto the existing
   defined-parameter, equality and constraint machinery. Inequalities are
   [deliberately refused](../scope.md#inequality-constraints-deliberately-refused): the rejection explains why and
   points to `covariance = "psd"` or `"barrier"` for admissibility. MODEL
   INDIRECT (IND, VIA) becomes defined parameters.
5. **Data files.** The C++ frontend parses the data description into a data
   plan: file reference, free or fixed FORMAT (its own grammar production),
   individual or summary data, NOBSERVATIONS and MISSING codes. `mplus_data()`
   in the lab reads the file through R, applies NAMES and missing codes and
   selects USEVARIABLES. Mplus's analysis-sample rules (dropping cases missing
   an x variable or every dependent variable) belong to estimation conventions,
   not to the reader.

If a construct needs representation beyond the existing LISREL matrix
representation, for example an indicator that is also a structural predictor,
it reuses the general-equation work of the
[EQS sequence](eqs.md#implementation-sequence-planned). The Mplus frontend adds
no second representation.

### Not planned

Rejected with a classified reason: mixture and latent class models; Bayes;
random slopes; XWITH; ESEM rotation; count, censored, nominal, survival and
two-part outcomes; categorical ML by numerical integration; complex-survey
analysis and sampling or frequency weights; EFA; imputation and plausible
values; DSEM and time series; Monte Carlo and MODEL POPULATION. TWOLEVEL
input targets a parked model family and has its own
[trigger entry](../backlog/speculative.md#mplus-twolevel-input). An
[estimation-convention preset](../backlog/speculative.md#mplus-estimation-convention-preset)
that reproduces Mplus's printed numbers is also a trigger entry, not a planned
increment.

## Lowering into the model triple

- **`LatentStructure`** carries every parameter row the input denotes,
  including rows generated by Mplus defaults, with free/fixed status and
  values; equalities from labels, label lists and group invariance defaults;
  MODEL CONSTRAINT restrictions; and defined parameters from NEW and INDIRECT.
  Rows generated by a default carry provenance distinguishing them from rows
  the user wrote.
- **`LatentNames`** carries observed names from NAMES (matched
  case-insensitively to supplied columns), factor names, Mplus labels, group
  labels and the x-variable classification.
- **`Starts`** carries `*` values. Mplus's automatic starts are not emulated;
  magmaan's start policy applies otherwise.
- Build options disable every lavaan automatic addition, as
  `compat::eqs::build_options()` does. The frontend materializes all Mplus
  defaults explicitly. x variables lower to the lab's fixed-x convention,
  because Mplus conditions on them.
- Specifications retain the original input text, resolved schema and options.
  They rebuild through the C++ Mplus entry point or a lossless portable
  triple, never through a row-only lavaan string; the EQS frontend showed that
  such a string cannot carry the full contract. A lavaan-syntax projection is
  an optional interoperability output.

Grouped models use canonical integer code strings (`"1"`, `"-1"`) in
`LatentNames::group_labels`, ordered by numeric value. The grouping variable
retains its NAMES spelling. `MplusInput` and `api::MplusModel` retain source
labels separately as ordered `MplusGroup { label, code }` metadata. Negative
integers and integral decimal spellings are accepted; fractional values are
rejected following Mplus 9.1 (recode them in R; DEFINE is not imported).
Default cross-group equalities use reserved `.mgN.` labels, distinct from
user equality numbers `.eqN.`. Group modifiers carry free/fixed/value/label
and start vectors. A group-only parameter has a generated fixed-zero entry
in other groups; `compat::mplus::apply_provenance()` marks generated defaults
and zero entries in `LatentNames::row_user` before the partable projection.
Repeated group sections apply cumulatively in source order. Changes to the
variable-role sets in a group section are rejected: the shared role contract
cannot preserve group-specific indicator/predictor/dependent classifications.
`mplus_data()` drops unlisted codes and reports their row counts. Direct
raw-data fitting still rejects unlisted codes; filter explicitly or use the reader.

Increment 5 stores a typed `MplusDataPlan` in `MplusInput` and `api::MplusModel`;
lab specs expose it as `$mplus_data_plan` and retain the input directory for
working-directory-first relative FILE resolution. Separate `FILE (label) =`
entries define groups in statement order; summary `NGROUPS` defines g1, g2, ... .
These groups have empty codes, label-valued triple group labels, and reserved
`.mplus_group` metadata. For individual file groups the reader creates that
column with the source labels. Group lowering otherwise follows the same
invariance, provenance and reference-group rules as GROUPING, with the first
declared file or summary group as reference. FILE groups and NGROUPS cannot
be combined with GROUPING; summary inputs require NOBSERVATIONS per group.

Summary TYPE without MEANS lowers no intercept/mean rows; explicit bracket
mentions require MEANS (Demo P-DA3). CORRELATION without STDEVIATIONS uses
unit variances on input, recorded in a note (P-DA4). Summary matrices have
divisor N-1 on input and are rescaled by (N-1)/N for ML, matching Mplus and
lavaan's default sample.cov.rescale (P-DA5). FREE and bounded fixed FORMAT
operations (Fw.d or Fw, w.d, X, Tn, /, repeats and nested groups) are parsed by
C++; the R reader executes them and compares MISSING flags after decimal
scaling. Global symbols are `MISSING = .`, `*`, or `BLANK`; numeric flags use
NAMES lists/ALL and ranges. Fixed blanks without BLANK missing read as zero.
NOBSERVATIONS limits individual observations and is reported. LISTWISE and
analysis-sample deletion are reported but left to fitting conventions.

| Data rules | Evidence |
| --- | --- |
| CL02–CL03, CL09, MG01–MG03, DA01–DA06 | Typed-plan/parser/API units; independently written R frames/matrices; 25 Demo data cases / 28 group moment comparisons; P-DA1–P-DA5; single/multigroup corpus data-to-fit round trips |


### Continuous group coverage

| Rules | Accepted subset and gate |
| --- | --- |
| MG01–MG03, CL11, CL26 | One explicit integer-code GROUPING; numeric order, case-insensitive labels, repeated sections; reader units, Demo P-MG3/P-MG8–P-MG13, ordered-data golden and R fit |
| MG04–MG08, MG11, MG13 | Default equalities, second-order and structural freedom, identifier ties, group means, starts/fixes and releases; independent grouped rows, TECH1 partitions and numerical goldens |
| IV01, IV02, IV04, IV05 | One CONFIGURAL/METRIC/SCALAR, continuous first-order BY, marker or explicitly fixed-one factor variance identification; 38/34/30 parameters and 16/20/24 df in both identification variants |
| MG09, MG12, IV03 | Categorical scale rules await increment 3; KNOWNCLASS remains rejected |

Grouped meaning/numeric fixtures are generated independently by
`cpp/tests/tools/regen_oracle_mplus_groups.R` into the `multigroup_cases` section of `golden.json`.
The local sweep is reproducible with `check_mplus_input_corpus.R`; the end-to-end
corpus gate remains `check_mplus_corpus.R`.

## Oracles and evidence

Categorical numerical gates retain native/lavaan conventions. An independently
written explicit model with lavaan `mimic="Mplus"` must reproduce the Demo's
printed estimates, SEs, scaled test, df and parameter count; the same model
under default lavaan conventions is the native numerical reference. For P-IV2
ordinal SCALAR, the default scaled test is about 27.483 and Mplus mimic/Demo
about 27.535: the robust scaling factors differ (delta 0.864268291773 versus
0.862539755189; theta 0.864268266371 versus 0.862539729839). This does not
introduce an Mplus estimation preset. Corpus test/SE mismatches count as
convention differences only when Mplus mimic reproduces the printed result
and default lavaan reproduces magmaan; estimates and df remain direct gates.
The existing printed-precision allowance is unchanged.

`regen_oracle_mplus_categorical.R` freezes ten independently specified default-
lavaan WLSMV references in `mplus/golden_categorical.json`. C++ lowers the
original inputs and gates rows, free flags, estimates, SEs, scaled/shifted test
and df. Live lab tests cover grouped defaults/shortcuts, prepared reconstruction
and fixed/equal DELTA scales. P-IV2 Demo TECH1 checks meaning separately.
The optional generator `--demo` gates parameter count, df, estimates and scaled
tests at printed precision. Printed SEs are retained as convention observations;
the categorical Demo/mimic differences are recorded in the test ledger.

The local corpus gate accepts 31 of 68 cases: 25 match and six report
unsupported fit routes; two of the matched cases have independently verified
SE/test convention differences. Increment 2 accepted and matched 22.
The 2,440-input sweep advances reader acceptance from 291 to 456 and model
lowering from 217 to 359, with every rejection classified.

**Meaning** (which model an input denotes) is gated against Mplus:

- The User's Guide v8 language chapters (VARIABLE/DATA/DEFINE, ANALYSIS,
  MODEL, the language summary), the special-modeling-issues chapter and the
  language addenda through 9.1, summarized in the source inventory with
  documented/derived/unresolved classes as for EQS. No full guide newer than
  v8 exists; the addenda cover 8.1 to 9.1.
- The Mplus 9.1 Demo, installed locally at `~/mplusdemo/mpdemo` (at most six
  dependent and two independent observed variables). `mpdemo probe.inp` writes
  `probe.out` beside the input in about a second and touches nothing else.
  TECH1 prints Mplus's own parameter specification per matrix: free-parameter
  numbering and fixed values. It independently checks rows, fixed/free status
  and equalities for small probes, much as LISREL's Parameter Specifications
  check the Little translations. Probes are agent work: a maintainer tool
  under `cpp/tests/tools/` runs them and writes derived summaries, following
  `regen_oracle_twolevel_mplus.R`.
- The optional textbook corpus, used in two tiers:
  - **End-to-end gate.** 79 verified cases keep the original `.inp`, the
    shipped `.out`, the analysed data and Mplus-verified expected values
    (User's Guide v8 40, Muthén et al. 2017 20, Brown 2015 19); Geiser 2013
    adds 28 with inputs in the raw companion files. Every case the frontend
    accepts is fitted from its original input and compared with Mplus on N,
    free-parameter count, df, chi-square, log-likelihood and every printed
    estimate, where the fit conventions match.
  - **Sweep.** Every Mplus input in the corpus, about 2,300 including the
    archives (also the Mplus versions of Little's and Newsom's models), is
    read and lowered: accepted or rejected with a classified reason, never a
    crash. The tally of rejection reasons is the coverage evidence and
    guides which later increments matter most.
  A maintainer script extracts the inputs to an ignored cache; the gates
  run locally when the corpus is present.

**Numerics** stay with lavaan, the component oracle: the lowered triple's
lavaan projection, fitted by pinned lavaan, must match magmaan's fit as for
any other model source.

The corpus translator (`ingest/_mplus_helpers.R` in the corpus repository) is a
cross-check only. The C++ frontend is implemented from the manual and Demo
evidence; agreement with that translator is a regression signal, not proof.

No Mplus inputs, outputs or manual text are tracked. C++ fixtures use
magmaan's own small inputs with independently written expected rows; Demo
results may be checked in as derived parameter summaries. Corpus and Demo
gates run locally when present and are recorded in the
[test ledger](../validation/test_ledger.md). CI needs neither.

## Stability bar

The frontend is stable for a subset when all of the following hold:

- A normative `mplus_grammar.ebnf` covers it, and every parser function cites
  its production.
- Every accepted rule has a fixture with independently written expected rows.
  Every default rule also has a Demo TECH1 check or a corpus `.out` match.
- Every rejected construct class has a diagnostic fixture with source span and
  reason class. Each rejection message states what was found, what Mplus
  does with it, why magmaan does not reproduce it (or which increment will
  accept it), and what to write instead. Examples: NOMEANSTRUCTURE without
  `INFORMATION = EXPECTED` (Mplus keeps the means and only warns: add the
  INFORMATION setting or remove NOMEANSTRUCTURE); a mentioned x variance
  (Mplus models that one variable with no covariances while conditioning on
  the others: remove the mention); `a1b-a3b` in NAMES (Mplus generates A01,
  A02, A03: list the names).
- Robustness sweep: every Mplus input in the corpus (about 2,300, including
  those using unsupported features) is accepted or rejected with a
  classified reason. None crashes, hangs or exceeds the expansion bounds,
  under sanitizers (`just test-dev`). Accepted corpus inputs match their `.out` on free-parameter
  count and df, and on chi-square, log-likelihood and estimates where fit
  conventions agree.
- Lab round trips agree: partable, fresh versus prepared fit, rebuild/refit and
  save/reload into a fresh worker.
- A coverage matrix keyed to inventory IDs is in this document, and the lab
  help page lists exactly the accepted subset.

An increment merges into `main` only when complete. Every merged state is
stable for its documented subset and rejects everything else; there is no
alpha exposure.

## Exposure

- **C++.** A parser in `parse::`, compatibility settings and projections in
  `compat::mplus`, and an `api::` model constructor, following the EQS layout.
  Names are fixed by the first increment.
- **Lab.** `magmaanlab::mplus_model()` after each gated increment;
  `mplus_data()` from increment 5. Thin adapters; no R parser.
- **Ordinary.** Ordinary integration is the goal, unlike EQS, whose ordinary
  adoption is undecided: migrating applied users are ordinary users. The
  ordinary constructor accepts an explicitly marked Mplus source, never by
  automatic detection, and applies the ordinary inference policy. Decide
  first how an Mplus model with x variables meets the
  [ordinary fixed-x decision](../scope.md#ordinary-fixed-x-decision): ordinary
  construction rejects fixed-x specifications and never silently converts
  them, whereas Mplus conditions on x by default.

## Implementation sequence

0. **Source inventory and grammar baseline** (planner). Inventory with
   evidence classes and pages, a probe list for unresolved rules (each with a
   probe input and the TECH1 or MODEL RESULTS entry that settles it) and the
   normative EBNF for increment 1.
   **Demo probes** (lane). Run the probe list through the Demo, record each
   result against its inventory ID and check in derived summaries. The
   planner resolves the inventory and grammar from the results.
1. **Increments 1–5** in order, each a self-contained lane: grammar, C++
   lowering, fixtures, Demo and corpus gates, lab exposure, `just vendor`.
   Increment 5 needs only increment 1 and may run in parallel with 2–4.
2. **Stability closeout.** Robustness sweep, coverage matrix, lab help and
   test-ledger entries.
3. **Ordinary integration**, after the fixed-x decision.
