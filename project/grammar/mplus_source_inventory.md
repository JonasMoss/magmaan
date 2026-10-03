# Mplus input language: source inventory

Target: the linear single-level SEM subset of an Mplus input file, as planned
in the [Mplus frontend plan](mplus.md). This inventory records source evidence,
implementation consequences and the probes that settle open rules. It does not
implement anything or claim Mplus numerical parity. The normative grammar is
[mplus_grammar.ebnf](mplus_grammar.ebnf); task state is in the
[backlog](../backlog/todo.md#mplus-input-frontend).

## Sources and reading map

Muthén, L. K., & Muthén, B. O. (1998–2017). *Mplus User's Guide*, eighth
edition. Muthén & Muthén. Public PDF at
<https://www.statmodel.com/download/usersguide/MplusUserGuideVer_8.pdf>:
950 physical pages, SHA-256
`b055701a72f0515c192f44f95a5df5f1e6081b44db1f42f318eea9ad9185f1aa`.
Printed page **p** is physical PDF page **p + 6**. References below use printed
pages; "UG" means this guide.

Language addenda from the same site (no newer full guide exists for the
installed 9.1 Demo; the site lists no addenda for 8.2–8.4 or 8.6–8.8, so
changes made only in those versions are invisible here and are left to the
probes):

| Addendum | Pages | SHA-256 (prefix) |
| --- | --- | --- |
| Version 8.1 Language Addendum | 9 | `d002c274881a` |
| Version 8.5 Language Addendum | 7 | `46a79f92918c` |
| Version 8.9, 8.10 and 8.11 Addendum | 13 | `1f76f2948330` |
| Version 9 Language Addendum | 5 | `400798a3f62e` |
| Version 9.1 Language Addendum | 2 | `1e24eb9e4cc3` |

PDFs, full checksums (`SHA256SUMS`), per-page text from `pdftotext -layout`
(`pages/pdf_NNN.txt`) and a helper that marks bold text (`bold_pages.py`, used
for documented setting abbreviations) are cached in ignored
`external/refs/mplus/`. Third-party material stays untracked. The rules below
are original summaries, not copied manual passages.

| Printed pages | Purpose |
| --- | --- |
| 13–14 | Command structure, line limit, abbreviations, comments, IS/ARE/=, lists |
| 515–520 | Parameter default settings and default starting values |
| 529–546 | Multiple group analysis: requesting, first group, defaults, group-specific MODEL, equalities, means, scale factors, data |
| 541–546 | Measurement-invariance model definitions |
| 563–650 | TITLE, DATA, VARIABLE, DEFINE |
| 651–710 | ANALYSIS |
| 711–790 | MODEL and its variations |
| 893–928 | Language summary |

## Evidence classes

**D** means directly stated or demonstrated in the UG or an addendum. **A**
means a consequence derived from documented rules. **U** means the sources do
not settle the exact behavior; each U rule needed by increments 1–2 has a
probe in the [probe list](#probe-list). **C** marks a rule that the
textbook-corpus translator reproduces against shipped Mplus output or an
earlier Demo run; C is supporting evidence, not a source. **P** marks a rule
settled by a Mplus 9.1 Demo probe ([results](#probe-results-mplus-91-demo));
where a probe contradicts the guide, the frontend follows 9.1 and the row
says so.

## Command and option classification

Classes follow the [input boundary](mplus.md#input-boundary): **S** schema
(imported), **DD** data description (data plan only), **E** execution
(recognized, reported as not imported), **R** rejected. "Later" marks a rule
that is rejected until its increment lands; the diagnostic says so. Setting
abbreviations are the bold stems of the option tables (LX03); settings with no
bold stem must be written in full.

| ID | Command / option | Class | Evidence and notes |
| --- | --- | --- | --- |
| CL01 | TITLE | E | 563. Text is reported. |
| CL02 | DATA: FILE | DD | 567. Quoted when the path has blanks; bare names are looked up locally, then beside the input. `FILE (label) =` per group also defines groups (MG01): later (increment 5). |
| CL03 | DATA: FORMAT, TYPE (INDIVIDUAL, COVARIANCE, CORRELATION, FULLCOV, FULLCORR, MEANS, STDEVIATIONS), NOBSERVATIONS, NGROUPS, LISTWISE | DD | 567–574. Stems: IND, COVA, CORR, STD; FULLCOV, FULLCORR, MEANS in full. NOBSERVATIONS with individual data keeps only the first N records and LISTWISE deletes cases: both change the sample and are reported. |
| CL04 | DATA: TYPE = MONTECARLO, IMPUTATION; SWMATRIX | R | 571–574. Multiple data sets; two-level WLS. |
| CL05 | DATA: VARIANCES | E | 575. Zero-variance check only. |
| CL06 | DATA IMPUTATION, WIDETOLONG, LONGTOWIDE, TWOPART, MISSING, SURVIVAL, COHORT | R | 575–595. Data transformations and non-ignorable missingness. |
| CL07 | VARIABLE: NAMES | S + DD | 598. |
| CL08 | VARIABLE: USEVARIABLES | S | 599–600. Default: all NAMES variables. Original variables precede DEFINE variables; the order governs later ranges (NM03). `ALL` as first entry means all NAMES variables. |
| CL09 | VARIABLE: MISSING | DD | 601–603. |
| CL10 | VARIABLE: CATEGORICAL (plain list) | S, later (increment 3) | 604–608. The category-set forms `(*)`, explicit sets and `(gpcm)`, `(3pl)`, `(4pl)` are ML-only and R. |
| CL11 | VARIABLE: GROUPING | S, later (increment 2) | 612–613. |
| CL12 | VARIABLE: IDVARIABLE; AUXILIARY plain list | E | 613–615. Only saved or plotted. |
| CL13 | VARIABLE: USEOBSERVATIONS, SUBPOPULATION | R | 599, 626. Case selection (see [trigger entry](../backlog/speculative.md#mplus-define-and-case-selection)). |
| CL14 | VARIABLE: CENSORED, NOMINAL, COUNT, DSURVIVAL, SURVIVAL, TIMECENSORED, LAGGED, TINTERVAL, TSCORES | R | 603–615, 635–639. Out-of-scope outcome types, survival, time series, random time scores. |
| CL15 | VARIABLE: AUXILIARY with modifiers, CONSTRAINT, PATTERN, FREQWEIGHT, WEIGHT and the other weight options, STRATIFICATION, CLUSTER, FINITE, REPWEIGHTS, CLASSES, KNOWNCLASS, TRAINING, WITHIN, BETWEEN | R | 614–635. Auxiliary-variable methods, data-dependent constraints, designs, weights, mixtures, multilevel. |
| CL16 | DEFINE | R | 639–650. See [trigger entry](../backlog/speculative.md#mplus-define-and-case-selection). |
| CL17 | ANALYSIS: TYPE | S | 651, 657–665; P. GENERAL (explicit or implied) accepted. The legacy settings MISSING, MEANSTRUCTURE and `GENERAL MISSING H1` are accepted by 9.1 as no-ops (P-DF6) and reported. BASIC requests descriptives only and is rejected for lowering. RANDOM, COMPLEX, MIXTURE, TWOLEVEL, THREELEVEL, CROSSCLASSIFIED and EFA are R. |
| CL18 | ANALYSIS: ESTIMATOR | E + screen | 652, 665–669. Reported. Screens: with CATEGORICAL, ML/MLR/MLF denote a full-information link model by numerical integration (666–667) and BAYES a Bayesian model; both R. MUML implies two-level: R. |
| CL19 | ANALYSIS: MODEL = NOMEANSTRUCTURE, NOCOVARIANCES | S | 652, 671. Stems NOMEAN, NOCOV. NOMEANSTRUCTURE only with TYPE = GENERAL. |
| CL20 | ANALYSIS: MODEL = CONFIGURAL, METRIC, SCALAR | S, later (increment 2) | 652, 670–671; see IV rules. Stem CONFIG; METRIC and SCALAR in full. |
| CL21 | ANALYSIS: MODEL = ALLFREE; ALIGNMENT and its controls | R | 671–674, 699–701. Mixture/Bayes alignment. |
| CL22 | ANALYSIS: PARAMETERIZATION = DELTA, THETA | S, later (increment 3) | 652, 674–675. LOGIT, LOGLINEAR, PROBABILITY and RESCOVARIANCES are mixture/ML-categorical settings: R. |
| CL23 | ANALYSIS: LINK; DISTRIBUTION other than NORMAL; MATRIX = CORRELATION | R | 652, 674, 677, 701. Link models, non-normal distributions, correlation-structure analysis. DISTRIBUTION = NORMAL and MATRIX = COVARIANCE are the defaults and E. |
| CL24 | ANALYSIS: ROTATION, ROWSTANDARDIZATION, PARALLEL, RSTARTS and other EFA/ESEM options; REPSE, MULTIPLIER; BASEHAZARD | R | 653–655, 678–695. |
| CL25 | ANALYSIS: NESTED (v8.1 addendum pp. 7–8), INFORMATION, BOOTSTRAP, DIFFTEST, COVERAGE, ADDFREQUENCY, iteration, convergence, start, integration-control, Bayes-engine, PROCESSORS and INTERACTIVE options | E | 654–710. Integration controls also signal a rejected model (CL18). ADDFREQUENCY changes polychoric inputs and is reported with that note. |
| CL26 | MODEL and `MODEL label:` | S | 713–715, 781; group-specific sections in increment 2. |
| CL27 | MODEL CONSTRAINT; MODEL INDIRECT (IND, VIA) | S, later (increment 4) | 759–772. Causal IND/MOD effects and data-dependent constraints (VARIABLE CONSTRAINT) are R. |
| CL28 | MODEL TEST | E | 772–774. A Wald-test request, reported. |
| CL29 | MODEL PRIORS, MODEL POPULATION, MODEL COVERAGE, MODEL MISSING; MONTECARLO | R | 775–790; Chapter 19. Bayes and Monte Carlo. |
| CL30 | OUTPUT, SAVEDATA, PLOT | E | Chapter 18; SAVEDATA NESTED (v8.1 addendum). |
| CL31 | `MODEL label:` without GROUPING | R | v8.9–8.11 addendum PDF pp. 1–3: with `ANALYSIS: MODEL = CONFIGURAL …` and no groups, `MODEL t1:`, `MODEL t2:` … are time points of a single-group longitudinal invariance model. A label section is accepted only as a group section of a declared GROUPING. |
| CL32 | MODEL PRIORS under any estimator | R | v8.9–8.11 addendum PDF pp. 8–9: penalties (PSEM) now also apply with ML, MLR and WLSMV, so MODEL PRIORS no longer implies Bayes. |

Options added by the addenda fall into these classes; CL25 and CL30–CL32
cite the additions that matter for the linear subset. All other addenda
changes concern rejected families (DSEM, mixtures, Bayes, alignment,
multilevel, SEFA rotation) or output only.


### Option names by command

From the language summary (UG 893–904) and the addenda. Class as in the CL
rules; "later N" means rejected until increment N, with a diagnostic saying
so. Option names resolve by a unique prefix of four or more letters within
their command (LX03); an ambiguous or unknown name is rejected, except in
OUTPUT, SAVEDATA and PLOT, whose options are all reported without
validation.

| Command | Option names | Class |
| --- | --- | --- |
| DATA | FILE, FORMAT, NOBSERVATIONS, NGROUPS, LISTWISE | DD (later 5 for summary data and multi-file groups) |
| DATA | TYPE | DD: INDIVIDUAL accepted; COVARIANCE, CORRELATION, FULLCOV, FULLCORR, MEANS, STDEVIATIONS later 5 (they change the mean structure); MONTECARLO, IMPUTATION R |
| DATA | VARIANCES | E |
| DATA | SWMATRIX | R |
| VARIABLE | NAMES, USEVARIABLES | S |
| VARIABLE | MISSING | DD |
| VARIABLE | GROUPING | S, later 2 |
| VARIABLE | CATEGORICAL | S, later 3 |
| VARIABLE | IDVARIABLE; AUXILIARY without modifier | E |
| VARIABLE | USEOBSERVATIONS, SUBPOPULATION, CENSORED, NOMINAL, COUNT, DSURVIVAL, FREQWEIGHT, TSCORES, AUXILIARY with a modifier, CONSTRAINT, PATTERN, STRATIFICATION, CLUSTER, WEIGHT, WTSCALE, BWEIGHT, B2WEIGHT, B3WEIGHT, BWTSCALE, REPWEIGHTS, FINITE, CLASSES, KNOWNCLASS, TRAINING, WITHIN, BETWEEN, SURVIVAL, TIMECENSORED, LAGGED, TINTERVAL | R |
| ANALYSIS | TYPE | S (settings below) |
| ANALYSIS | MODEL | S: NOMEANSTRUCTURE (see MS11), NOCOVARIANCES; CONFIGURAL, METRIC, SCALAR later 2; ALLFREE R |
| ANALYSIS | ESTIMATOR | E + screen (CL18) |
| ANALYSIS | PARAMETERIZATION | S, later 3 (DELTA, THETA); LOGIT, LOGLINEAR, PROBABILITY, RESCOVARIANCES R |
| ANALYSIS | INFORMATION | E, read by MS11 |
| ANALYSIS | DISTRIBUTION | NORMAL E; other settings R |
| ANALYSIS | MATRIX | COVARIANCE E; CORRELATION R |
| ANALYSIS | LINK, ALIGNMENT, ROTATION, ROWSTANDARDIZATION, PARALLEL, REPSE, MULTIPLIER, BASEHAZARD, RSTARTS, RITERATIONS, RCONVERGENCE, ASTARTS, AITERATIONS, ACONVERGENCE, SIMPLICITY, TOLERANCE, METRIC | R |
| ANALYSIS | CHOLESKY, ALGORITHM, INTEGRATION, MCSEED, ADAPTIVE, BOOTSTRAP, LRTBOOTSTRAP, STARTS, STITERATIONS, STCONVERGENCE, STSCALE, STSEED, OPTSEED, K-1STARTS, LRTSTARTS, H1STARTS, DIFFTEST, COVERAGE, ADDFREQUENCY, ITERATIONS, SDITERATIONS, H1ITERATIONS, MITERATIONS, MCITERATIONS, MUITERATIONS, CONVERGENCE, H1CONVERGENCE, LOGCRITERION, RLOGCRITERION, MCONVERGENCE, MCCONVERGENCE, MUCONVERGENCE, MIXC, MIXU, LOGHIGH, LOGLOW, UCELLSIZE, VARIANCE, POINT, CHAINS, BSEED, STVALUES, PREDICTOR, BCONVERGENCE, BITERATIONS, FBITERATIONS, THIN, MDITERATIONS, KOLMOGOROV, PRIOR, INTERACTIVE, PROCESSORS, NESTED | E |
| OUTPUT, SAVEDATA, PLOT | any | E |

Setting stems that the frontend must recognize (complete word or exact stem
only, LX03): TYPE: GENERAL/GEN, BASIC/BAS, RANDOM/RAND, COMPLEX/COM,
MIXTURE/MIX, TWOLEVEL/TWO, THREELEVEL/THREE, CROSSCLASSIFIED/CROSS, EFA, and
the legacy MISSING, MEANSTRUCTURE and H1 (CL17). ANALYSIS MODEL:
CONFIGURAL/CONFIG, METRIC, SCALAR, NOMEANSTRUCTURE/NOMEAN,
NOCOVARIANCES/NOCOV, ALLFREE/ALL. ESTIMATOR: ML, MLM, MLMV, MLR, MLF, MUML,
WLS, WLSM, WLSMV, ULS, ULSMV, GLS, BAYES (full names only). PARAMETERIZATION:
DELTA, THETA, LOGIT, LOGLINEAR/LOGLIN, PROBABILITY/PROB,
RESCOVARIANCES/RESCOV. INFORMATION: OBSERVED/OBS, EXPECTED/EXP,
COMBINATION/COMB. DISTRIBUTION: NORMAL/NORM, SKEWNORMAL/SKEW,
TDISTRIBUTION/TDIST, SKEWT. MATRIX: COVARIANCE/COVA, CORRELATION/CORR. DATA
TYPE: INDIVIDUAL/IND, COVARIANCE/COVA, CORRELATION/CORR, FULLCOV, FULLCORR,
MEANS, STDEVIATIONS/STD, MONTECARLO/MONTE, IMPUTATION/IMP. LISTWISE: ON,
OFF.

## Lexical rules and file structure

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| LX01 | D, 13–14 | Ten commands. Each begins on a new line and is followed by a colon; commands may come in any order; DATA and VARIABLE are required. Options end with semicolons and several may share a line. The lexer tracks line starts; a command head is recognized only at a line start. |
| LX02 | D, 14; P; corpus | Input records are at most 90 columns. 9.1 truncates a longer line with a warning (P-LX4), which can change or break a statement. In the corpus, every line beyond 90 columns is TITLE text or comment text, where truncation is harmless. The frontend therefore rejects a line only when content outside comments and TITLE text extends beyond column 90. |
| LX02a | P | No addendum changes the limit, and 9.1 still truncates at 90 columns (P-LX4). Settled; no separate rule. |
| LX03 | D, 14, 563, 567; P | Commands and options can be shortened to four or more letters (`USEVAR`, `ESTI` accepted; three-letter `USE` rejected). Settings accept only the complete word or the exact documented stem: `GEN` and `GENERAL` work, `GENE` and `NOCOVAR` are errors (P-LX2). An ambiguous or intermediate prefix is rejected. |
| LX04 | D, 14, 598 | Keywords and names are case-insensitive. Canonical matching folds case; source spelling is retained for diagnostics and names. |
| LX05 | D, 14; P | `!` starts a comment to the end of the line. `!* … *!` on one line removes only the delimited text. When `*!` closes on a later line, 9.1 also drops code that preceded `!*` on the opening line, and keeps text after `*!` (P-LX1). The frontend accepts a block that opens at a line start or closes on its opening line, and rejects a block opened after code that closes on a later line. |
| LX06 | D, 563; P | TITLE text runs to the next command head at a line start; `model: x` inside a title line is text (P-LX3). |
| LX07 | D, 14 | IS, ARE and `=` are interchangeable in all commands except DEFINE, MODEL CONSTRAINT and MODEL TEST. List items are separated by blanks or commas. A hyphen denotes a list of variables or numbers; ALL denotes all variables where an option documents it. |

## Names and lists

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| NM01 | D, 598, 744; P | Names and labels begin with a letter and contain only letters, digits and underscore. The guide limits them to 8 characters, but 9.1 accepts longer variable names and labels and keeps them distinct (TECH1 alone prints an 8-character prefix; P-NM2, P-LB6). The frontend accepts longer names and compares full names. |
| NM01a | P | Settled by P-NM2 and P-LB6; see NM01. |
| NM02 | D, 598; P | NAMES ranges generate names from a shared stem with a numeric suffix, keeping the endpoint digit width (`y08-y11` gives Y08 to Y11), or a single-letter suffix (`itema-itemd`). A mixed form is not an error in 9.1 but generates unrelated names: `a1b-a3b` becomes A01, A02, A03 and `x1y-x3y` becomes X01, X02, X03 (P-NM1; planner Demo check 2026-10-03). Such an input almost certainly does not mean what Mplus does, and reproducing it would rename data columns silently, so the frontend rejects other shapes and says what Mplus would generate. |
| NM03 | D, 730–731; P | A hyphenated range in USEVARIABLES or MODEL selects variables by position: in USEVARIABLES by NAMES position, in MODEL by the analysis (USEVARIABLES) order. `y1-y3` therefore includes any variable placed between them (P-NM3a: x1 became an indicator), and a range whose first endpoint comes later is an error (P-NM3b). It never expands by numeric suffix. (lavaan's `mplus2lavaan()` gets this wrong.) |
| NM04 | D, 731; P | Latent variables are ordered by their BY statements, then `|`-defined random effects. A range mixing a latent and an observed endpoint is an error (P-NM3c). |
| NM05 | D, 731–732 | A list on the left of ON or WITH implies one statement per element; a list on the right is a list of variables. BY accepts lists on its right. |

## MODEL statements

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| MS01 | D, 713–715 | Statement forms: BY, ON, PON, WITH, PWITH, a bare list (variances or residual variances), `[list]` (means, intercepts, thresholds), `{list}` (scale factors), `|` (growth, random effects, interactions). Modifiers: `*` frees with an optional start, `@` fixes at an optional value, `(…)` equalities or labels. MODEL is optional only for EFA, LCA, a baseline model and TYPE = BASIC. |
| MS02 | D, 718; P | The marker is the first indicator of the factor's first BY statement; later BY statements add free loadings (P-MS1a). A later mention of the marker, even as the first item of another BY statement, frees it (P-MS1b: the model became unidentified). A `*` on a range that includes the first indicator frees it (P-MS1c). The pp. 532–533 wording ("each BY statement") does not hold. |
| MS03 | D, 719 | A factor may appear on the right of BY only after its own BY definition (second-order factors). Out-of-order use is an input error. |
| MS04 | D, 722–723 | `y ON x` frees a regression coefficient with start zero. Observed or latent variables may appear on either side. |
| MS05 | D, 725–726 | `a b PON c d` pairs elements; both sides need equal length. `a b WITH c d` crosses all left and right elements; PWITH pairs them. |
| MS06 | D, 726, 745; A | WITH frees covariances among continuous variables (residual covariances for dependent variables); for categorical or censored variables only with weighted least squares. `y1-y3 WITH y1-y3` yields the distinct unordered pairs and no variances (shown by the label example on p. 745). Self-pairs and duplicate orientations collapse to one parameter. |
| MS07 | D, 728 | A bare variable list refers to variances of independent and residual variances of dependent variables; mentioning frees them. Categorical observed variables have no variance parameter (Theta exceptions belong to increment 3). |
| MS08 | D, 723; P | Mentioning the variance or mean of an observed independent variable brings that variable into the model: it gains a free mean and variance, its cases with missing values are kept (N 450 to 500), and it has no covariance with the independent variables that stay conditioned on. `x1 WITH x2` brings both in with their covariance (P-MS2). This mixed conditioning has no single-convention counterpart in magmaan, so increment 1 rejects such mentions. Bringing every independent variable in with all their covariances equals the joint random-x model; supporting that case later needs its own design. |
| MS09 | D, 719–722 | `(*label)` after BY defines ESEM factor sets; `~` gives target-rotation values. Both belong to ESEM and are rejected. |
| MS10 | D, 714–715, 723–725, 742–743 | `#` labels (latent classes, nominal categories, inflation parts, hazards), `%OVERALL%`, `%class%`, `%WITHIN%`, `%BETWEEN%` and the MODEL variants for mixtures, multilevel models and Monte Carlo belong to out-of-scope families and are rejected. |
| MS11 | D, 729–730; P | Means, intercepts and thresholds are in the model by default. ANALYSIS `MODEL = NOMEANSTRUCTURE` removes them for TYPE = GENERAL, but 9.1 ignores it with a warning under observed information, the default with raw-data ML (P-DF5). Increment 1 honors NOMEANSTRUCTURE only with an explicit `INFORMATION = EXPECTED` and otherwise rejects it, saying that Mplus would ignore it. `[x]` refers to means, intercepts or thresholds (`u$k`) by the variable's role. |

## Starts, fixes, equalities and labels

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| LB01 | D, 732–733, 736; P | `*` and `@` apply to the preceding item; after a range they apply to every element. A bare `*` frees at the default start. A bare `@` fixes at Mplus's default starting value, which depends on the data (0.05 for a factor variance, half the sample variance for a residual variance; P-LB1); the frontend rejects bare `@`. Starts go to `Starts`; Mplus's automatic starts are not emulated. |
| LB02 | D, 734–735, 744 | Only one parenthesized label or equality may appear per line, and it applies only to the items on its own line. Statements spanning lines can carry one label per line. The parser therefore keeps line segments inside statements. |
| LB03 | D, 742; P | The guide says text after an equality label on the same line is ignored; 9.1 instead reports an error for characters after the right parenthesis (P-LB5, P-LB6). The frontend rejects such tokens, matching 9.1. |
| LB04 | D, 734–735, 737; P | The same number in parentheses makes parameters equal. A number list pairs with right-hand items, `f BY y1-y4 (1-4)`, with the marker left fixed and unconstrained. A label list must also have one label per item: `(a2-a4)` for four items is an error, so the p. 744 example is wrong for 9.1, and the label paired with the fixed marker is not a usable label (P-LB4). A list cannot follow individually listed items. |
| LB05 | D, 738, 745; P | With a left-hand list, one number or label per left-hand element. With lists on both sides, either one group per left-hand element or a single list assigned row by row (`y1-y3 ON x1-x2 (p1-p6)` labels y1/x1, y1/x2, y2/x1, …; P-LB2). The p. 740 statement that a single list cannot be used does not hold in 9.1. |
| LB06 | D, 744–745; P | A name in parentheses is a parameter label; references are case-insensitive. Parameters sharing a label are equal (P-LB3). Labels exist only for free parameters: a label on a fixed parameter is an unknown label in MODEL CONSTRAINT (P-LB6 fixed_label_own_line). Label lists `(p1-p5)` expand by numeric suffix; a WITH label list over a square list fills the upper triangle row by row. |
| LB07 | D, 741–742; P | A parameter mentioned more than once takes its last specification, within a statement and across statements (P-MS1b); for equalities the overriding item must sit on its own line. |
| LB08 | D, 519 | Mentioning a parameter that is not free by default frees it at the default start unless `*value` or `@value` is given. |

## Default parameters (single group)

UG 515–519 lists the default parameters. In the conditional model the observed
independent (x) variables carry no parameters.

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| DF01 | D, 516–517, 723 | Means, variances and covariances of observed independent variables are not model parameters: the model conditions on them. Lower to the lab's fixed-x convention. |
| DF02 | D, 516 | With a mean structure, intercepts (and thresholds) of observed dependent variables are free. |
| DF03 | D, 516 | Means and intercepts of continuous latent variables are fixed at zero in single-group analysis. |
| DF04 | D, 517 | Variances and residual variances of continuous observed dependent variables and continuous latent variables are free. |
| DF05 | D, 518 | Covariances among continuous latent independent variables are free (random effects from ON/XWITH with `|` excepted). |
| DF06 | D, 518 | Covariances between continuous latent independent variables and observed independent variables are fixed at zero. |
| DF07 | D, 518; P | Covariances among observed variables that are neither dependent nor independent are fixed at zero. Such a variable (in USEVARIABLES but not in any relation) has a free mean and variance, with a warning (P-DF2). |
| DF08 | D, 518; P | Residual covariances among observed dependent variables are fixed at zero except free when neither influences any other variable and neither is a factor indicator, for continuous variables (and categorical or censored ones under weighted least squares). P-DF7: with `y1 y2 ON x1; y3 ON y1`, only y2–y3 is free. |
| DF09 | D, 518, 723; P | Residual covariances among continuous latent dependent variables are free when neither influences any variable other than its own indicators and neither indicates a second-order factor; otherwise zero (P-DF3). |
| DF10 | P | A final observed dependent variable that is not a factor indicator and a final latent dependent variable also get a free residual covariance (P-DF1: `f ON x1; y4 ON x1` frees f–y4). DF08 and DF09 therefore describe one family: all final dependent variables that are not factor indicators covary. |
| DF11 | D, 518 | Regression coefficients are zero unless mentioned. |
| DF12 | D, 719 | Residual variances of continuous factor indicators are free; residual covariances among factor indicators are zero. |
| DF13 | D, 726; P | ANALYSIS `MODEL = NOCOVARIANCES` fixes every covariance and residual covariance at zero; WITH frees selected ones (P-DF4). |

## Multiple groups

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| MG01 | D, 530 | One data set: GROUPING requests groups. Separate files: one `FILE (label) =` per group. Summary data: NGROUPS, labels g1, g2, …. Increment 2 supports GROUPING; separate files and summary groups belong to the data increment. |
| MG02 | D, 530; P | With one data set, the **first group is the group with the lowest value** of the grouping variable, whatever the label order (P-MG3: `(2 = b 1 = a)` puts a first). With separate files, the first FILE statement; with summary data, g1. The first group is the reference for latent means and scale factors. |
| MG03 | D, 538–539, 613; P | `GROUPING IS g (1 = male 2 = female)` maps codes to labels used by `MODEL label:`; unlisted codes leave the analysis (P-MG3). `g (101-200 225)` uses the values as labels; `g (2)` alone means two groups labelled g1, g2 by ascending value, and a count that does not match the data is an error (P-MG4). Only one grouping variable. |
| MG04 | D, 516–518, 531 | Defaults across groups: loadings of observed factor indicators equal; intercepts and thresholds of observed factor indicators equal; residual variances free and unequal; all structural parameters (factor variances, covariances, regressions, latent intercepts) free and unequal; latent means fixed at zero in the first group and free in the others. Intercepts of observed dependent variables that are not factor indicators are free and unequal. |
| MG05 | D, 518; P | Loading equality covers observed factor indicators only; second-order loadings are free and unequal across groups (P-MG1). |
| MG06 | D, 532–534 | `MODEL label:` states differences from the overall model. Mentioning a parameter there relaxes its cross-group equality (frees it for that group); including the first indicator in a group-specific BY frees its marker loading. The marker rule MS02 therefore applies only in the overall MODEL. |
| MG07 | D, 534–535; P | Equality numbers and labels are global identifiers. An overall-MODEL statement with a number or label holds in every group, so the parameter is equal across groups (and to others sharing the identifier). The same identifier inside a group section ties to it as well. Mentioning a parameter in a group section without an identifier releases it for that group (P-MG5). |
| MG08 | D, 536 | Group-specific `[f]` frees a latent mean (also in the first group); `[f@0]` fixes it. |
| MG09 | D, 517, 531, 537; P | Delta: scale factors of categorical factor indicators are fixed at one in the first group and free in the others; Theta: their latent-response residual variances likewise. A categorical dependent variable that is not a factor indicator stays fixed in every group (P-MG2), so p. 531 holds and p. 517 overstates. |
| MG10 | D, 541–546 | Configural, metric and scalar invariance models are defined per outcome type and parameterization; ANALYSIS `MODEL = CONFIGURAL METRIC SCALAR` requests them (see IV rules). |
| MG11 | D, 518–519; P | Only BY loadings of observed indicators are equal across groups. Regressions on a factor written with ON are unequal even for an indicator of another factor; that indicator's intercept stays equal, and a non-indicator's intercept is unequal (P-MG7). |
| MG12 | D, 516–519, 628–629 | KNOWNCLASS multiple groups follow mixture defaults (equal variances and slopes, reference in the last class), not GROUPING defaults. KNOWNCLASS is rejected (CL15). |
| MG13 | P | Settled by P-MG5; see MG07. |

## Measurement-invariance shortcuts

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| IV01 | D, 670–671 | `MODEL = CONFIGURAL`, `METRIC` or `SCALAR` sets up a multiple-group model from a MODEL that contains only first-order BY statements, with GROUPING. Metric by a loading fixed at one in every group or the factor variance fixed at one in one group (the first group under GROUPING). No partial invariance. Not available for mixed outcome types. |
| IV02 | D, 541–542 | Continuous outcomes. Configural: loadings, intercepts and residual variances free across groups; factor means zero in all groups. Metric: loadings equal, the rest free, factor means zero in all groups. Scalar: loadings and intercepts equal, factor means zero in one group and free elsewhere. With variance identification the variance is one in all groups (configural) or one in one group (metric, scalar). Residual variances are never equal (no strict setting). |
| IV03 | D, 542–546; v8.9–8.11 addendum PDF p. 6 | Categorical outcomes under weighted least squares: configural and scalar only. The guide's ordinal metric model with threshold pins (pp. 544–545) is superseded by the addendum, which treats ordinal metric as not identified. Increment 3. |
| IV04 | D, 670 | Several settings run several models and difference tests. A list is a model family plus a test plan: increment 2 accepts one setting and rejects a list until its result shape is decided. |
| IV05 | P | Mplus expands a shortcut into ordinary MODEL syntax (printed with `(MODEL)`, P-IV1): every model fixes factor means at zero in the overall MODEL; configural repeats the non-marker BY items and frees all indicator intercepts in every group section; metric frees only the intercepts; scalar is the default multiple-group model. Factor covariances are free per group. With variance identification, TECH1 shows variances fixed at one in both groups for configural and in the first group only for metric and scalar, although the printed commands show `f@1` in the overall MODEL: the printed text is a summary, TECH1 is authoritative. |

## Categorical outcomes (increment 3)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| CT01 | D, 604–608 | CATEGORICAL lists binary and ordinal dependent variables. Categories come from the data; at most 10; thresholds = categories − 1; each variable is recoded so its lowest observed category is 0. The threshold count therefore depends on data, as in magmaan's ordinal preparation. |
| CT02 | D, 729–730, 742 | Thresholds are `[u$k]`, lowest first; categorical outcomes have thresholds instead of intercepts; a threshold has the opposite sign of an intercept. |
| CT03 | D, 735; A, 748–751; P | Threshold equalities use one threshold index per statement, `[u1$1 u2$1 u3$1] (2)`. Ranges `[u1$1-u4$1]` (across variables) and `[u1$1-u1$2]` (within one variable) are accepted; bare `[u]` for a categorical variable is an error (P-CT1). |
| CT04 | D, 517, 519, 675 | Delta (default): scale factors fixed at one; latent-response residual variances cannot be freed. Theta: residual variances fixed at one; scale factors cannot be freed. |
| CT05 | D, 675 | Theta is required when a categorical dependent variable both influences and is influenced by another dependent variable or a factor. Under Delta such an input is rejected. |
| CT06 | D, 726 | WITH among categorical variables is allowed only with weighted least squares. |
| CT07 | P | A CATEGORICAL variable must be dependent: using it only as a predictor is an error (P-CT2). Every group must contain every category of each categorical variable (P-CT3). |
| CT08 | D, 570 | Summary data with categorical outcomes can only be a correlation matrix (data increment). |

## Growth (increment 4)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| GR01 | D, 746–747 | `i s q | y1@0 y2@1 y3@2 y4@3` names growth factors on the left and outcomes with time scores on the right. It equals BY statements with unit intercept loadings, time-score slope loadings and squared scores for q, with outcome intercepts fixed at zero and free growth-factor means. Lower to those rows. |
| GR02 | D, 747; P | An outcome without `@` has a free time score. A quadratic with free time scores is an error (too few fixed scores); four growth factors give a cubic with loadings t³ (P-GR3). |
| GR03 | D, 746–747; P | Defaults are overridden by mentioning parameters, before or after the `|` statement alike (P-GR3). |
| GR04 | D, 752–753 | Growth-factor variances and covariances are free. Means: free for continuous outcomes; for categorical outcomes and multiple-indicator growth the intercept-factor mean is zero (first group in multiple groups) and slope means free. |
| GR05 | D, 747–753 | Continuous outcomes: intercepts zero, residual variances free, residual covariances zero (A from DF12). Categorical: thresholds equal over time per index; Delta scale factor fixed at time one and free later; Theta residual variance likewise. |
| GR06 | P | Multiple-group categorical growth: the time-one scale factor (Delta) or residual variance (Theta) is free in later groups, and the I and S means are free there. Continuous multiple-group growth: outcome intercepts fixed at zero and growth means free in every group (P-GR2). |
| GR07 | D, 714, 753–757 | `|` with ON, PON, BY, XWITH, AT or a single bare variable is a random slope, random loading, interaction, individually varying time or random variance: R. `@` on the right identifies growth; a bare `i | y1` is ambiguous and is rejected. |

## Constraints, indirect effects and tests (increment 4)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| CN01 | D, 766–768 | MODEL CONSTRAINT holds explicit (`p1 = f(…)`) and implicit (`0 = f(…)`) constraints and inequalities (`>`, `<`) over MODEL labels and NEW parameters, with DEFINE's arithmetic and functions except absolute value. Lower onto existing defined-parameter, equality and constraint machinery; inequalities without an enforcing backend give unsupported-fit. |
| CN02 | D, 766–769; P | `NEW (c*.6)` declares a parameter with a start (default 0.5); lists allowed. A NEW name on a right-hand side adds a free parameter (P-CN1: one more free parameter, one less df); on a left-hand side it is a derived quantity (no change in df). Explicit and implicit forms of one constraint give the same model. |
| CN03 | D, 769–770 | DO loops (`DO (1,3) r# = p#/q#;`, nested with `$` and `%`) expand before lowering. LOOP and PLOT are plotting requests: E. |
| CN04 | D, 766 | With MODEL CONSTRAINT the default INFORMATION becomes OBSERVED: an estimation convention for the [preset](../backlog/speculative.md#mplus-estimation-convention-preset), not imported. |
| CN05 | D, 759–762; P | MODEL INDIRECT `y IND [m …] x` and `y VIA m x` become defined parameters (products of coefficients); a specific path keeps the written mediator order, and paths through a factor are allowed (P-CN2). Causal effects with `(values)` and MOD are R. |
| CN06 | D, 772 | MODEL TEST is a joint Wald test of `0 = …` restrictions: E (reported). |

## Data files (increment 5)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| DA01 | D, 564, 567–569 | Numeric ASCII data; records at most 10,000 characters. Free format (default): entries separated by comma, blank or tab; read until one value per NAMES variable, then continue with the next record. Fixed format: a Fortran-like FORMAT (`F`, `x`, `t`, `/`, repeat counts, implied decimals). FORMAT needs its own grammar production. |
| DA02 | D, 570–571, 539–540 | Summary data are free-format: lower-triangular or full covariance or correlation matrices, means and standard deviations, each type starting on a new record; groups follow one another with NOBSERVATIONS per group and NGROUPS. Map to magmaan's sample-statistics input. |
| DA03 | D, 601–603; P | MISSING: one non-numeric flag (`.`, `*`, `BLANK` with fixed format only) for all variables, or numeric flags per variable or ALL, with value ranges and comma-separated negatives. Flags compare with the value after FORMAT scaling: with F2.1 the field `99` reads as 9.9 and only the flag `9.9` matches; `-9` and `-9.0` are the same flag (P-DA2). |
| DA04 | D, 613 | Rows with an unlisted GROUPING value are excluded; the reader reports how many. |
| DA05 | D, UG 443, 548 | In the analysis, cases missing on an x variable are deleted and the remaining missingness is handled by FIML. These are estimation-sample rules for the preset, not reader behavior. |
| DA06 | P | Free format: an empty field between commas is an error; extra fields and observations wrapped over several records are accepted (P-DA1). |

## Probe list

Probes run through the local Mplus 9.1 Demo (at most six dependent and two
independent observed variables) by the probe lane (board TASK-58), which writes
derived summaries; the planner then updates the rules above. Each probe states
the inputs to run and the output entry that decides it. TECH1 is requested in
every probe; "rows" means the free/fixed pattern and parameter numbering in
TECH1's parameter specification. Data are synthetic; any data set that fits
the model works unless noted. Variable roles: y continuous dependent, x
independent, u ordinal (3 categories unless noted), g grouping.

**Increments 1–2 (needed before TASK-51):**

| Probe | Settles | Inputs | Read |
| --- | --- | --- | --- |
| P-LX1 | LX05 | `y1 ON x1; !* comment *! y2 ON x1;` on one line; and a block comment opened mid-line and closed on a later line | Accepted? Which statements survive (rows) |
| P-LX2 | LX03 | `ANALYSIS: TYPE = GENE;` (longer than stem), `TYPE = GEN;`, `USEVAR`, `USE` (3 letters), `ESTI = ML`, `MODEL = NOCOVAR;` | Accepted or error message |
| P-LX3 | LX06 | `TITLE: test anal: x` and `TITLE: test model: x` | Does the title end at the abbreviated or full command word? |
| P-LX4 | LX02, LX02a | A MODEL line of 91 and of 200 columns, statement split so truncation changes the model | Error, warning or truncation; rows |
| P-NM1 | NM02 | `NAMES = y08-y11 a1b-a3b z;` | Printed variable list in the summary |
| P-NM2 | NM01, NM01a | `NAMES = longname_1 longname_2 y3;` and two names sharing their first 8 characters | Error, warning, truncation or collision |
| P-NM3 | NM03, NM04 | (a) `NAMES = y1 x1 y2 y3; USEV = y1-y3;` (b) `USEV = y3 y1 y2 x1;` with `f BY y1-y3` (c) a range spanning a factor and an observed variable | (a) Is x1 analyzed? (b) Variable order in sample statistics and TECH1; marker chosen (c) Error or expansion |
| P-MS1 | MS02, LB07 | (a) `f BY y1 y2; f BY y3 y4;` (b) `f BY y1-y3; f BY y1;` (c) `f BY y1-y4*0.5 y2@1;` | Which loading is fixed at one in each case |
| P-MS2 | MS08, DF01 | `y1 ON x1 x2;` plus, in turn, `x1;`, `[x1];`, `x1 WITH x2;`, and data with 10% missing on x1 | Rows (does x1 gain variance, mean or covariance parameters; is x2 also brought in); N used |
| P-LB1 | LB01 | `f BY y1-y3; f@;` and `f BY y1* y2-y3; y1@;` | Fixed value printed in results; rows |
| P-LB2 | LB05 | `y1-y3 ON x1-x2 (p1-p6);` with `MODEL CONSTRAINT: NEW(d); d = p2 - p3;` | Accepted? Which coefficients p2, p3 label (row-major or not) |
| P-LB3 | LB06 | `y1 ON x1 (b); y2 ON x1 (b);` and `y1 y2 (v);` | Shared TECH1 parameter number? |
| P-LB4 | LB04, LB06 | `f BY y1-y4 (1-4);`, `f BY y1-y4 (a2-a4);`, `f BY y1-y4 (a1-a4);` (each with a constraint naming every label) | Accepted forms; which loading each label attaches to |
| P-LB5 | LB03 | `f BY y1-y4 (1) y5;` on one line | Is y5's loading in the model? |
| P-LB6 | LB06, NM01 | `f BY y1@1 (l1) y2-y3;` with `MODEL CONSTRAINT: NEW(r); r = l1;`; a mixed-case label referenced in another case; a 9-character label | Accepted? Errors |
| P-DF1 | DF10 | `f BY y1-y3; f ON x1; y4 ON x1;` | Is f WITH y4 free? |
| P-DF2 | DF07 | `USEV = y1-y4 x1; MODEL: f BY y1-y3;` | Rows for y4 and x1; warnings |
| P-DF3 | DF09 | `f1 BY y1-y2; f2 BY y3-y4; f3 BY f1 f2; f1 f2 ON x1;` and `f1 BY y1-y3; f2 BY y4-y6; f1 ON x1; f2 ON x1;` | Is the f1–f2 residual covariance free in each? |
| P-DF4 | DF13 | `MODEL = NOCOVARIANCES;` with `f1 BY y1-y3; f2 BY y4-y6; f1 f2 ON x1; y1 WITH y4;` | Rows: factor and residual covariances |
| P-DF5 | MS11, CL19 | `MODEL = NOMEANSTRUCTURE;` single-group and two-group CFA | df; mean rows per group |
| P-DF6 | CL17 | `TYPE = MISSING;`, `TYPE = MEANSTRUCTURE;`, `TYPE = GENERAL MISSING H1;` | Error, warning or silent acceptance; rows |
| P-MG1 | MG05 | Two groups: `f1 BY y1-y2; f2 BY y3-y4; f3 BY f1 f2;` | Are second-order loadings equal across groups? |
| P-MG3 | MG02, MG03 | Data with g in {1, 2, 3}: `GROUPING = g (2 = b 1 = a);` with a 3-indicator CFA | Group order in output; which group has latent mean fixed at zero; per-group N; is group 3 dropped |
| P-MG4 | MG03 | Values {5, 7}: `GROUPING = g (2);` and `g (3)` | Labels assigned; error for a wrong count |
| P-MG5 | MG07, MG13 | Two groups: overall `y1 (a);`; overall `y1 (a); y2 (a);`; overall `y1 (1);` with `MODEL g2: y1;`; overall `y1 (a);` with `MODEL g2: y2 (a);` | Rows: equal across groups? Released in g2? Error or tie for a reused label |
| P-MG7 | MG11 | Two groups: `f1 BY y1-y3; f2 BY y4-y5; y4 ON f1; y6 ON f1;` | Are the y4 and y6 slopes and y6's intercept equal across groups? |
| P-DF7 | DF08 | `y1 y2 ON x1; y3 ON y1;` | Which residual covariances among y1, y2, y3 are free |
| P-MG6 | MG06 | Two groups: overall `f BY y1-y3;`, group-specific `MODEL g2: f BY y1;` and `MODEL g2: f BY y2;` | Is y1 freed in g2? Is y2's equality released only in g2? |
| P-MG8 | MG06 | Overall CFA; `MODEL g2: y1 WITH y2;` | Can a group section add a parameter absent from the overall model? |
| P-MG9 | MG06, LB07 | Repeated `MODEL g2:` sections: disjoint mentions, conflicting loading fixes, then a bare loading mention | Cumulative application, last fix wins, and later bare mention releases a fix |
| P-MG10 | MG06 | `MODEL g1 g2: [y3];` | Are multi-label group section heads allowed? |
| P-MG11 | MG03 | `GROUPING = g (-1 = g1 2.5 = g2);` | Are fractional grouping codes accepted? |
| P-MG12 | MG03 | Negative integers, with integer and decimal spellings | Are negative codes and integral decimal spellings accepted? |
| P-IV1 | IV01–IV05 | `MODEL = CONFIGURAL METRIC SCALAR (MODEL);` on a two-factor, six-indicator, two-group continuous CFA (once with markers, once with `f@1`) | Generated MODEL commands; reference group; factor covariances |

**Later increments:**

| Probe | Settles | Inputs | Read |
| --- | --- | --- | --- |
| P-MG2 | MG09 | Two groups, Delta and Theta: `f BY u1-u3; u4 ON x1;` | Scale factor or residual variance of u4 free in group 2? |
| P-CT1 | CT03 | `[u1$1-u1$2];`, `[u1$1-u4$1] (1);`, `[u1];` | Accepted forms; TAU rows |
| P-CT2 | CT07 | `CATEGORICAL = u1; MODEL: y1 ON u1;` | Error or treatment |
| P-CT3 | CT07, CT01 | Two groups under WLSMV where category 3 of u1 occurs only in group 1 | Error or pooled thresholds |
| P-CT4 | CL18 | `ESTIMATOR = ML;` with (a) a categorical CFA (b) `u1 ON x1 x2;` | Integration reported? |
| P-GR1 | GR04, GR05 | `i s | u1@0 u2@1 u3@2 u4@3;` with Delta, Theta and ML | TAU, DELTA, THETA rows; intercept-factor mean |
| P-GR2 | GR06 | P-GR1 with two groups; also continuous `i s | y1@0 y2@1 y3@2 y4@3;` with two groups | Time-one scale factor or residual variance in group 2; growth means and intercepts per group |
| P-GR3 | GR02, GR03 | `i s q c | y1@0 … y5@4`; `i s q | y1@0 y2@1 y3 y4`; piecewise with shared `i`; `[y1-y4] (1)` before and after `|` | Rows |
| P-CN1 | CN02 | `p1 = p2**2 + p3**2;` versus `0 = p1 - p2**2 - p3**2;`; `NEW(c); p2 = p1 + c;`; `NEW(r); r = p1/q1;` | Free-parameter count and df |
| P-CN2 | CN05 | `y3 IND y2 y1 x1` versus `y3 IND y1 y2 x1`; `y IND f x` through a factor; continuous `y IND m x` | Effects printed |
| P-DA1 | DA06 | Free-format records with `1,,3`, extra fields and wrapped observations | N and sample means |
| P-DA2 | DA03 | `FORMAT = 3F2.1; MISSING = ALL (99);` versus `(9.9)`; `-9` versus `-9.0` | N and sample means |


## Probe results (Mplus 9.1 Demo)

Observed output from the standalone maintainer tool
[`regen_mplus_probes.R`](../../cpp/tests/tools/regen_mplus_probes.R) is summarized
below; [`probes.json`](../../cpp/tests/fixtures/mplus/probes.json) contains all
41 probes and 100 variants, their input sections, seeds, inventory IDs and parsed
output. Scratch inputs, data and original output stay under
`~/.cache/magmaan-logs/mplus-probes/` and are not checked in.

“Accepted” means no `*** ERROR` header, not successful estimation. Identification,
nonconvergence and other estimation messages are retained separately. Empty
fit-statistic arrays mean that the Demo did not print those statistics. TECH1
entries retain the printed row/column names, parameter numbers and lower
triangles; vectors use a `vector` row. Repeated truncated TECH1 names gain
`#1` suffixes to preserve distinct printed cells; summary/results names retain
the full printed spelling. Additional parameters have their own specification. Zero is a fixed-cell parameter number,
not its fixed value. Group and invariance-model identifiers are kept separately.
Synthetic estimation failures are reported without altering the probe syntax.
P-NM1 uses analyzed subsets to respect Demo caps; its NAMES statement is intact.
These observations do not resolve or change the inventory rules.

| Probe | Settles | Observed facts |
| --- | --- | --- |
| P-LX1 | LX05 | Inline: Y1 ON X1 and Y2 ON X1 printed (7 free parameters). Multiline: only Y1 ON X1 printed (5 free). |
| P-LX2 | LX03 | GEN, USEVAR and ESTI accepted. GENE: unrecognized TYPE setting; USE: unknown option; NOCOVAR: unrecognized MODEL setting. |
| P-LX3 | LX06 | Both title strings accepted; the printed titles retain `anal: x` and `model: x`; the later CFA has 9 free parameters. |
| P-LX4 | LX02, LX02a | 91 columns: accepted with warning that input exceeds 90 characters; all three indicators appear, 9 free parameters. 200 columns: error, no semicolon before new command; output echoes only F BY Y1 Y2 in the diagnostic. |
| P-NM1 | NM02 | Original NAMES accepted with USEV selecting Y08 Y09 Y10 Y11 Z; those names print in that order. Selecting explicit A1B A2B A3B Z errors on unknown A2B. Analyzed subset reduced to five/four variables for Demo caps; NAMES unchanged. |
| P-NM2 | NM01, NM01a | Both variants accepted; LONGNAME_1/LONGNAME_2 and ABCDEFGH1/ABCDEFGH2 print distinctly in the summary and MODEL RESULTS; TECH1 prints both names as the same eight-character prefix. |
| P-NM3 | NM03, NM04 | (a) X1 is an indicator between Y1 and Y2; order Y1 X1 Y2 Y3, marker Y1. (b) error expanding Y1-Y3. (c) F-Y3 unknown. |
| P-MS1 | MS02, LB07 | Two BY statements: Y1 fixed at 1, Y3 free (parameter 6). Repeated Y1: all three loadings free (4,5,6); SEs unavailable. Start/fix: Y1 free (5), Y2 fixed at 1. |
| P-MS2 | MS08, DF01 | Baseline N=450; X1 variance/mean mentions each give N=500, free X1 mean and variance, X2 fixed. X1 WITH X2 gives N=500 and both means, variances and covariance free (9 total). |
| P-LB1 | LB01 | Factor @: F variance fixed, printed 0.050. Residual @: Y1 residual fixed, printed 0.719; SEs unavailable. TECH1 numbers are zero for those fixed cells. |
| P-LB2 | LB05 | Accepted; BETA orders Y1/X1,Y1/X2,Y2/X1,Y2/X2,Y3/X1,Y3/X2 as 4–9. Printed D=-0.021 equals the rounded Y1/X2 minus Y2/X1 estimates. |
| P-LB3 | LB06 | Slopes share one parameter number and printed estimate. Bare Y1 Y2 (v) variances likewise share one number; 3 free parameters total. |
| P-LB4 | LB04, LB06 | (1-4) accepted with Y1 fixed at 1. (a2-a4): label-count mismatch. (a1-a4) plus constraint referencing all labels: unknown label A1. |
| P-LB5 | LB03 | Error: extra characters after the right parenthesis; no fitted Y5 row. |
| P-LB6 | LB06, NM01 | Fixed-label form errors on characters after parenthesis. Mixed-case reference and nine-character label accepted. Added variant with the label on its own line (planner): error, unknown parameter label L1 in MODEL CONSTRAINT. |
| P-DF1 | DF10 | Accepted; PSI[Y4,F]=13 (free). |
| P-DF2 | DF07 | Accepted; Y4 and X1 have free means and variances; warnings say they are uncorrelated with all other variables. |
| P-DF3 | DF09 | Second order: PSI[F2,F1]=0. First order: PSI[F2,F1] is free. |
| P-DF4 | DF13 | Accepted; factor residual covariance zero; explicit Y1 WITH Y4 free in THETA. |
| P-DF5 | MS11, CL19 | Both accepted. Single df=0, group df=4; NU intercepts remain free; group-2 F mean remains free despite NOMEANSTRUCTURE. |
| P-DF6 | CL17 | All three settings accepted; each has 9 free parameters and df=0 with free intercepts. |
| P-MG1 | MG05 | Accepted; second-order F2 BY F3 coefficient is BETA[F2,F3]=11 in G1, 22 in G2; first-order loadings share numbers. SEs unavailable; identification messages printed. |
| P-MG3 | MG02, MG03 | Order A then B, N=500 each (1000 total); unlisted group 3 absent. F mean fixed at zero in A, free in B. |
| P-MG4 | MG03 | (2) accepted, labels G1/G2 with N=500 each. (3) errors: G3 (2147483647) has 0 observations. |
| P-MG5 | MG07, MG13 | Named: Y1 variance number 3 in both groups. Named pair: all four variances number 3. Released numeric label: Y1 numbers 3/7. Reused a: G1 Y1 and G2 Y1/Y2 all number 3. |
| P-MG7 | MG11 | Accepted; Y4 ON F1 numbers 14/28 and Y6 ON F1 numbers 15/29. Y4 intercept number 12 shared; Y6 intercepts 13/27. Iteration limit reached; no fit chi-square printed. |
| P-DF7 | DF08 | Accepted; PSI[Y3,Y2]=9 free; Y1/Y2 and Y1/Y3 residual covariance cells zero. |
| P-MG6 | MG06 | Both accepted; G2 Y1 mention frees its loading while G1 Y1 stays fixed. G2 Y2 mention assigns a distinct loading number; Y3 remains shared. |
| P-MG8 | MG06 | Accepted: group-2-only residual covariance adds one free parameter (15 versus the default 14). |
| P-MG9 | MG06, LB07 | Repeated sections accepted cumulatively (16 free parameters for disjoint mentions). Later loading fix wins (0.8); a later bare loading mention releases the earlier fix. |
| P-MG10 | MG06 | Rejected: Unknown group name G1 G2 specified in group-specific MODEL command. Write one label per section. |
| P-MG11 | MG03 | Rejected: grouping value has to be an integer (2.5); Mplus suggests DEFINE. Frontend policy: recode to integers in R because DEFINE remains rejected. |
| P-MG12 | MG03 | Accepted: negative integer -1 and integral decimal spellings -1.0 and 2.0. Canonical frontend codes must be integer strings. |
| P-IV1 | IV01–IV05 | Planner added variant variance_both (`f1@1 f2@1`; the lane's `f1 f2@1` fixed only f2): accepted, free counts 38/34/30, df 16/20/24; TECH1 PSI fixed in G1, free in G2 for metric and scalar. Original variants: both accepted; generated commands and separate CONFIGURAL/METRIC/SCALAR TECH1 matrices printed. Marker variant: free counts 38/34/30, df 16/20/24. Variance variant prints identification messages; only metric fit statistics printed (35 free, df 19). |
| P-MG2 | MG09 | Both accepted. G2 U4 scale factor (Delta) or residual variance (Theta) has number 0; the corresponding U1–U3 cells are free. |
| P-CT1 | CT03 | Within-variable range accepted, both U1 thresholds free. Across-variable range accepted, first thresholds share number 1. Bare [U1] errors as an ignored statement. |
| P-CT2 | CT07 | Error: CATEGORICAL is for dependent variables only; U1 is independent in this model. |
| P-CT3 | CT07, CT01 | Error: group 2 does not contain all values of categorical U1. |
| P-CT4 | CL18 | Both accepted. Categorical CFA reports numerical integration (one dimension); observed regression reports zero integration dimensions. |
| P-GR1 | GR04, GR05 | All accepted. Threshold indices shared over time; I mean fixed zero, S mean free. Delta: U1 scale fixed, U2–U4 free. Theta: U1 residual fixed, U2–U4 free. ML prints fixed response residual cells and no chi-square. |
| P-GR2 | GR06 | Delta/Theta accepted; G2 time-one scale/residual free; I and S means free in G2. Continuous accepted: all observed intercepts fixed zero, both growth means free in both groups. Multiple-group ML errors: integration unavailable. |
| P-GR3 | GR02, GR03 | Cubic accepted, 19 free, df 1; cubic fixed loadings printed 0,1,8,27,64. Free-time form errors: insufficient fixed time scores. Shared-I piecewise accepted. Before/after intercept mentions both yield shared free NU number 1 and free I mean; SEs unavailable. |
| P-CN1 | CN02 | Explicit/implicit each 11 free, df 1 and same chi-square. NEW(c) form 12 free, df 0. Derived ratio form 9 free, df 3; R printed as an additional parameter. |
| P-CN2 | CN05 | Both mediator orders accepted; printed specific paths retain the requested order (forward estimate 0.001, reverse 0.000). Factor path prints indirect 0.012; continuous mediation prints indirect 0.010. |
| P-DA1 | DA06 | Empty comma field errors (non-missing blank; zero observations). Extra field and wrapped records accepted, N=500 and identical means (-0.023,0.058,-0.045). |
| P-DA2 | DA03 | All accepted, N=500. Y1 mean: 99 → 5.465; 9.9 → 4.972; -9 and -9.0 → 4.385. F2.1 fields are 99 or -9 in the first 50 rows; only flag 9.9 marks those 99 fields missing. |

## Corpus tally: input reader only (2026-10-03)

A throwaway driver ran `MplusParser::read()` (TASK-51.1, before MODEL
parsing) over every Mplus input in the textbook corpus: 2,438 files from case
sources, raw folders and zip archives, 1,178 with distinct content. No input
crashed or hung. 174 inputs passed the input-file level. Files rejected per
rule (a file counts once per rule): CL17 analysis types 340 (mixture 196,
two-level 112, random 84), CL27 MODEL CONSTRAINT/INDIRECT 277, CL16 DEFINE 239,
CL29 Monte Carlo 237, CL10 CATEGORICAL 231, CL15 designs, weights and mixture
options 219, NM03 196, LX02 98, CL18 Bayes 93, CL11 GROUPING 78, CL31 75, CL03
summary data 72, CL26 group sections 70, CL14 outcome types 68, CL22 56, CL13
case selection 51.

Findings fed back into TASK-51.2: all LX02 lines were TITLE or comment text
(rule refined above); NM03 hits were variables created by DEFINE or DATA
transformations, so the message must say so; CL31 also caught group sections
whose labels come from `FILE (label)` or NGROUPS; combined summary-data TYPE
settings (`CORRELATION MEANS STDEVIATIONS`) were misreported. The tally
suggests the order of value after increment 1: categorical outcomes, MODEL
CONSTRAINT/INDIRECT and multiple groups each block several hundred inputs;
DEFINE blocks 239.
