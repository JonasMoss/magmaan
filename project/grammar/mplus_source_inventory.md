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
earlier Demo run; C is supporting evidence, not a source.

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
| CL17 | ANALYSIS: TYPE | S | 651, 657–665. GENERAL (explicit or implied) accepted. BASIC requests descriptives only and is rejected for lowering. RANDOM, COMPLEX, MIXTURE, TWOLEVEL, THREELEVEL, CROSSCLASSIFIED and EFA are R. Pre-v5 settings such as `TYPE = MISSING` are U (probe P-DF6). |
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


## Lexical rules and file structure

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| LX01 | D, 13–14 | Ten commands. Each begins on a new line and is followed by a colon; commands may come in any order; DATA and VARIABLE are required. Options end with semicolons and several may share a line. The lexer tracks line starts; a command head is recognized only at a line start. |
| LX02 | D, 14 | Input records are at most 90 columns in v8; upper and lower case and tabs are allowed. See LX02a for later versions. A longer line is rejected with its span rather than truncated. |
| LX02a | A; U | No addendum through 9.1 mentions a change to the record limit (searched for line, column and length). Whether the 9.1 Demo still enforces 90 columns is U (probe P-LX4). |
| LX03 | D, 14, 563, 567 | Commands and options can be shortened to four or more letters. Option settings accept the complete word or the documented short form (bold in the option tables). Resolution must be unambiguous; an ambiguous prefix is rejected. |
| LX04 | D, 14, 598 | Keywords and names are case-insensitive. Canonical matching folds case; source spelling is retained for diagnostics and names. |
| LX05 | D, 14 | `!` starts a comment to the end of the line. A block of lines is commented out by starting its first line with `!*` and ending its last line with `*!`. Whether `!*` and `*!` must sit at line start/end is U (probe P-LX1). |
| LX06 | D, 563 | TITLE text can contain anything except a command word followed by a colon; colons after other words are allowed. |
| LX07 | D, 14 | IS, ARE and `=` are interchangeable in all commands except DEFINE, MODEL CONSTRAINT and MODEL TEST. List items are separated by blanks or commas. A hyphen denotes a list of variables or numbers; ALL denotes all variables where an option documents it. |

## Names and lists

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| NM01 | D, 598, 744 | Variable names and parameter labels are at most 8 characters in v8, begin with a letter and contain only letters, digits and underscore. See NM01a for later versions. Over-long names are rejected. |
| NM01a | A; U | No addendum through 9.1 mentions longer names. Whether 9.1 accepts, truncates or rejects longer names is U (probe P-NM2). |
| NM02 | D, 598 | In NAMES, a hyphenated range generates names: `y1-y5` from numeric suffixes, `itema-itemd` from letter suffixes. Generation rules for mixed or mismatched stems are U (probe P-NM1). |
| NM03 | D, 730–731 | In MODEL, a hyphenated range of observed variables follows the NAMES order when all NAMES variables are analyzed, otherwise the USEVARIABLES order. It never expands by numeric suffix. (lavaan's `mplus2lavaan()` gets this wrong.) |
| NM04 | D, 731 | Latent variables are ordered by their BY statements in MODEL order, followed by `|`-defined random effects in order. Ranges of latent variables follow this order. Whether a range may mix observed and latent variables is U (probe P-NM3). |
| NM05 | D, 731–732 | A list on the left of ON or WITH implies one statement per element; a list on the right is a list of variables. BY accepts lists on its right. |

## MODEL statements

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| MS01 | D, 713–715 | Statement forms: BY, ON, PON, WITH, PWITH, a bare list (variances or residual variances), `[list]` (means, intercepts, thresholds), `{list}` (scale factors), `|` (growth, random effects, interactions). Modifiers: `*` frees with an optional start, `@` fixes at an optional value, `(…)` equalities or labels. MODEL is optional only for EFA, LCA, a baseline model and TYPE = BASIC. |
| MS02 | D, 718, 532–533 | In `f BY y1 y2 …` the loading of the first variable after BY is fixed at one by default; the others are free with start one. `y1*` frees the first loading; `f@1` fixes the factor variance. When one factor has several BY statements, p. 718 ("the first variable after BY") and pp. 532–533 ("the first factor loading in each BY statement") can disagree: U (probe P-MS1). |
| MS03 | D, 719 | A factor may appear on the right of BY only after its own BY definition (second-order factors). Out-of-order use is an input error. |
| MS04 | D, 722–723 | `y ON x` frees a regression coefficient with start zero. Observed or latent variables may appear on either side. |
| MS05 | D, 725–726 | `a b PON c d` pairs elements; both sides need equal length. `a b WITH c d` crosses all left and right elements; PWITH pairs them. |
| MS06 | D, 726, 745; A | WITH frees covariances among continuous variables (residual covariances for dependent variables); for categorical or censored variables only with weighted least squares. `y1-y3 WITH y1-y3` yields the distinct unordered pairs and no variances (shown by the label example on p. 745). Self-pairs and duplicate orientations collapse to one parameter. |
| MS07 | D, 728 | A bare variable list refers to variances of independent and residual variances of dependent variables; mentioning frees them. Categorical observed variables have no variance parameter (Theta exceptions belong to increment 3). |
| MS08 | D, 723 | Means, variances and covariances of observed independent variables should not be mentioned, because the model conditions on them. What happens when they are mentioned is U (probe P-MS2). Until settled, increment 1 rejects such mentions. |
| MS09 | D, 719–722 | `(*label)` after BY defines ESEM factor sets; `~` gives target-rotation values. Both belong to ESEM and are rejected. |
| MS10 | D, 714–715, 723–725, 742–743 | `#` labels (latent classes, nominal categories, inflation parts, hazards), `%OVERALL%`, `%class%`, `%WITHIN%`, `%BETWEEN%` and the MODEL variants for mixtures, multilevel models and Monte Carlo belong to out-of-scope families and are rejected. |
| MS11 | D, 729–730 | Means, intercepts and thresholds are in the model by default; ANALYSIS `MODEL = NOMEANSTRUCTURE` removes them for TYPE = GENERAL. `[x]` refers to means of independent variables and of variables that are neither dependent nor independent, intercepts of continuous dependent variables, thresholds via `u$k`. |

## Starts, fixes, equalities and labels

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| LB01 | D, 732–733, 736 | `*` and `@` apply to the preceding item; after a range they apply to every element (`y7-y9*0.9`, `f1-f3@1`). `@` or `*` without a number fixes at or frees with the default value. Bare `@` is U (probe P-LB1); increment 1 rejects it until settled. Starts go to `Starts`; Mplus's automatic starts (UG 519–520) are not emulated. |
| LB02 | D, 734–735, 744 | Only one parenthesized label or equality may appear per line, and it applies only to the items on its own line. Statements spanning lines can carry one label per line. The parser therefore keeps line segments inside statements. |
| LB03 | D, 742 | Anything after an equality label on the same line is ignored by Mplus. The frontend rejects such tokens instead of dropping them; this is a deliberate, documented deviation. |
| LB04 | D, 734–735, 737 | The same number in parentheses makes parameters equal: `(1)` on several statements, or one `(1)` over a list. A list of numbers pairs with a list of right-hand items, `f BY y1-y5 (1-5)`; the count must equal the number of items, and the fixed marker takes no equality. A number list cannot follow individually listed items. For label lists p. 744 instead gives `f BY y1-y10 (z2-z10)`, nine labels for ten items: U (probe P-LB4). |
| LB05 | D/U, 738, 740, 745 | With a left-hand list, one number or label per left-hand element (`y1-y3 ON x (1 2 3)`); with lists on both sides, one group per left-hand element (`(1-2 3-4 5-6)`). For labels p. 740 says a single list cannot be used, whereas p. 745 expands `y1-y3 ON x1-x2 (p1-p6)` row by row. U (probe P-LB2). |
| LB06 | D, 744–745 | A name in parentheses is a parameter label, following the variable-name rules (NM01). Labels are used by MODEL CONSTRAINT, MODEL TEST and MODEL PRIORS. Label lists `(p1-p5)` expand by numeric suffix; WITH label lists over a square list fill the upper triangle row by row. Whether a repeated label also imposes equality is U (probe P-LB3). |
| LB07 | D, 741–742 | A parameter mentioned more than once takes its last specification (start, fix or free) within a statement; for equalities the overriding item must sit on its own line. Whether later separate statements also override is A from "mentioned in the MODEL command more than once" and should be confirmed (probe P-MS1 covers BY). |
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
| DF07 | D, 518 | Covariances among observed variables that are neither dependent nor independent are fixed at zero. Such a variable (in USEVARIABLES but not in any relation) has a free mean and variance (UG 729; C, Demo-confirmed). |
| DF08 | D, 518; C | Residual covariances among observed dependent variables are fixed at zero except free when neither influences any other variable and neither is a factor indicator, for continuous variables (and categorical or censored ones under weighted least squares). C: Demo-confirmed for `y1 ON x; y2 ON x`. |
| DF09 | D, 518, 723 | Residual covariances among continuous latent dependent variables are free when neither influences any variable other than its own indicators and neither indicates a second-order factor; otherwise zero. |
| DF10 | A, 518; U | No default frees a residual covariance between an observed and a latent dependent variable: DF08 and DF09 list separate families. The corpus translator's wording ("final latent and final observed") is ambiguous. U (probe P-DF1). |
| DF11 | D, 518 | Regression coefficients are zero unless mentioned. |
| DF12 | D, 719 | Residual variances of continuous factor indicators are free; residual covariances among factor indicators are zero. |
| DF13 | D, 726 | ANALYSIS `MODEL = NOCOVARIANCES` fixes every covariance and residual covariance among latent and observed variables at zero; WITH frees selected ones. |

## Multiple groups

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| MG01 | D, 530 | One data set: GROUPING requests groups. Separate files: one `FILE (label) =` per group. Summary data: NGROUPS, labels g1, g2, …. Increment 2 supports GROUPING; separate files and summary groups belong to the data increment. |
| MG02 | D, 530 | With one data set, the **first group is the group with the lowest value** of the grouping variable, not the first label declared. With separate files, the first FILE statement; with summary data, g1. The first group is the reference for latent means and scale factors, so this mapping must be exact. |
| MG03 | D, 538–539 | `GROUPING IS g (1 = male 2 = female)` maps codes to labels used by `MODEL label:`. Observations whose code is not listed are excluded from the analysis (a data-side effect that the frontend must report). Only one grouping variable. |
| MG04 | D, 516–518, 531 | Defaults across groups: loadings of observed factor indicators equal; intercepts and thresholds of observed factor indicators equal; residual variances free and unequal; all structural parameters (factor variances, covariances, regressions, latent intercepts) free and unequal; latent means fixed at zero in the first group and free in the others. Intercepts of observed dependent variables that are not factor indicators are free and unequal. |
| MG05 | D/U, 518 | Loading equality applies to regressions of an observed dependent variable that is a factor indicator on a continuous latent variable. Whether second-order loadings (latent indicators) are equal by default is not stated and reads as no (A); U (probe P-MG1). |
| MG06 | D, 532–534 | `MODEL label:` states differences from the overall model. Mentioning a parameter there relaxes its cross-group equality (frees it for that group); including the first indicator in a group-specific BY frees its marker loading. The marker rule MS02 therefore applies only in the overall MODEL. |
| MG07 | D, 534–535 | Equality labels in the overall MODEL hold across all groups (each labeled parameter equal in every group and to the others with the same number). Labels in a group-specific MODEL apply within that group only. Mentioning a parameter in a group-specific MODEL without a label removes it from the overall equality for that group. |
| MG08 | D, 536 | Group-specific `[f]` frees a latent mean (also in the first group); `[f@0]` fixes it. |
| MG09 | D/U, 517, 531, 537 | Delta: scale factors of categorical dependent variables are fixed at one in the first group and free in the others; Theta: residual variances of their latent responses likewise. p. 531 restricts this to categorical factor indicators; pp. 517 and 537 do not. U (probe P-MG2, increment 3). |
| MG10 | D, 541–546 | Configural, metric and scalar invariance models are defined per outcome type and parameterization; ANALYSIS `MODEL = CONFIGURAL METRIC SCALAR` requests them (see IV rules). |
| MG11 | D/U, 518–519 | Free regressions are unequal across groups except regressions of an observed factor indicator on a continuous latent variable. Whether `y ON f` written with ON, for y indicating another factor, counts as such a loading, and what holds for y indicating no factor, is U (probe P-MG7). |
| MG12 | D, 516–519, 628–629 | KNOWNCLASS multiple groups follow mixture defaults (equal variances and slopes, reference in the last class), not GROUPING defaults. KNOWNCLASS is rejected (CL15). |
| MG13 | U | Whether a named label in the overall MODEL ties a parameter across groups as an equality number does, and what reusing that label inside `MODEL label:` does (probe P-MG5). |

## Measurement-invariance shortcuts

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| IV01 | D, 670–671 | `MODEL = CONFIGURAL`, `METRIC` or `SCALAR` sets up a multiple-group model from a MODEL that contains only first-order BY statements, with GROUPING. Metric by a loading fixed at one in every group or the factor variance fixed at one in one group (the first group under GROUPING). No partial invariance. Not available for mixed outcome types. |
| IV02 | D, 541–542 | Continuous outcomes. Configural: loadings, intercepts and residual variances free across groups; factor means zero in all groups. Metric: loadings equal, the rest free, factor means zero in all groups. Scalar: loadings and intercepts equal, factor means zero in one group and free elsewhere. With variance identification the variance is one in all groups (configural) or one in one group (metric, scalar). Residual variances are never equal (no strict setting). |
| IV03 | D, 542–546; v8.9–8.11 addendum PDF p. 6 | Categorical outcomes under weighted least squares: configural and scalar only. The guide's ordinal metric model with threshold pins (pp. 544–545) is superseded by the addendum, which treats ordinal metric as not identified. Increment 3. |
| IV04 | D, 670 | Several settings run several models and difference tests. A list is a model family plus a test plan: increment 2 accepts one setting and rejects a list until its result shape is decided. |
| IV05 | U | Factor covariances, the reference group for means and the generated rows. Probe P-IV1 (version 9.1 can print the generated MODEL commands). |

## Categorical outcomes (increment 3)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| CT01 | D, 604–608 | CATEGORICAL lists binary and ordinal dependent variables. Categories come from the data; at most 10; thresholds = categories − 1; each variable is recoded so its lowest observed category is 0. The threshold count therefore depends on data, as in magmaan's ordinal preparation. |
| CT02 | D, 729–730, 742 | Thresholds are `[u$k]`, lowest first; categorical outcomes have thresholds instead of intercepts; a threshold has the opposite sign of an intercept. |
| CT03 | D, 735; A, 748–751; U | Threshold equalities use one threshold index per statement, `[u1$1 u2$1 u3$1] (2)`. `[u1$1-u4$1]` means the first threshold of u1 to u4 (A, from growth tables). Within-variable ranges and bare `[u]` are U (probe P-CT1). |
| CT04 | D, 517, 519, 675 | Delta (default): scale factors fixed at one; latent-response residual variances cannot be freed. Theta: residual variances fixed at one; scale factors cannot be freed. |
| CT05 | D, 675 | Theta is required when a categorical dependent variable both influences and is influenced by another dependent variable or a factor. Under Delta such an input is rejected. |
| CT06 | D, 726 | WITH among categorical variables is allowed only with weighted least squares. |
| CT07 | U | A CATEGORICAL variable used as a predictor, and category coverage that differs across groups (probes P-CT2, P-CT3). |
| CT08 | D, 570 | Summary data with categorical outcomes can only be a correlation matrix (data increment). |

## Growth (increment 4)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| GR01 | D, 746–747 | `i s q | y1@0 y2@1 y3@2 y4@3` names growth factors on the left and outcomes with time scores on the right. It equals BY statements with unit intercept loadings, time-score slope loadings and squared scores for q, with outcome intercepts fixed at zero and free growth-factor means. Lower to those rows. |
| GR02 | D, 747; U | An outcome without `@` has a free time score. The quadratic loading for a free time score, four-factor (cubic) forms and piecewise forms sharing `i` are U (probe P-GR3). |
| GR03 | D, 746–747; U | Defaults are overridden by mentioning parameters after the `|` statement; mentions before it are U (probe P-GR3). |
| GR04 | D, 752–753 | Growth-factor variances and covariances are free. Means: free for continuous outcomes; for categorical outcomes and multiple-indicator growth the intercept-factor mean is zero (first group in multiple groups) and slope means free. |
| GR05 | D, 747–753 | Continuous outcomes: intercepts zero, residual variances free, residual covariances zero (A from DF12). Categorical: thresholds equal over time per index; Delta scale factor fixed at time one and free later; Theta residual variance likewise. |
| GR06 | D/U, 749–751 | Multiple-group categorical growth tables disagree about time-one scale factors in later groups (probe P-GR2). |
| GR07 | D, 714, 753–757 | `|` with ON, PON, BY, XWITH, AT or a single bare variable is a random slope, random loading, interaction, individually varying time or random variance: R. `@` on the right identifies growth; a bare `i | y1` is ambiguous and is rejected. |

## Constraints, indirect effects and tests (increment 4)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| CN01 | D, 766–768 | MODEL CONSTRAINT holds explicit (`p1 = f(…)`) and implicit (`0 = f(…)`) constraints and inequalities (`>`, `<`) over MODEL labels and NEW parameters, with DEFINE's arithmetic and functions except absolute value. Lower onto existing defined-parameter, equality and constraint machinery; inequalities without an enforcing backend give unsupported-fit. |
| CN02 | D, 766–769; U | `NEW (c*.6)` declares a parameter with a start (default 0.5); lists allowed. A NEW name on a right-hand side is an extra free parameter; on a left-hand side a derived quantity. The deciding rule and explicit versus implicit classification are U (probe P-CN1). |
| CN03 | D, 769–770 | DO loops (`DO (1,3) r# = p#/q#;`, nested with `$` and `%`) expand before lowering. LOOP and PLOT are plotting requests: E. |
| CN04 | D, 766 | With MODEL CONSTRAINT the default INFORMATION becomes OBSERVED: an estimation convention for the [preset](../backlog/speculative.md#mplus-estimation-convention-preset), not imported. |
| CN05 | D, 759–762; U | MODEL INDIRECT `y IND [m …] x` (all or one specific indirect path) and `y VIA m x` become defined parameters (products of coefficients). Order of several mediators, paths through BY links and the choice between conventional and causal effects are U (probe P-CN2). Causal effects with `(values)` and MOD are R. |
| CN06 | D, 772 | MODEL TEST is a joint Wald test of `0 = …` restrictions: E (reported). |

## Data files (increment 5)

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| DA01 | D, 564, 567–569 | Numeric ASCII data; records at most 10,000 characters. Free format (default): entries separated by comma, blank or tab; read until one value per NAMES variable, then continue with the next record. Fixed format: a Fortran-like FORMAT (`F`, `x`, `t`, `/`, repeat counts, implied decimals). FORMAT needs its own grammar production. |
| DA02 | D, 570–571, 539–540 | Summary data are free-format: lower-triangular or full covariance or correlation matrices, means and standard deviations, each type starting on a new record; groups follow one another with NOBSERVATIONS per group and NGROUPS. Map to magmaan's sample-statistics input. |
| DA03 | D, 601–603 | MISSING: one non-numeric flag (`.`, `*`, `BLANK` with fixed format only) for all variables, or numeric flags per variable or ALL, with value ranges and comma-separated negatives. Implied-decimal interaction is U (probe P-DA2). |
| DA04 | D, 613 | Rows with an unlisted GROUPING value are excluded; the reader reports how many. |
| DA05 | D, UG 443, 548 | In the analysis, cases missing on an x variable are deleted and the remaining missingness is handled by FIML. These are estimation-sample rules for the preset, not reader behavior. |
| DA06 | U | Empty fields, extra fields and wrapped records in free format (probe P-DA1). |

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
