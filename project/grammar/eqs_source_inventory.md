# EQS 6 SEM syntax: source inventory

Target: the documented linear SEM model language, with its parameter meanings
preserved in magmaan's Jöreskog/LISREL model contract. This inventory records
source evidence and validation requirements; it does not extend the implemented
[normative grammar](eqs_grammar.ebnf) or claim EQS numerical parity.
Implementation state belongs in the [frontend contract](eqs.md) and
[roadmap](../architecture/roadmap.md); accepted work belongs in the
[active backlog](../backlog/todo.md#api-and-r-boundary).

## Source and reading map

Bentler, P. M. (2006), *EQS 6 Structural Equations Program Manual*, Multivariate
Software. [Public PDF](https://www3.nd.edu/~kyuan/courses/sem/EQS-Manual6.pdf).
The reviewed file has 422 physical PDF pages and SHA-256
`5c1afc86ab68da993edfca12aef1838fedf23e310d622ae9fe5e7f5fa320d899`.
Arabic printed page **p** is physical PDF page **p + 4**; PDF viewers use
one-based physical pages, whereas screenshot tools may use zero-based indices.
References below use printed pages.

The source PDF and text extracted with `pdftotext -layout` are cached in ignored
`external/refs/eqs/`. Per-page text is in `pages/pdf_NNN.txt`. Third-party source
material stays untracked. The rules below are original summaries, not copied
manual passages. They describe this manual version; later EQS releases may
differ.

| Printed pages | Physical PDF pages | Purpose |
| --- | --- | --- |
| 2–3, 21–58 | 6–7, 25–62 | Linear-system interpretation, error paths, scaling and worked models; p. 49 explicitly mixes measured indicators and structural predictors |
| 13–15 | 17–19 | Section inventory; brief description of `/DEFINE` |
| 59–69 | 63–73 | Lexical organization and model-relevant `/SPECIFICATIONS` declarations; separate data/estimator options |
| 70–75 | 74–79 | Labels, explicit equations, variances, covariance lists, variable selection and aliases |
| 76–77 | 80–81 | `/MODEL` expansion, identification defaults and `/RELIABILITY` |
| 78–82 | 82–86 | Parameter references, equalities, cross-group `SET` and explicit bounds |
| 83–88, 101–102 | 87–92, 105–106 | Data sections, technical controls, output boundary and `/END` |
| 170, 172–173 | 174, 176–177 | Pattern-matrix naming and `SET` vocabulary; consult before equating EQS sets with lavaan equality families |
| 197–202 | 201–206 | Multiple model segments and correlation-analysis declarations |
| 203–209, 216 | 207–213, 220 | `V999`, intercepts, indirect mean effects and parameter restrictions |
| 227–237 | 231–241 | Group-specific latent intercepts and cross-group mean restrictions |
| 241–266 | 245–270 | Growth examples as ordinary equations/means/constraints, including cohort groups |
| 267–274 | 271–278 | Multilevel declarations and distinction from ordinary group segments |

## Rule inventory

**D** means directly stated or demonstrated in the manual. **A** means a
mathematical consequence or proposed adapter requirement. **U** means the
inspected source does not settle exact EQS behavior. A rule can be well specified
even when its magmaan adapter is missing.

### Lexical rules and names

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| L01 | D, 59–60, 102 | Slash section headers have documented three-letter abbreviations; `!` comments end at the line boundary; statements normally end at `;`. `/END` ends a segment. Preserve line boundaries and source spans. |
| L02 | D, 59–60, 72 | The manual describes an 80-column input convention, headers on their own lines, and multiline equations with coefficient/variable units kept together. Current snippet parsing is more permissive. Record any deliberate convenience extensions rather than presenting them as tested EQS acceptance. |
| L03 | D/U, 71 | Numeric identities use V/F/E/D and indices 1–999; the manual says leading zeros are suppressed without settling acceptance of a typed `V01`. Normalize accepted identities before collision, alias and range resolution; confirm leading-zero input separately. `V999` is reserved, not an ordinary data column. |
| N01 | D, 70, 75 | `/LABELS` assigns one-to-eight-character names to V/F identities. Defined names can replace numeric IDs in model statements; E/D retain numeric identifiers. Labels are aliases during resolution and names in `LatentNames`, not parameter-equality labels. |
| N02 | D/U, 59, 73, 75 | Aliased endpoints denote ranges in underlying numeric identity order. The manual warns about `TO` after labels shorter than eight characters. A demonstrated hyphenated label also creates lexical ambiguity with subtraction/ranges. Whole declared aliases must be resolved before interpreting internal punctuation. Exact collision/precedence behavior needs a small runtime check. |

### Explicit equations and independent moments

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| E01 | D, 71–72 | Only V/F occur on equation left sides; appearing there makes a variable dependent even if it predicts another variable. Each dependent variable has one equation, and equation order is arbitrary. E/D are independent variables in this terminology. |
| E02 | D, 71–72 | A bare predictor has fixed coefficient one; a number alone fixes its coefficient; `number*` declares a free coefficient with that start; `*` declares a free coefficient with no supplied start. These map separately to fixed structure, a free parameter, and `Starts`. |
| E03 | D, 72 | A self-predictor stops the run. Although the manual discourages a repeated RHS predictor, it specifies retaining its last occurrence. The current frontend rejects repeated predictors; the documented recovery is a concrete extension target. |
| E04 | D, 72; example 49 | A free error loading is explicitly allowed. Measured variables can both receive factor paths and predict observed/latent variables. The current unit-error and indicator-regression restrictions are implementation boundaries, not a definition of SEM. |
| E05 | A from 2–3, 71–72 | Independent E/D predictors need not be reduced to one owner residual if their coefficients or relationships are more general. Shared/multiple/errorless constructions follow from the linear-system representation, but exact parser acceptance of unusual spellings should be checked before an EQS acceptance claim. Preserve original parameters when constructing an augmented LISREL representation. |
| V01 | D, 73 | Variances belong to independent variables, including E/D. The manual says missing independent variances are repaired internally. Its start-generation rules are separate from explicit start hints and do not require reproducing EQS's start algorithm. |
| V02 | D, 73 | Same-family numeric ranges use dash or `TO`; comma lists specify several assignments. Subsequent specific assignments override earlier range-generated values. General duplicate-assignment precedence is not fully specified by this example. |
| V03 | D, 73–74 | Observed selection is described through equations and variances, in original numeric column order. The covariance section also says its variables need variance declarations. Current covariance-only selection is therefore a convenience whose exact EQS counterpart needs confirmation. |
| P01 | D, 74–75 | Covariances refer to independent variables; absent pairs are zero. A range generates distinct off-diagonal pairs, and later specific pairs override generated values. Pair order is immaterial to the covariance value. |
| P02 | D, 76 | Shorthand explicitly names all independent-variable covariance families: `VV,VF,FF,VE,FE,EE,VD,FD,ED,DD`. Error/predictor and E/D covariance exclusions in the initial frontend need a richer lowering contract, not a new estimator family. |

### `/MODEL` and `/RELIABILITY`

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| M01 | D, 76 | `/MODEL` replaces explicit equations/variances/covariances; combining these forms makes EQS stop. `(left) ON (right)` expands directed paths over the Cartesian product, and repeated left-side `ON` statements combine into one equation. |
| M02 | D, 76 | A list before `ON`, `COV` or `PCOV` needs parentheses; a list after the keyword can omit them. Lists mix individual identities and ranges. Resolve ranges/aliases before expansion. |
| M03 | D, 76 | Omitting the assignment suffix means a free parameter for `ON`, `COV` and `PCOV`, but **fixed one** for `VAR`. Unmentioned variances are free. Fixed/free/start rules must survive expansion. |
| M04 | D, 76 | `COV` accepts an independent-moment family selection, all distinct pairs within a list, or all unequal cross-list pairs. `PCOV` instead pairs equal-position elements of equally sized lists. Never substitute a Cartesian expansion for paired covariances. |
| M05 | D/U, 77; example 49 | Shorthand inserts E/D residuals and fixes the first factor coefficient to one when all its coefficients are free and its variance is not fixed. The p. 49 example demonstrates indicator-list order affecting the marker. Exact interactions with repeated statements, later explicit fixes and structural factor paths need targeted checks. Explicit equations must not acquire these shorthand-only fixes automatically. |
| M06 | D, 76 | `EQM` requests a generated refit file; it changes output behavior, not the model. Recognizing it must not imply file-writing support. |
| R01 | D/U, 77 | `/RELIABILITY` with `SCALE` describes a one-independent-factor model shorthand and also requests a reliability calculation. Its printed expansion uses several factor numbers, contradicting that description. The intended corrected expansion is an inference until checked against EQS output or independent authoritative evidence. |

### Parameter restrictions

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| C01 | D, 78–79 | `(dependent,predictor)` identifies a directed coefficient; `(variable,variable)` a variance; a pair identifies a covariance. Resolve EQS references to original parameter identities after all declarations/expansions. Coefficient direction must never be canonicalized like a symmetric covariance. |
| C02 | D, 78–80 | `/CONSTRAINTS` accepts equality chains and general linear expressions with numeric constants. Referenced parameters must be free; supplied starts should meet restrictions. Adapt to existing constraint expressions/equality groups and linear-constraint resolution. Fixed-reference diagnostics and start handling need explicit contracts. |
| C03 | D, 81–82 | `/INEQUALITIES` accepts `GT,GE,LT,LE` and mathematical one-/two-sided forms. EQS executes strict forms as closed bounds. Preserve requested spelling while specifying how bounds enter the shared parameter-space contract; unsupported fitting must fail explicitly. |
| C04 | D, 81–82 | Automatic nonnegative variance and fixed-variance covariance limits are EQS execution conventions. They are distinct from written restrictions. Full model-language support requires honoring explicit bounds; it does not authorize importing all EQS optimization defaults. |
| C05 | D/U, 79–81, 172–173 | Cross-group `SET` expands restrictions selected by EQS pattern-matrix names, with forced-free exceptions. Those names reflect dependent/independent roles, not simply observed/latent roles. Reuse an explicit classification and reference map; do not map them indiscriminately to lavaan's broad equality families. |

The p. 172–173 tables define the families from which parameter sets are drawn:

| Pattern | Classification | Documented submatrix spellings |
| --- | --- | --- |
| PHI, prefix P | Independent-variable moments, lower-triangle family order | `VV,FV,FF,EV,EF,EE,DV,DF,DE,DD` |
| GAMMA, prefix G | Dependent V/F outcome, independent V/F/E/D predictor | `VV,VF,VE,VD,FV,FF,FE,FD` |
| BETA, prefix B | Dependent V/F outcome, dependent V/F predictor | `VV,VF,FV,FF` |

Prefixing the table names gives references such as `PEE`, `GVF` and `BVF`.
The same section documents whole-pattern selectors/abbreviations in LM tests,
where they select default subsets rather than every entry. Do not assume all
LM-specific selector/grouping behavior applies unchanged to `/CONSTRAINTS SET`.
Likewise, the `/MODEL COV` family list in M04 uses its own displayed pair order;
acceptance of transposed spelling variants needs confirmation. In a mean model,
V999 is an independent V predictor, so an observed intercept can fall in GVV
alongside other V-on-independent-V coefficients. A broad lavaan intercept or
loading equality option cannot reproduce arbitrary EQS SET selection.

### Means and group segments

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| I01 | D, 203–206 | `V999` denotes constant one and has zero variance/covariances. A path from it is an intercept, not an observed-column regression. `ANALYSIS=MOMENT` requires it in the model. `/MEANS` supplies data summaries; it does not define intercept parameters. |
| I02 | D, 203–206 | An independent variable's nonzero mean can be expressed by a new equation with a `V999` intercept and centered residual. A dependent variable's implied mean includes indirect paths; an intercept and an implied mean are different quantities. Map to the existing Nu/Alpha mean machinery without adding an observed V999 dimension. |
| I03 | D, 216, 227–237 | Intercepts use the same parameter references, e.g. `(F1,V999)`. Reference-group fixed-zero and other-group free latent intercepts are explicit parameter choices, not defaults inferred from group position. Preserve omitted versus explicitly fixed-zero paths in source/projection metadata. |
| G01 | D, 197–198 | Group models are successive segments ending at `/END`; the first `/SPECIFICATIONS` gives `GROUPS=n`. Model statements and starts are group-local and may differ. Adapt to existing group/block representation rather than concatenating unowned rows or requiring identical source models. |
| G02 | D, 80–81, 197–198 | Cross-group parameter references add a one-based segment index; cross-group constraints/`SET` appear in the final segment. Earlier segments can have their own local constraints. Resolve all models before expanding cross-group restrictions. |
| G03 | D, 201–202, 235–237 | `ANALYSIS=CORRELATION` is a target declaration, not a synonym for covariance input. Multigroup means combine segment-local V999 paths and cross-group restrictions. Language representation and estimator/inference availability are separate gates. |

### Schema and execution boundaries

| ID | Evidence | Rule and implementation consequence |
| --- | --- | --- |
| W01 | D, 60–69 | A minimal model envelope needs semantic declarations such as group count, analyzed moment kind and `CATEGORY`. `METHOD`, missingness, weights, case count, input matrix type and file format must be classified as analysis/data settings. Recognizing them must not silently change the selected magmaan route. |
| W02 | D, 83–88, 101–102 | Matrix/means/standard-deviation payloads, numerical controls, output requests and file saving belong outside model lowering. Full SEM model syntax does not require complete EQS job execution. Unsupported execution requests must be reported or returned separately by an explicitly extract-only API. |
| S01 | D, 241–266 | Growth examples use ordinary paths, fixed/free coefficients, mean terms and restrictions. Cohort examples additionally use group segments. Do not add a special growth estimator merely to accept their model statements. |
| S02 | D/U, 13–15, 267–274 | `MULTILEVEL`, `CLUSTER` and HLM-specific transfer instructions carry level/schema semantics; ordinary segment group indices cannot stand in for levels. `/DEFINE` is only briefly described in the reviewed manual, without a complete production. These are recorded separately under the existing scope/capability gates, not silently reduced to single-level models. |

## Shared-model requirements

Existing core support includes group blocks, Nu/Alpha means, equality groups and
general constraint expressions. Much of the new work is EQS reference resolution
and adapter validation. The remaining general-equation cases need deliberate
builder/matrix representation work. A feature must account for:

- `LatentStructure`: declared variable roles, all original estimands, fixed
  values, directed coefficients, independent moments and resolved restrictions.
- `LatentNames`: original identities, aliases, group/level identities and a
  stable mapping from EQS parameter references to core rows. Generated internal
  coordinates must not lose the user's parameter identities. Aliases that are
  legal EQS names but invalid lavaan identifiers need a safe rebuild/projection
  strategy; emitting them unquoted into lavaan syntax is insufficient.
- `Starts`: explicit hints indexed after parameter merges/constraint resolution;
  absence of a hint continues to select magmaan's documented start policy.
- `matrix_rep`/evaluation/fit: every accepted parameter and restriction either
  contributes correctly or produces an explicit unsupported-capability error.
  No successful parse may silently drop a free error path or written bound.

For example, collapsing `a*E1` into a residual variance changes the parameter
from `(a, Var(E1))` to `a^2 Var(E1)`. Even when the implied covariance agrees,
that collapse loses individual estimands, starts and restrictions. Prefer an
exact augmented LISREL construction when the general equation system requires
explicit error predictors. This is an adapter-design requirement, not a new
estimator or a completed representation proof.

## Validation without an EQS installation

Each feature gets an independently specified expected model, rather than using
the frontend's own output as its oracle. Compare named rows, free/fixed states,
starts, aliases, blocks and restrictions; where appropriate, compare an
independently written lavaan model and numerical implied moments. Record the
source rule and whether the expected result is documented or derived.

| Fixture family | Distinguishing check |
| --- | --- |
| Identity/lexical | V1/V01 handling, upper/lower-case commands, 999 boundary, aliases with punctuation, multiline units, comments and convenience extensions |
| Explicit recovery | Last repeated predictor versus last range override; self-predictor failure; covariance-only variable selection left unresolved until checked |
| `/MODEL` paths | Cartesian ON, combined equations, explicit fixed coefficient, fixed factor variance and residual generation |
| `/MODEL` moments | Bare VAR fixed-one versus unmentioned free variance; within-list, cross-list and paired COV; independent-family selections |
| Original error parameters | Fixed/free nonunit error coefficient with fixed variance; error/predictor covariance; shared/multiple independent-error graph as a derived construction |
| Restrictions | Ordered coefficient references, symmetric moments, equality chains, signed linear restrictions, unknown/fixed references and explicit bounds |
| Means | Absent/fixed/free V999 paths, independent-variable mean equation, direct versus indirect mean effects, no data column or variance for the constant |
| Groups | Different segment models/starts, group-qualified references, SET selections and exceptions, reference-group fixed-zero intercept |
| Growth/schema | Fixed time loadings through ordinary rows; explicit categorical/level metadata and unsupported-fit diagnostics |

A first-principles check for general equations can partition dependent variables
as `z` and independent variables as `u`, with
`z = A z + C u + a`. For invertible `I-A`, let `T=(I-A)^(-1) C` and
`K=(I-A)^(-1)`. If `Cov(u)=Phi`, selection matrix `H` picks observed variables
from `[z;u]`, and `R=[T;I]`, then

`Sigma = H R Phi R' H'`

and

`mu = H [K a + T mu_u; mu_u]`.

These equations follow by solving the linear system and taking moments. They
give a validation oracle independent of parser/partable output. Add coefficient
perturbation checks so equivalent covariance matrices cannot conceal lost or
merged original parameters. Identification and inferential claims remain
separate questions.

## Small EQS checks that would close remaining uncertainties

An installation is useful for these focused probes, not a prerequisite for the
well-specified language work. Inputs plus setup/log expansions are usually enough;
exact numerical fit output is not the target. Request version/build metadata and
retain only permitted derived expectations as checked-in fixtures.

| Probe | Why it matters | Evidence to request |
| --- | --- | --- |
| SCALE one-factor expansion | Printed `/RELIABILITY` expansion contradicts its description | Generated equations/variances and actual factor identities |
| Marker selection | Several possible meanings of first coefficient after combined/reordered ON statements | Generated fixed/free paths before fitting |
| Alias resolution | Hyphenated aliases, aliases ending in digits, range endpoints, label case/collisions and forward use | Accepted/rejected inputs and resolved identities |
| Repeated declarations | Only later specific overrides of generated ranges are documented clearly | Final coefficient/variance/covariance declarations for both orders |
| Variance repair and observed selection | Missing-variance repair and covariance-variable /VAR requirement overlap | Selected observed order and repaired declarations, with/without equation use |
| SET exceptions | Exact chains for absent/fixed parameters and exclusions in three groups are not fully enumerated | Generated group-qualified restrictions |
| Constant handling | Explicit zero moment declarations and omitted/fixed-zero V999 paths may be normalized differently | Model expansion and source-parameter listing |
| Legacy lexical conventions | Leading-zero input and 80-column/header/continuation rules may be historical or release-specific | Acceptance diagnostics for bounded input variations |
| Unusual error graphs | General linear-system derivation does not prove exact EQS parser acceptance | Setup result for multiple/shared/errorless and cross-family error paths |
| HLM transfer | `/DEFINE` lacks a complete production in this manual | Additional authoritative language documentation before an adapter is scoped |

Do not label an inferred repair as EQS-verified. Until these probes are available,
document bounded uncertainties and keep unsupported ambiguous cases explicit.
