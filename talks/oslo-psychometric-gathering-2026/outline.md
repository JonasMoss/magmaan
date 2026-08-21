# How I made magmaan — talk outline

## Working contract

- **Venue:** Oslo Psychometric Gathering.
- **Assumed slot:** 15 minutes plus questions. The deck is deliberately modular;
  the cuts below produce a 10-minute version.
- **Audience:** psychometricians who probably use ChatGPT, including paid plans,
  but mostly have not used a repository-scale coding agent such as Codex.
- **Thesis:** coding agents have changed the feasible scale of individual
  psychometric methods development. They make implementation abundant, which
  makes architecture, specification, validation, and scientific judgment more
  important rather than less.
- **Evidence:** magmaan is the running case, not the nominal subject. The
  misspecification-robust ordinal-DWLS thread is the concrete scientific case.

## Narrative and timing

| Time | Slide | Job |
|---:|---|---|
| 0:00 | Title | State the historical claim: this project was not realistically available to one researcher a few years ago. |
| 0:30 | A guessing game | Ask how long and how many developers a codebase of this apparent scope required. Do not reveal immediately. |
| 1:15 | The reveal | Show generated repository metrics. Stress that canonical C++ excludes builds and vendored R copies. |
| 2:15 | The anti-flex | Say that volume is not evidence of correctness. Introduce the main thesis: the bottleneck moved from implementation to specification and falsification. |
| 3:00 | ChatGPT versus Codex | Explain the qualitative difference between an answer-producing chatbot and an agent operating a closed repository loop. |
| 4:00 | The development loop | Walk once around question → contract → implementation → oracle → properties → simulation. |
| 5:00 | AI amplifies architecture | Make the crucial engineering claim: modularity determines whether the agent compounds knowledge or compounds coupling. “Architecture is context engineering.” |
| 6:00 | Why C++ first | Explain the single statistical implementation, fast simulation/derivative loop, explicit types/failures, and thin bindings. |
| 7:00 | Small vocabulary, many compositions | Present the domain primitives and the core/frontier split. New methods must reuse an extension point, not fork the engine. |
| 8:00 | The fitted object is not the program | Contrast a small fit value plus explicit post-fit branches with a precomputed mega-object. Acknowledge that lavaan's shape is excellent for end users. |
| 9:00 | The test corpus | Show the generated counts and the validation ladder: focused algebra, finite differences, synthetic lavaan corpus, real-data parity, literature replications and simulations. |
| 10:15 | Five-day methods thread | Tell the 19–23 June estimated-weight story from the repository history. This is the worked example of the development loop. |
| 11:30 | Psychometric result | Explain why estimated DWLS weights contribute at leading order under misspecification and report the simulation result. Keep the mathematics verbal unless asked. |
| 12:30 | Composition compounds | Show how one score/information/restriction substrate yields many test families without independent reimplementation. This is a montage, not a test lecture. |
| 13:15 | What stays human | Estimands, contracts, oracle choice, counterexamples, interpretation, scope and stopping rules. |
| 14:15 | Close | Three lessons and the closing sentence. |

## Core messages by section

### 1. The scale changed

The opening numbers establish that something changed in the production function.
They must not be treated as quality metrics. Every rendered number comes from
`tools/repo_metrics.py`; definitions are on the appendix slide and in the CSV.

### 2. The tool changed

The audience already knows conversational AI. The unfamiliar part is that Codex
can read the working rules, inspect the repository, edit several layers, compile,
test, diagnose, revise, and preserve state. The unit of work is the closed loop,
not the code snippet.

Optional visual to add later: a 20–30 second screen recording of one real turn
that begins with a statistical question, edits C++ and a test, encounters a
failure, and ends with a focused green test. Use a recording, not a live demo.

### 3. Architecture determines whether acceleration compounds

The talk should be explicit about the magmaan philosophy:

- C++ is the statistical source of truth; R is a thin interface, not a second
  implementation.
- Values, local copies and `std::expected` make failures and data flow visible.
- A small vocabulary of free-function extension points replaces a framework of
  inheritance and hidden state.
- Core functions receive the smallest object they need.
- Point fitting and post-fit inference are separate operations.
- `core` marks compatibility and stability claims; per-domain `frontier` marks
  useful research surfaces with weaker API promises, not weaker testing.
- A frontier method must compose the existing engine. If it needs a parallel fit
  stack, the abstraction is missing or the method does not belong yet.

The lavaan comparison should remain respectful. A rich fitted mega-object and
string-based inspection are valuable end-user ergonomics. They are costly as the
internal architecture of a methods workbench because dependencies become hidden,
post-fit choices are easily precomputed or cached implicitly, and a new method
touches too much state.

### 4. Validation is the scientific product

The corpus has complementary jobs:

- algebraic/property and finite-difference tests catch local structural errors;
- 26 small synthetic models exercise the complete lavaan-shaped pipeline;
- real-data parity cases supply realistic conditioning and bookkeeping;
- checked-in fixtures make CI independent of an R installation;
- simulations and independent references support non-lavaan frontier claims;
- an oracle-defect claim requires an independent reference plus a violated
  first-principles property.

The line to land: **AI made plausible code abundant, so falsification became the
scarce work.**

### 5. Scientific case: estimated weights under misspecification

The short version:

1. Under ordinal DWLS, the weight matrix is estimated from the same data as the
   moments.
2. Under exact fit, the extra weight-influence term disappears; under fixed
   misspecification it is leading order.
3. An observed-Hessian sandwich can therefore still be incomplete if it treats
   the estimated weight as fixed.
4. The complete infinitesimal-jackknife implementation recovered the empirical
   SD to about 0.3% in the focused experiment and achieved 0.947 coverage versus
   0.927 for the observed-bread interval at nominal 0.95.
5. This was not merely coded: it was derived, finite-difference checked,
   compared with resampling, generalized across estimator families, simulated,
   documented, and exposed through R.

Primary local sources:

- `experiments/35-misspec-robust-se/report.qmd`
- `docs/research/notes/misspec_observed_bread.tex`
- `docs/architecture/roadmap.md`
- commits dated 19–23 June 2026

### 6. Human responsibility

Avoid both “the AI did everything” and “it is merely autocomplete.” The useful
division is:

- **Agent:** repository search, propagation, implementation, mechanical
  derivation checks, build/test loops, fixture comparison, experiment plumbing,
  documentation synchronization.
- **Researcher:** question, estimand, mathematical contract, evidence standard,
  diagnostic design, interpretation, priority, and the decision not to claim too
  much.

## Ten-minute cut

Skip “Composition compounds” and merge “Why C++ first” with “Small vocabulary,
many compositions.” Tell only the 19 June and 23 June endpoints of the case-study
timeline. Keep the validation slide intact.

## Material to add before the event

- Confirm the exact speaking slot and trim accordingly.
- Replace the generic Codex-loop description with one authentic screenshot or
  short recording.
- Decide whether the title should retain the explicit “2023” comparison.
- Add contact/GitHub details to the closing slide if the repository is public by
  the event.
- Re-render on the presentation machine; the metrics regenerate automatically.
