# magmaan model grammar — v0

Source of truth for the parser. Read this in conjunction with
[`grammar.ebnf`](grammar.ebnf) (the formal spec) and
[`lexer.md`](lexer.md) (token classes and lexer-level rules).

## Status

Version: **v0** (parser-first phases P1–P4 of the project plan).
Compatible with lavaan **0.6-22.2560** semantics for the supported
subset of operators.

## Source-of-truth contract

1. The grammar in `grammar.ebnf` is normative. If the parser disagrees
   with the EBNF, the parser is wrong.
2. Every production name in `grammar.ebnf` corresponds 1:1 to a function
   in `src/parse/parser.cpp` (or `src/parse/lexer.cpp` for lexical
   productions). Functions carry a comment of the form
   `// production: name = ...` immediately above the function body.
3. A grammar-coverage test (lands in P1) iterates the production names
   in the EBNF and asserts each appears in at least one fixture's token
   stream or parsed output. An unreferenced production fails CI — that
   forces fixtures to keep up with grammar changes.
4. When the grammar changes, the order is: edit `grammar.ebnf` first,
   then update the parser/lexer to match, then regenerate fixtures via
   `tests/tools/regen_oracle.R`. PR review checks that the EBNF was touched
   when behavior changed.

## v0 scope — what is in

| Operator | Meaning | Example |
|---|---|---|
| `=~` | latent definition (loadings) | `f =~ x1 + x2 + x3` |
| `~`  | regression | `y ~ x1 + x2` |
| `~~` | (residual) covariance / variance | `x1 ~~ x2`, `f ~~ f` |
| `\|` | ordinal threshold | `x1 \| t1 + t2` |
| `~*~` | latent response scale (delta parameterization) | `x1 ~*~ 1*x1` |
| `~ 1` | free intercept/mean (parser-level, not a single token) | `y ~ 1` |
| `~ n` | fixed intercept/mean at numeric value `n` | `y ~ 0` |
| `:=` | defined parameter | `indirect := a * b` |
| `==` | equality constraint | `a == 0.5` |
| `<` | inequality (upper bound) | `a < 0.5` |
| `>` | inequality (lower bound) | `a > 0` |

| Modifier form | Meaning | Example |
|---|---|---|
| `n * x` | fix parameter to `n` | `1*x1`, `0*x2` |
| `lbl * x` | label parameter `lbl` (equality via shared label) | `a*x1 + a*x2` |
| `(lbl) * x` | parenthesized label parameter `lbl` | `(a)*x1 + (a)*x2` |
| `"lbl" * x` | quoted label | `"my_label" * x1` |
| `start(v) * x` | starting value | `start(0.5) * x1` |
| `v ? x` | starting-value shorthand | `0.5 ? x1` |
| `NA * x` | explicitly free (default) | `NA * x1` |
| `c(m1, m2, ...) * x` | per-group modifier (one entry per group) | `c(1, 1.2) * x1` |

Multi-line formulas continue across `Newline` tokens when a `+` floats
on either side. Comments use `#` or `!` to end of line. Statements are
separated by `;` or `Newline`.

## v0 scope — what is out

The following operators tokenize normally but produce
`ParseError::Kind::UnsupportedOperator` with a span pointing at the
operator:

| Operator | Why deferred |
|---|---|
| `\|~` | Ordinal slope syntax — not needed for the first ordinal LS path. |

The following modifiers are not parsed and produce
`ParseError::Kind::ModifierEvalFailed` with a helpful message:

- `efa("set")` — exploratory factor analysis blocks.
- `prior(...)` — Bayesian priors.
- `rv(...)` — random-variable / random-slope labels (multilevel).
- `lower(...)` / `upper(...)` — explicit bounds (use `<` / `>` instead in v0).

The block syntax `group:` / `level:` / `block:` is rejected at the
parser level: an identifier immediately followed by `:` produces
`ParseError::Kind::UnsupportedOperator` naming the keyword.

## Walking the grammar

A `model` is a sequence of `statement`s separated by `;` or newlines.
A `statement` is one of:

- A **formula**: `lhs_list operator rhs_list`. The LHS is one or more
  identifiers joined by `+`; in v0 the multi-LHS form is only meaningful
  for `~` (regression). The RHS is one or more `rhs_term`s joined by `+`.
  Threshold formulas use ordinary RHS identifiers such as `t1`, `t2`, and
  response-scale formulas use the `~*~` operator.
- A **constraint**: `expr (==|<|>) expr`.
- A **define-param**: `identifier := expr`.

A `rhs_term` is an optional `modifier`, then `*` or `?`, then an
identifier. The `?` separator forces start-value interpretation of the
modifier; `*` interprets the modifier per its kind (numeric → fixed,
identifier → label, `NA` → free, `c(...)` → per-group, `start(...)` →
start value).

The intercept forms are detected by the parser when `regression` is followed
by a single numeric RHS with no `+` continuation. Bare `y ~ 1` means a free
intercept/mean; other bare numeric values such as `y ~ 0` are equivalent to
fixed `y ~ 0*1`. This is **not** a separate operator at the lexer level — it
stays context-free.

The expression sublanguage is standard precedence-climbing: `+ -` < `* /`
< `^` (right-associative) < unary `+ -`. No function calls in v0; if a
methods developer needs `min(a, b)` inside a `:=`, that's a future
extension and a major-version bump (the AST is closed).

## Examples

```
# A 1-factor CFA on three indicators
visual =~ x1 + x2 + x3

# Multi-line continuation
visual =~ x1 + x2 +
          x3 + x4

# Labels for equality across loadings
visual =~ a*x1 + a*x2 + a*x3

# Fix the marker indicator, free the rest, set a start
visual =~ 1*x1 + NA*x2 + start(0.7)*x3

# Multi-group, per-group fixed value
visual =~ 1*x1 + c(0.8, 1.2)*x2 + x3 ; group: 1 # ← `:` rejected in v0

# Defined indirect effect
indirect := a * b

# Equality and inequality
a == b
a > 0
```

## Open questions deferred

- **O1**: arithmetic inside `c(...)` modifier atoms (e.g. `c(0, 0.5+0.5, 1)`).
  Currently rejected. Recommendation: hold the line in v0; revisit if
  the corpus needs it.

## Plan to keep this honest

- Every parser/lexer function carries `// production: name = ...` matching
  the EBNF.
- A `tests/property/grammar_coverage_test.cpp` (P1) walks the EBNF and
  fails CI if a production isn't referenced by at least one fixture.
- When `grammar.ebnf` changes, the change is reviewed by reading the
  diff against the parser code in the same PR.
