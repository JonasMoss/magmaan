# Bug: lavaan-syntax parser rejects two standard constructs

**Status:** open — found via the snlls-constrained Newsom textbook corpus.
**Location:** `src/parse/parser.cpp` (the lavaan-syntax parser).
**Impact:** 88 of 142 Newsom longitudinal-SEM models fail `model_spec()`.
This is *not* a corpus defect — the models are verbatim lavaan extracted from
Newsom's own R companion scripts and run unmodified in lavaan.

Filed as a standalone note rather than appended to `docs/backlog/todo.md`
because that file was mid-edit by another agent; fold a one-line pointer into
the backlog index when convenient.

## Gap 1 — fixed intercept/mean `var ~ <number>` (~42 models)

magmaan accepts `x ~ 1` (free intercept) but rejects any other bare numeric:

```
w1vst1 ~ 0
# magmaan parse error [ExpectedRhsTerm]:
#   bare numeric literal is only valid as the intercept form `~ 1`
```

lavaan semantics, confirmed via `lavaanify(..., meanstructure = TRUE)`:

| syntax      | meaning                          |
|-------------|----------------------------------|
| `x ~ 1`     | intercept/mean **free**          |
| `x ~ 0`     | intercept/mean **fixed at 0**    |
| `x ~ 0*1`   | intercept/mean fixed at 0        |

`~ 0` is core longitudinal-SEM identification syntax — latent means fixed to
zero at a reference wave, marker-intercept fixing, and so on. The parser
should accept `var ~ <numeric>` as a fixed intercept/mean, with `~ 1` keeping
its special "free intercept" meaning to match lavaan.

## Gap 2 — parenthesized modifier labels `(label)*var` (~37 models)

```
dep1 =~ NA*cesdna1 + (lambda1)*cesdna1 + (lambda2)*cesdpa1
# magmaan parse error [ExpectedRhsTerm]: expected RHS term, got LParen
```

lavaan accepts `(lambda1)*x` as equivalent to `lambda1*x`. magmaan's parser
only takes the bare-label form.

## Minor

- Constraint RHS beginning with a numeric token with no following space:
  `nu1 == 0- nu2 - nu3`.
- One `start(...)` modifier wanting a numeric literal.

## Reproduce

```r
magmaan::model_spec("f =~ x1 + x2 + x3\n x1 ~ 0", meanstructure = TRUE)
```

Corpus exemplars: `corpus/textbook-corpus/raw/newsom/models/ex1_2a.lav` (Gap 1),
`corpus/textbook-corpus/raw/newsom/models/ex14_2a.lav` (Gap 2).
