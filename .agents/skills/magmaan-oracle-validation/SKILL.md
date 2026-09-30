---
name: magmaan-oracle-validation
description: Investigate magmaan component parity failures against lavaan or regenerate checked-in oracle fixtures. Use for parser/partable/estimate/SE/test mismatches and independently proved oracle defects, not for choosing ordinary-user defaults or generic literature review.
---

# magmaan oracle validation

Work from the repository root and applicable C++/R instructions. Start with the
specific failing test or feature, model/data, oracle options/version, and expected
component contract. If some are missing, inspect the test and its fixture before
asking for inputs. Use installed package output as the oracle; implement from
formulas, never by porting upstream source.

## Investigate a mismatch

1. Read `project/validation/oracle-defects.md` for an existing applicable case
   and the relevant roadmap contract. Compare identical data, estimator,
   identification, mean structure, groups, constraints, information and scaling
   conventions. Inspect the fixture's provenance and documented tolerances.
2. Reproduce the smallest relevant failure with `just test-area <area> [regex]`
   or the affected R check. Locate the first diverging component: parse,
   lavaanify/triple, partable projection, matrix representation, moments,
   criterion/derivatives, estimates or inference. Align parameters by
   `(lhs, op, rhs, group)`/labels rather than raw row positions.
3. Assume a magmaan defect. Fix the canonical component and add a meaningful
   focused numerical/property or golden gate. For grammar changes, edit the
   normative EBNF before parser code and fixtures. Do not mask a partable
   mismatch in R or relax tolerance merely to pass.
4. Only exempt oracle output after satisfying the ledger's proof standard:
   independent reference plus a violated first-principles property; robust/scaled
   tests also require the stated target-regime calibration. Record scope,
   version, reproduction, proof and replacement gate. Record investigated
   non-defects in that ledger's appropriate section.

## Regenerate fixtures

Read [references/fixtures.md](references/fixtures.md) when adding/changing
fixtures or changing the pinned oracle version. Regeneration supports an
intentional oracle update; it is not a repair for an unexplained discrepancy.

## Deliver

Produce the reproduction, component-level explanation, scoped canonical fix or
proved exemption, and appropriate regression evidence. Review affected fixture
diffs and record why they changed. Refresh vendors after C++ changes and update
maintained state/backlogs only when their contracts or remaining work change.
C++ gates use checked-in fixtures without R; R integration checks may use live
lavaan. State which checks ran and any unavailable oracle/dependency. Follow
the root's explicit-path commit rule.
