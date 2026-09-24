#pragma once

#include <vector>

#include "magmaan/expected.hpp"
#include "magmaan/parse/flat_partable.hpp"
#include "magmaan/spec/partable.hpp"  // CompositeInfo

namespace magmaan::spec {

// Result of Henseler-Ogasawara composite expansion.
//
// LIFETIME CONTRACT: `flat`'s *unmodified* rows / modifiers / constraints keep
// `string_view`s into the ORIGINAL `FlatPartable` passed to
// `expand_composites` — only the synthesized H-O rows own their storage (in
// `flat.source_text`). The returned `flat` must therefore not outlive the
// input. `spec::build` consumes it strictly build-locally, which satisfies
// this; no other caller is expected.
struct CompositeExpansion {
  parse::FlatPartable        flat;        // `<~` rewritten to `=~` / `~~`
  std::vector<CompositeInfo> composites;  // one per user composite, in order
};

// Rewrite every `<~` (Op::Composite) statement into a reflective
// Henseler-Ogasawara sub-model — the original Schuberth/Rademaker/Henseler
// (2021) specification, not the Yu (2023) refinement.
//
// A composite `C <~ x1 + ... + xK` becomes K mutually orthogonal synthetic
// latents — 1 emergent (`C`, the composite of interest) and K-1 excrescent
// nuisance latents — with the K indicators as reflective `=~` indicators of
// all K. The synthesized rows are:
//   * a K×K loading block: the emergent column is free except for one loading
//     fixed to 1 (scale); excrescent column j (1-based) has its first j-1
//     loadings fixed to 0 (cascading-triangular pattern) and one fixed to 1;
//   * `xi ~~ 0*xi` zero residuals for every indicator (the composite is an
//     exact weighted sum — no measurement error);
//   * `~~ 0*` covariances pinning every excrescent latent uncorrelated with
//     every other latent in the model;
//   * free `~~`-self variances for the emergent and excrescent latents.
// The emergent latent then participates in the structural model exactly like
// an ordinary latent; excrescent latents never appear in any `~` path.
//
// Errors: CompositeTooFewIndicators (K < 2); CompositeOverlap (two composites
// share an indicator, a composite name is reused as an indicator, or a name is
// both a composite and a `=~` factor).
partable_expected<CompositeExpansion>
expand_composites(const parse::FlatPartable& flat);

}  // namespace magmaan::spec
