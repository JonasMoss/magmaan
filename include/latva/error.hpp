#pragma once

#include <cstdint>
#include <string>

#include "latva/source_span.hpp"

namespace latva {

struct ParseError {
  enum class Kind : std::uint8_t {
    UnexpectedChar,
    UnknownOperator,
    UnsupportedOperator,    // emitted by the parser, not the lexer
    UnterminatedString,
    MalformedNumber,

    // Parser-level kinds (filled in during P2):
    ExpectedLhs,
    ExpectedOperator,
    ExpectedRhsTerm,
    ModifierEvalFailed,
    GroupVecMismatch,
  };

  Kind        kind   = Kind::UnexpectedChar;
  SourceSpan  span   = {};
  std::string detail = {};

  friend bool operator==(const ParseError&, const ParseError&) = default;
};

// Errors raised while turning a FlatPartable into a complete LatentStructure.
struct PartableError {
  enum class Kind : std::uint8_t {
    BadGroupSpec,             // n_groups < 1, c(...) arity ≠ n_groups, group.label arity ≠ n_groups,
                              // or incompatible identification options (std_lv + effect_coding)
    UnknownLabelInConstraint, // referenced label has no source row
    InconsistentModifiers,    // e.g. start(0.5) AND fixed value on same row
    EmptyModel,               // no formula rows, no constraints
  };
  Kind        kind   = Kind::BadGroupSpec;
  std::string detail = {};

  friend bool operator==(const PartableError&, const PartableError&) = default;
};

// Errors raised while fitting a model. Either originating from the
// discrepancy/optimizer or from underlying numerical failures (Σ not
// PD, line-search runaway, etc).
struct FitError {
  enum class Kind : std::uint8_t {
    OptimizerNonConvergence,
    NonFiniteObjective,
    NonPositiveDefiniteSigma,
    NonPositiveDefiniteSample,
    LineSearchFailed,
    InvalidStartValues,
    NumericIssue,
  };
  Kind         kind        = Kind::OptimizerNonConvergence;
  std::string  detail      = {};
  int          iterations  = 0;
  double       f_value     = 0.0;

  friend bool operator==(const FitError&, const FitError&) = default;
};

// Errors raised while building a MatrixRep / running a ModelEvaluator.
struct ModelError {
  enum class Kind : std::uint8_t {
    UnsupportedRowKind,       // a partable row isn't representable yet (e.g. ~ in v0.5.1)
    UnknownVariable,          // LatentStructure row references a name not in any block
    NonPositiveDefinite,      // P6+: implied Σ has bad eigenvalues
    EmptyMatrix,              // partable has no estimable moments
    EigenAssertion,           // re-thrown from EIGEN_NO_EXCEPTIONS path
  };
  Kind        kind   = Kind::UnsupportedRowKind;
  std::string detail = {};

  friend bool operator==(const ModelError&, const ModelError&) = default;
};

// Errors raised by post-fit inference (SEs, χ², bootstrap, ...).
struct PostError {
  enum class Kind : std::uint8_t {
    InfoMatrixSingular,       // Fisher information not invertible (under-identified / collinear free params)
    BootstrapFailed,          // resampling step could not produce a usable replicate set
    NumericIssue,             // upstream numerical failure (non-PD Σ at θ̂, evaluator build, ...)
  };
  Kind        kind   = Kind::InfoMatrixSingular;
  std::string detail = {};

  friend bool operator==(const PostError&, const PostError&) = default;
};

}
