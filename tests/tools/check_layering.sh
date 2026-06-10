#!/usr/bin/env bash
#
# Dependency-layering checker. The repo is a layered DAG; leaves are sinks.
# Leaves: each papers/<name>/, each experiments/<NN>-*/, benchmarks/, tests/.
# A leaf consumes only lower tiers (core = include/ src/ r-package/, plus the
# shared experiments/_support and build artifacts) and never references a
# sibling leaf. Core never references any leaf. Retired experiments live frozen
# under experiments/_archive/<NN>-*/ and are treated as one archive zone (a path
# token there truncates to experiments/_archive): an archived file may reference
# within the archive but still not a paper, a live experiment, or tests. See
# AGENTS.md 'Dependency layering' and experiments/AGENTS.md / papers/AGENTS.md.
#
# Static, dependency-free (POSIX-ish bash + awk/grep/find). No R, no build.
# Scans code files only (comments stripped), so prose/comments never trip it.
# papers/* are gitignored and may be absent (CI): their internal rule then
# no-ops; the enforceable core (tracked code never references a leaf) is intact.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || { echo "check_layering: cannot cd to repo root" >&2; exit 2; }

DOC="see AGENTS.md 'Dependency layering'"
SELF_SCRIPT="tests/tools/check_layering.sh"
status=0
fail() { printf '%s:%s: %s; %s\n' "$1" "$2" "$3" "$DOC"; status=1; }

# --- paper R-package denylist (hardcoded knowns + any present DESCRIPTIONs) ----
deny_list() {
  printf '%s\n' snllscontinuous snllsconstrained pairwiserobustsem \
    alphaomegawald ugammafast compositeml rawstandardizedalpha \
    hpolychoricssem ordinalsnlls
  if [ -d papers ]; then
    for d in papers/*/r-package/DESCRIPTION; do
      [ -f "$d" ] && sed -n 's/^Package:[[:space:]]*//p' "$d"
    done
  fi
}
DENY_RE="$(deny_list | grep -vix magmaan | sort -u | paste -sd'|' -)"
[ -n "$DENY_RE" ] || DENY_RE='__none__'

# Tokens that denote a leaf reference. Path forms catch source()/file.path()/
# #include/sys.source; the quoted-"papers" form catches indirection like
# repo_path("papers", ...); the pkg:: form catches paper namespaces.
TOKEN_RE="papers/[A-Za-z0-9._-]+|experiments/[A-Za-z0-9._-]+|[\"']papers[\"']|benchmarks/|tests/|(${DENY_RE}):::?"

# --- classify a file into a zone (sets Z and SELF) ----------------------------
classify_zone() {
  SELF=""
  case "$1" in
    include/*|src/*|r-package/*)         Z=CORE ;;
    justfile|CMakeLists.txt|cmake/*)     Z=ORCH ;;
    experiments/_support/*)              Z=SUPPORT ;;
    experiments/_archive/*)              Z=EXP; SELF="experiments/_archive" ;;
    experiments/*) Z=EXP;   SELF="experiments/$(printf '%s' "$1" | cut -d/ -f2)" ;;
    papers/*)      Z=PAPER; SELF="papers/$(printf '%s' "$1" | cut -d/ -f2)" ;;
    benchmarks/*)                        Z=BENCH ;;
    tests/*)                             Z=TESTS ;;
    *)                                   Z=OTHER ;;
  esac
}

# --- decide whether one (file,line,token) is a violation ----------------------
check_token() {
  local f="$1" ln="$2" tok="$3" t
  # strip surrounding quotes
  t="${tok%\"}"; t="${t#\"}"; t="${t%\'}"; t="${t#\'}"
  case "$t" in
    experiments/_support*) return ;;                       # the one shared sibling
    papers/*|papers)
      [ "$Z" = PAPER ] && { [ -z "${SELF##$t}" ] || [ "$t" = papers ] ; } && return
      fail "$f" "$ln" "$Z leaf references paper '$t'" ;;
    experiments/*)
      [ "$Z" = EXP ] && [ "$SELF" = "$t" ] && return
      fail "$f" "$ln" "$Z leaf references experiment '$t'" ;;
    benchmarks/)
      # benchmarks/ is shared harness experiments may consume (like _support);
      # only papers/core/tests may not reach into it.
      { [ "$Z" = BENCH ] || [ "$Z" = ORCH ] || [ "$Z" = EXP ]; } && return
      fail "$f" "$ln" "$Z leaf references benchmarks/" ;;
    tests/)
      { [ "$Z" = TESTS ] || [ "$Z" = ORCH ]; } && return
      fail "$f" "$ln" "$Z leaf references tests/" ;;
    *)                                                     # paper package ns
      [ "$Z" = PAPER ] && return
      fail "$f" "$ln" "references paper package '$t'" ;;
  esac
}

# --- main: scan every code file ----------------------------------------------
strip_comments() {  # $1 = iscpp(0/1); strips comments + build/ artifact paths
  awk -v iscpp="$1" '{
    l=$0
    if (iscpp=="1") { sub(/\/\/.*/,"",l); gsub(/\/\*[^*]*\*\//,"",l); sub(/\/\*.*/,"",l) }
    else           { sub(/#.*/,"",l) }
    gsub(/build\/[^ \t"():]*/,"",l)
    print l
  }'
}

while IFS= read -r f; do
  [ "$f" = "$SELF_SCRIPT" ] && continue
  classify_zone "$f"
  [ "$Z" = OTHER ] && continue
  case "$f" in *.cpp|*.hpp|*.h) iscpp=1 ;; *) iscpp=0 ;; esac
  while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    check_token "$f" "${hit%%:*}" "${hit#*:}"
  done < <(strip_comments "$iscpp" <"$f" | grep -noE "$TOKEN_RE")
done < <(
  # Invariant 6 ("reports read only from their own results/") is intentionally
  # NOT mechanically enforced: */results/* is excluded and .qmd files are never
  # scanned, because the checker's design rule is code-files-only so prose can
  # never false-positive. Report inputs are reviewed manually.
  find . -type f \( -name '*.R' -o -name '*.r' -o -name '*.cpp' -o -name '*.hpp' \
      -o -name '*.h' -o -name 'CMakeLists.txt' -o -name '*.cmake' -o -name '*.sh' \
      -o -name 'justfile' \) \
    -not -path './build/*' -not -path './external/*' -not -path './third_party/*' \
    -not -path './corpus/*' -not -path './.git/*' -not -path '*/.quarto/*' \
    -not -path '*/results/*' \
  | sed 's#^\./##' | sort
)

if [ "$status" -eq 0 ]; then
  echo "check_layering: OK (no cross-leaf references)"
fi
exit "$status"
