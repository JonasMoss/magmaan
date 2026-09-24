#!/usr/bin/env bash
#
# Tracked-content guard: magmaan's history holds magmaan and nothing else.
# See AGENTS.md 'What belongs in this repository'. Checks the index
# (`git ls-files`), so it sees exactly what a commit would publish:
#
#   1. Only known top-level entries. A new top-level folder is a layout
#      decision; add it here and to AGENTS.md deliberately.
#   2. Nothing inside the ignored mount points (papers/, private/, external/,
#      external/) except their tracked convention files.
#   3. No archives, office documents, serialized R/Python objects, or build
#      products anywhere, and no IDE/session state.
#   4. No PDFs (working notes are maintained independently).
#   5. No file over 1 MB unless explicitly allowlisted below.
#
# Static and dependency-free (bash + git + awk). Fast; run in CI and `just check`.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT" || { echo "check_tracked_files: cannot cd to repo root" >&2; exit 2; }

DOC="see AGENTS.md 'What belongs in this repository'"

TOP_ALLOWED=" .github .gitignore AGENTS.md CLAUDE.md LICENSE README.md THIRD_PARTY_NOTICES.md benchmarks cpp experiments external justfile papers private project r-package "

# Tracked convention files inside otherwise-ignored mount points.
MOUNT_ALLOWED=" papers/AGENTS.md papers/CLAUDE.md papers/STYLE.md \
private/README.md external/README.md "

# No PDFs belong to this repository.
PDF_ALLOWED_RE='^$'

MAX_BYTES=1048576
SIZE_ALLOWED_RE='^project/assets/logo/[^/]+\.png$'

BINARY_RE='\.(zip|rar|7z|tar|tgz|gz|bz2|xz|rdata|rda|rds|sav|dta|por|xls|xlsx|doc|docx|ppt|pptx|key|odt|ods|pyc|o|a|so|dylib|dll|exe)$'
STATE_RE='(^|/)(\.rproj\.user|__pycache__|\.ipynb_checkpoints|\.quarto|\.vscode|\.idea)/|(^|/)(\.rdata|\.rhistory|\.ds_store)$'

# One pass over the index: "<blob size>\t<path>" for every staged file, so the
# check sees exactly what the next commit would publish.
paths="$(mktemp)"; trap 'rm -f "$paths"' EXIT
git -c core.quotePath=off ls-files -s | awk -F'\t' '{ split($1, a, " "); print a[2] "\t" $2 }' >"$paths"

cut -f1 "$paths" | git cat-file --batch-check='%(objectsize)' |
  paste - <(cut -f2 "$paths") |
  TOP_ALLOWED="$TOP_ALLOWED" MOUNT_ALLOWED="$MOUNT_ALLOWED" \
  PDF_ALLOWED_RE="$PDF_ALLOWED_RE" MAX_BYTES="$MAX_BYTES" \
  SIZE_ALLOWED_RE="$SIZE_ALLOWED_RE" BINARY_RE="$BINARY_RE" \
  STATE_RE="$STATE_RE" DOC="$DOC" \
  awk -F'\t' '
    # Patterns come through ENVIRON: awk -v would expand the backslash escapes.
    BEGIN {
      top_allowed = ENVIRON["TOP_ALLOWED"]; mount_allowed = ENVIRON["MOUNT_ALLOWED"]
      pdf_ok = ENVIRON["PDF_ALLOWED_RE"]; max_bytes = ENVIRON["MAX_BYTES"] + 0
      size_ok = ENVIRON["SIZE_ALLOWED_RE"]; binary_re = ENVIRON["BINARY_RE"]
      state_re = ENVIRON["STATE_RE"]; doc = ENVIRON["DOC"]
    }
    function fail(path, why) { printf "%s: %s; %s\n", path, why, doc; bad = 1 }
    {
      size = $1; f = $2; low = tolower(f)
      top = f; sub(/\/.*/, "", top)
      if (index(top_allowed, " " top " ") == 0)
        fail(f, "unexpected top-level entry \x27" top "\x27")
      if (f ~ /^(papers|private|external)\// && index(mount_allowed, " " f " ") == 0)
        fail(f, "tracked file inside an ignored mount point")
      if (low ~ binary_re)
        fail(f, "archive, office document, serialized object, or build product")
      if (low ~ state_re)
        fail(f, "IDE or session state")
      if (low ~ /\.pdf$/ && f !~ pdf_ok)
        fail(f, "PDF in the public repository (references belong in external/refs/)")
      if (size + 0 > max_bytes && f !~ size_ok)
        fail(f, size " bytes exceeds the 1 MB limit")
    }
    END {
      if (bad) exit 1
      print "check_tracked_files: OK (only magmaan content tracked)"
    }'
