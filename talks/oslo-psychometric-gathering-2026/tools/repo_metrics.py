#!/usr/bin/env python3
"""Generate auditable repository metrics for the Oslo magmaan talk.

Only canonical, tracked source paths are counted. Build trees, vendored R-package
copies, generated presentation output, and ignored experiment results are excluded
by construction. The output is a two-column-friendly CSV consumed by talk.qmd.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
TEST_CASE_RE = re.compile(r"\bTEST_CASE(?:_FIXTURE|_TEMPLATE)?\s*\(")
ASSERTION_RE = re.compile(r"\b(?:CHECK|REQUIRE)(?:_[A-Z_]+)?\s*\(")
EXPERIMENT_RE = re.compile(r"^experiments/[0-9]{2}-[^/]+/")


@dataclass(frozen=True)
class Metric:
    key: str
    value: str | int | float
    label: str
    definition: str


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_repo(candidate: Path) -> Path:
    root = run_git(candidate.resolve(), "rev-parse", "--show-toplevel")
    return Path(root).resolve()


def tracked_files(repo: Path) -> list[Path]:
    output = run_git(repo, "ls-files", "-z")
    return [Path(item) for item in output.split("\0") if item]


def selected(
    files: Iterable[Path], prefixes: tuple[str, ...], suffixes: set[str]
) -> list[Path]:
    return [
        path
        for path in files
        if path.suffix in suffixes
        and any(path.as_posix().startswith(prefix) for prefix in prefixes)
    ]


def read_text(repo: Path, path: Path) -> str:
    return (repo / path).read_text(encoding="utf-8", errors="replace")


def line_count(repo: Path, files: Iterable[Path]) -> int:
    total = 0
    for path in files:
        with (repo / path).open("r", encoding="utf-8", errors="replace") as handle:
            total += sum(1 for _ in handle)
    return total


def regex_count(repo: Path, files: Iterable[Path], pattern: re.Pattern[str]) -> int:
    return sum(len(pattern.findall(read_text(repo, path))) for path in files)


def corpus_model_count(repo: Path) -> int:
    corpus = json.loads((repo / "tests/fixtures/corpus.json").read_text())
    return len(corpus["models"])


def unique_experiment_count(files: Iterable[Path]) -> int:
    roots = {
        "/".join(path.parts[:2])
        for path in files
        if EXPERIMENT_RE.match(path.as_posix())
    }
    return len(roots)


def parity_case_count(files: Iterable[Path]) -> int:
    return sum(
        path.parts[:3] == ("tests", "fixtures", "parity")
        and len(path.parts) == 5
        and path.name == "reference.json"
        for path in files
    )


def collect_metrics(repo: Path) -> list[Metric]:
    files = tracked_files(repo)
    core_cpp = selected(files, ("include/", "src/"), CPP_SUFFIXES)
    test_cpp = selected(files, ("tests/",), CPP_SUFFIXES)

    first_commit = date.fromisoformat(
        run_git(repo, "log", "--reverse", "--format=%cs", "HEAD").splitlines()[0]
    )
    last_commit = date.fromisoformat(run_git(repo, "log", "-1", "--format=%cs"))
    snapshot = date.today()
    contributors = {
        email
        for email in run_git(repo, "log", "--format=%ae").splitlines()
        if email
    }

    json_fixtures = [
        path
        for path in files
        if path.as_posix().startswith("tests/fixtures/") and path.suffix == ".json"
    ]
    research_notes = [
        path
        for path in files
        if path.as_posix().startswith("docs/research/notes/")
        and path.suffix == ".tex"
    ]

    days = (snapshot - first_commit).days + 1
    commits = int(run_git(repo, "rev-list", "--count", "HEAD"))

    return [
        Metric("snapshot_date", snapshot.isoformat(), "Snapshot date", "Local calendar date when the metrics tool ran."),
        Metric("first_commit_date", first_commit.isoformat(), "First commit", "Date of the earliest commit reachable from HEAD."),
        Metric("last_commit_date", last_commit.isoformat(), "Latest commit", "Date of HEAD."),
        Metric("elapsed_days", days, "Elapsed days", "Inclusive calendar days from the first commit through the snapshot date."),
        Metric("commits", commits, "Commits", "Commits reachable from HEAD."),
        Metric("commits_per_day", round(commits / days, 1), "Commits per day", "Reachable commits divided by elapsed calendar days."),
        Metric("contributors", len(contributors), "Recorded authors", "Unique author email addresses in reachable commits."),
        Metric("tracked_files", len(files), "Tracked files", "Paths reported by git ls-files."),
        Metric("core_cpp_files", len(core_cpp), "Canonical C++ files", "Tracked C/C++ headers and sources under include/ and src/."),
        Metric("core_cpp_loc", line_count(repo, core_cpp), "Canonical C++ lines", "Physical lines in tracked C/C++ files under include/ and src/."),
        Metric("test_cpp_files", len(test_cpp), "C++ test files", "Tracked C/C++ headers and sources under tests/."),
        Metric("test_cpp_loc", line_count(repo, test_cpp), "C++ test lines", "Physical lines in tracked C/C++ files under tests/."),
        Metric("test_cases", regex_count(repo, test_cpp, TEST_CASE_RE), "C++ test cases", "doctest TEST_CASE-family macro invocations under tests/."),
        Metric("assertions", regex_count(repo, test_cpp, ASSERTION_RE), "C++ assertions", "CHECK*/REQUIRE* macro invocations under tests/."),
        Metric("json_fixtures", len(json_fixtures), "JSON fixtures", "Tracked JSON files under tests/fixtures/."),
        Metric("corpus_models", corpus_model_count(repo), "Synthetic corpus models", "Models in tests/fixtures/corpus.json."),
        Metric("parity_cases", parity_case_count(files), "Real-data parity cases", "Reference JSON cases directly below tests/fixtures/parity/."),
        Metric("experiments", unique_experiment_count(files), "Numbered experiments", "Tracked top-level experiments matching experiments/NN-*/."),
        Metric("research_notes", len(research_notes), "Research notes", "Tracked LaTeX notes directly or recursively under docs/research/notes/."),
    ]


def write_csv(metrics: list[Metric], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value", "label", "definition"))
        for metric in metrics:
            writer.writerow((metric.key, metric.value, metric.label, metric.definition))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Any path inside the git repository (default: inferred from script path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV destination. Without this option, write CSV to stdout.",
    )
    args = parser.parse_args()

    repo = resolve_repo(args.repo)
    metrics = collect_metrics(repo)
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = Path.cwd() / output
        write_csv(metrics, output)
    else:
        writer = csv.writer(__import__("sys").stdout)
        writer.writerow(("metric", "value", "label", "definition"))
        for metric in metrics:
            writer.writerow((metric.key, metric.value, metric.label, metric.definition))


if __name__ == "__main__":
    main()
