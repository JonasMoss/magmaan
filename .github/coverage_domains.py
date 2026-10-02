"""Summarize LLVM's source coverage by magmaan domain, without a gate."""
import json
import re
import sys
from collections import defaultdict

counts = defaultdict(lambda: [0, 0, 0, 0])
with open(sys.argv[1], encoding="utf-8") as source:
    report = json.load(source)
for data in report["data"]:
    for entry in data["files"]:
        match = re.search(r"cpp/(?:include/magmaan|src)/([^/]+)/", entry["filename"])
        domain = match.group(1) if match else "(top-level)"
        if domain == "util":
            domain = "(top-level)"
        summary = entry["summary"]
        values = counts[domain]
        for offset, key in [(0, "lines"), (2, "regions")]:
            values[offset] += summary[key]["count"]
            values[offset + 1] += summary[key]["covered"]
print("| Domain | Lines covered / total | Regions covered / total |")
print("| --- | ---: | ---: |")
for domain, (lines, covered_lines, regions, covered_regions) in sorted(counts.items()):
    print(f"| {domain} | {covered_lines} / {lines} | {covered_regions} / {regions} |")
