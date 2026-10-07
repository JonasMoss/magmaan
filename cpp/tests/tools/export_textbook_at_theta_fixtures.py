#!/usr/bin/env python3
"""Export two retained lavaan snapshots, without fitting or reading observations.

Usage: python3 cpp/tests/tools/export_textbook_at_theta_fixtures.py CORPUS_ROOT [OUTPUT_DIR]
"""
import hashlib
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
corpus_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else repo_root / "cpp/tests/fixtures/textbook_corpus"
pin = (repo_root / "cpp/tests/fixtures/lavaan_version.txt").read_text().strip()
for book, case_id in [("newsom_2015", "newsom_2015_ex9_3"),
                      ("little_2013", "little_2013_ch3_fig_3_6_1indicator")]:
    case_dir = corpus_root / "cases" / book / case_id
    sources = {name: (case_dir / name).read_bytes()
               for name in ["model.lav", "meta.json", "expected/lavaan_ml.json"]}
    meta = json.loads(sources["meta.json"])
    ref = json.loads(sources["expected/lavaan_ml.json"])
    if ref["lavaan_version"].replace("-", ".") != pin.replace("-", "."):
        raise ValueError(f"{case_id}: oracle version differs from pin {pin}")
    if meta["case_id"] != case_id or ref["case_id"] != case_id or ref["estimator"] != "ML":
        raise ValueError(f"{case_id}: inconsistent snapshot identity")
    theta = ref["theta"] if isinstance(ref["theta"], list) else [ref["theta"]]
    payload = {
        "_meta": {"format_version": 1, "fixture_kind": "textbook_corpus.at_theta",
                  "tool": "cpp/tests/tools/export_textbook_at_theta_fixtures.py",
                  "source_case": f"cases/{book}/{case_id}",
                  "source_sha256": {name: hashlib.sha256(raw).hexdigest()
                                    for name, raw in sources.items()},
                  "lavaan_version": ref["lavaan_version"], "lavaan_pin": pin,
                  "provenance": meta["provenance"]},
        "case_id": case_id, "model": sources["model.lav"].decode(),
        "model_options": meta["model_options"], "lavaan_function": meta["lavaan_function"],
        "lavaan": {"theta": theta, "implied": ref["implied"],
                   "parameter_table": ref["parameter_table"]}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{case_id}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported {case_id} (lavaan {ref['lavaan_version']})")
