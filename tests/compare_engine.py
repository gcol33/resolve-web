"""Compare the site's predictions with the engine's SuitePredictor on the same plots.

    <python with resolve_core> tests/compare_engine.py <suite dir> <species.csv> <header.csv> \
        <site.csv> [--max-entries N]

The site output comes from tests/run_suite.mjs. Every column both share is
compared: class labels must agree, probabilities and regressions within
TOLERANCE. `--max-entries` restricts the comparison to plots holding at most
that many distinct species, for suites whose engine kept a long plot's first
records rather than its most abundant ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# float32 in onnxruntime-web against the engine: the order the kernels sum in
# moves a single member's output by up to about 4e-4 (relative, or degrees).
TOLERANCE = {"probability": 1e-4, "relative": 5e-4, "degrees": 5e-4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path)
    ap.add_argument("species", type=Path)
    ap.add_argument("header", type=Path)
    ap.add_argument("site", type=Path)
    ap.add_argument("--max-entries", type=int, default=0)
    args = ap.parse_args()

    import resolve_core as rc

    manifest = json.loads((args.suite / "manifest.json").read_text(encoding="utf-8"))
    inp = manifest["inputs"]
    site = pd.read_csv(args.site, dtype={"plot_id": str}, keep_default_na=False, na_values=[""])
    ids = set(site["plot_id"])
    header = pd.read_csv(args.header, low_memory=False, dtype={inp["plot_id"]: str})
    plots = pd.read_csv(args.species, low_memory=False, dtype={inp["plot_id"]: str})
    header = header[header[inp["plot_id"]].isin(ids)].reset_index(drop=True)
    plots = plots[plots[inp["plot_id"]].isin(ids)].reset_index(drop=True)
    targets = sorted({c.split("_")[0] for c in site.columns if c != "plot_id"}
                     & {t["name"] for t in manifest["targets"]})

    predictor = rc.SuitePredictor.load(str(args.suite), targets=targets)
    engine = predictor.predict(plots, header, keep_members=True).to_pandas(probabilities=True)
    engine["plot_id"] = header[inp["plot_id"]].astype(str).to_numpy()
    merged = site.merge(engine, on="plot_id", suffixes=("_site", "_engine"))
    if args.max_entries:
        entries = plots.groupby(inp["plot_id"])[inp["species"]].nunique()
        short = set(entries[entries <= args.max_entries].index)
        merged = merged[merged["plot_id"].isin(short)]
    print(f"{len(merged)} plots compared")

    failed = False
    for col in site.columns:
        if col == "plot_id" or f"{col}_engine" not in merged.columns:
            continue
        a, b = merged[f"{col}_site"], merged[f"{col}_engine"]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            agree = float((a.astype(str) == b.astype(str)).mean())
            ok = agree == 1.0
            print(f"{col}: label agreement {agree:.5f}")
        else:
            a, b = a.to_numpy(float), b.to_numpy(float)
            if "_prob_" in col:
                err, tol = np.nanmax(np.abs(a - b)), TOLERANCE["probability"]
            elif col.endswith(("_latitude", "_longitude")):
                err, tol = np.nanmax(np.abs(a - b)), TOLERANCE["degrees"]
            else:
                err, tol = np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1.0)), TOLERANCE["relative"]
            ok = bool(err <= tol)
            print(f"{col}: max difference {err:.2e} (tolerance {tol:.0e})")
        failed |= not ok
    missing = [c for c in site.columns if c != "plot_id" and f"{c}_engine" not in merged.columns]
    if missing:
        print("site columns the engine does not write:", ", ".join(missing))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
