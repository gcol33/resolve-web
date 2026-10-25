"""Write the genus and family of every species name in the training plots,
and the training name each EVA name was matched to.

    python export/taxonomy.py <processed plots.csv> <raw EVA species table> \
        site/models/taxonomy.json

The page fills a record's genus and family from this table when the user's
table has no such columns, as every training record carried them. A name's
genus and family are the smallest non-empty ones its records carry, the rule
the engine applies. The file holds `genera` and `families` (sorted lists),
`taxa`, a map from each training name to [genus index, family index] (-1 for
none), and `aliases`, a map from an EVA name that is not itself a training
name to the training name most of its records were matched to, so a name
written as EVA writes it reads as the model knows it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLUMNS = ["WFO_TAXON", "WFO_GENUS", "WFO_FAMILY"]


def smallest(frame: pd.DataFrame, column: str) -> pd.Series:
    named = frame[frame[column].notna() & (frame[column] != "")]
    return named.groupby("WFO_TAXON")[column].min()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plots", type=Path)
    ap.add_argument("raw", type=Path, help="the EVA species table with EVA_TAXON and WFO_TAXON")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    parts = []
    for chunk in pd.read_csv(args.plots, usecols=COLUMNS, dtype=str, chunksize=5_000_000,
                             keep_default_na=False):
        parts.append(chunk.drop_duplicates())
    records = pd.concat(parts).drop_duplicates()
    records = records[records["WFO_TAXON"] != ""]
    genus = smallest(records, "WFO_GENUS")
    family = smallest(records, "WFO_FAMILY")
    taxa = sorted(records["WFO_TAXON"].unique())
    genera = sorted(genus.unique())
    families = sorted(family.unique())
    g_index = {g: i for i, g in enumerate(genera)}
    f_index = {f: i for i, f in enumerate(families)}
    table = {t: [g_index.get(genus.get(t), -1), f_index.get(family.get(t), -1)] for t in taxa}

    counts = None
    for chunk in pd.read_csv(args.raw, usecols=["EVA_TAXON", "WFO_TAXON"], dtype=str,
                             chunksize=5_000_000, keep_default_na=False):
        part = chunk[(chunk["EVA_TAXON"] != "") & (chunk["WFO_TAXON"] != "")].value_counts()
        counts = part if counts is None else counts.add(part, fill_value=0)
    pairs = counts.rename("n").reset_index().sort_values(["EVA_TAXON", "n", "WFO_TAXON"],
                                                         ascending=[True, False, True])
    best = pairs.drop_duplicates("EVA_TAXON")
    known = set(taxa)
    aliases = {e: w for e, w in zip(best["EVA_TAXON"], best["WFO_TAXON"])
               if e not in known and w in known}

    args.out.write_text(json.dumps({"genera": genera, "families": families, "taxa": table,
                                    "aliases": aliases},
                                   ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(taxa):,} names, {len(genera):,} genera, {len(families):,} families, "
          f"{len(aliases):,} aliases, {args.out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
