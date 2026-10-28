"""Export a RESOLVE suite to the files the browser predicts from.

    python export/export_suite.py E:/research/.../production/V_species_context \
        site/models/V_species_context

For each target the output directory gets `<target>/target.json` (the inputs a
member reads, its vocabularies, the heads and the rule that combines the
members) and one `<target>/seed_<k>.onnx` per member, its weights stored in
the `--storage` dtype.
`suite.json` lists the targets. Every member of a target must share one
schema; the exporter stops when they do not. Each exported graph is run in
onnxruntime against the same member in torch before it is written, and the
largest change the storage dtype makes against the float32 weights on that
probe batch is printed beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from member import Member, load_checkpoint, member_from  # noqa: E402

INPUT_NAMES = ["species_ids", "genus_ids", "family_ids", "weights", "mask", "has_cover", "continuous"]
FORMAT_VERSION = 1
# float16 halves the float32 files, int8 (blocks of the largest matrices,
# float16 for the rest; see member.store) takes them to under a third.
STORAGE = {"float32": torch.float32, "float16": torch.float16, "int8": torch.int8}
# The largest difference of an exported graph from the same member in torch
# on the probe batch: a probability, or a regression output relative to
# max(|value|, 1). It comes from the order the kernels sum in (up to about
# 2e-4 on the probe batch).
TOLERANCE = 5e-4


def probe_inputs(member: Member, n: int = 64, seed: int = 0) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    s = member.schema
    width = min(s.width, 40)
    lengths = torch.randint(1, width + 1, (n,), generator=g)
    mask = torch.arange(width)[None, :] < lengths[:, None]
    species = torch.randint(0, len(s.species_vocab), (n, width), generator=g) * mask
    genus = torch.randint(0, len(s.genus_vocab), (n, width), generator=g) * mask
    family = torch.randint(0, len(s.family_vocab), (n, width), generator=g) * mask
    weights = torch.log1p(torch.rand(n, width, generator=g) * 50) * mask
    continuous = member.mean + member.scale * torch.randn(n, member.mean.numel(), generator=g)
    continuous[torch.rand(continuous.shape, generator=g) < 0.1] = float("nan")
    return species, genus, family, weights, mask.float(), torch.ones(n, 1), continuous.float()


def forward(member: Member, inputs) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        return member(*inputs)


def difference(member: Member, got, reference) -> dict[str, float]:
    """Per head, the largest difference of `got` from `reference`: a
    probability, or a regression output relative to max(|value|, 1)."""
    report = {}
    for head, ref, out in zip(member.schema.heads, reference, got):
        ref, out = np.asarray(ref), np.asarray(out)
        if head.classification:
            report[head.name] = float(np.abs(out - ref).max())
        else:
            report[head.name] = float((np.abs(out - ref) / np.maximum(np.abs(ref), 1.0)).max())
    return report


def export_member(member: Member, inputs, path: Path) -> None:
    n = torch.export.Dim("n", min=1)
    width = torch.export.Dim("width", min=1)
    shapes = {"species_ids": {0: n, 1: width}, "genus_ids": {0: n, 1: width},
              "family_ids": {0: n, 1: width}, "weights": {0: n, 1: width},
              "mask": {0: n, 1: width},
              "has_cover": {0: n}, "continuous": {0: n}}
    program = torch.onnx.export(
        member, inputs, input_names=INPUT_NAMES,
        output_names=[h.name for h in member.schema.heads],
        dynamic_shapes=shapes, dynamo=True, external_data=False, optimize=True)
    program.save(str(path))


def check_member(path: Path, member: Member, inputs) -> dict[str, float]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {name: t.numpy() for name, t in zip(INPUT_NAMES, inputs)}
    report = difference(member, session.run(None, feeds), forward(member, inputs))
    for name, err in report.items():
        if err > TOLERANCE:
            raise SystemExit(f"{path}: {name} differs from the member in torch by {err:.2e}")
    return report


def schema_json(member: Member) -> dict:
    """The inputs the browser assembles for a member. A head's target scaler is
    applied inside the member's graph, so it is not part of the shared schema."""
    s = member.schema
    out = asdict(s)
    for head in out["heads"]:
        del head["scaler"]
    out["continuous_columns"] = s.continuous_columns
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--targets", default="", help="comma-separated subset of the suite's targets")
    ap.add_argument("--storage", choices=sorted(STORAGE), default="float16",
                    help="the dtype the weights are stored in")
    args = ap.parse_args()

    manifest = json.loads((args.suite / "manifest.json").read_text(encoding="utf-8"))
    wanted = set(filter(None, args.targets.split(",")))
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"format_version": FORMAT_VERSION, "name": manifest["name"],
             "description": manifest["description"], "licence": manifest["licence"],
             "engine_version": manifest["engine_version"], "taxonomy": manifest["taxonomy"],
             "storage": args.storage,
             "inputs": manifest["inputs"], "targets": []}
    for target in manifest["targets"]:
        if wanted and target["name"] not in wanted:
            continue
        tdir = args.out / target["name"]
        tdir.mkdir(exist_ok=True)
        schema, members = None, []
        for m in target["members"]:
            a = load_checkpoint(str(args.suite / m["file"]))
            member = member_from(a, storage=STORAGE[args.storage]).eval()
            this = schema_json(member)
            if schema is None:
                schema = this
            elif this != schema:
                raise SystemExit(f"{target['name']}: seed {m['seed']} has a schema of its own")
            inputs = probe_inputs(member)
            path = tdir / f"seed_{m['seed']}.onnx"
            export_member(member, inputs, path)
            errors = check_member(path, member, inputs)
            drift = difference(member, forward(member, inputs),
                               forward(member_from(a, storage=torch.float32).eval(), inputs))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            members.append({"seed": m["seed"], "file": path.name, "bytes": path.stat().st_size,
                            "sha256": digest, "checkpoint_sha256": m["sha256"]})
            print(f"{target['name']} seed {m['seed']}: {path.stat().st_size / 1e6:.1f} MB, graph "
                  + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()) + f"; {args.storage} storage "
                  + ", ".join(f"{k} {v:.1e}" for k, v in drift.items()), flush=True)
        entry = {"name": target["name"], "task": target["task"], "combine": target["combine"],
                 "outputs": target["outputs"], "units": target["units"],
                 "validation": target.get("validation", {})}
        (tdir / "target.json").write_text(
            json.dumps({**entry, "schema": schema, "members": members}, ensure_ascii=False),
            encoding="utf-8")
        index["targets"].append({**entry, "path": f"{target['name']}/target.json",
                                 "bytes": sum(m["bytes"] for m in members)})
    (args.out / "suite.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
