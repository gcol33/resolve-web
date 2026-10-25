"""Export a RESOLVE suite to the files the browser predicts from.

    python export/export_suite.py E:/research/.../production/V_species_context \
        site/models/V_species_context

For each target the output directory gets `<target>/target.json` (the inputs a
member reads, its vocabularies, the heads and the rule that combines the
members) and one `<target>/seed_<k>.onnx` per member, its weights stored in
the `--storage` dtype.
`suite.json` lists the targets. Every member of a target must share one
schema; the exporter stops when they do not. Each exported graph is run in
onnxruntime against the float32 checkpoint before it is written.
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
from member import PooledMember, load_checkpoint  # noqa: E402

INPUT_NAMES = ["species_ids", "genus_ids", "family_ids", "weights", "has_cover", "continuous"]
FORMAT_VERSION = 1
STORAGE = {"float32": torch.float32, "float16": torch.float16}
# The largest difference from the float32 checkpoint on the probe batch: a
# probability, or a regression output relative to max(|value|, 1). float32
# differs by the order the kernels sum in (up to about 2e-4 on the probe
# batch); float16 storage moves outputs by about 1e-3 of their scale.
TOLERANCE = {"float32": 5e-4, "float16": 5e-2}


def probe_inputs(member: PooledMember, n: int = 64, seed: int = 0) -> tuple[torch.Tensor, ...]:
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
    return species, genus, family, weights, torch.ones(n, 1), continuous.float()


def reference_forward(a: dict, inputs) -> tuple[torch.Tensor, ...]:
    """The same forward with the float32 checkpoint weights."""
    with torch.no_grad():
        return PooledMember(a, storage=torch.float32).eval()(*inputs)


def export_member(member: PooledMember, inputs, path: Path) -> None:
    n = torch.export.Dim("n", min=1)
    width = torch.export.Dim("width", min=1)
    shapes = {"species_ids": {0: n, 1: width}, "genus_ids": {0: n, 1: width},
              "family_ids": {0: n, 1: width}, "weights": {0: n, 1: width},
              "has_cover": {0: n}, "continuous": {0: n}}
    program = torch.onnx.export(
        member, inputs, input_names=INPUT_NAMES,
        output_names=[h.name for h in member.schema.heads],
        dynamic_shapes=shapes, dynamo=True, external_data=False, optimize=True)
    program.save(str(path))


def check_member(path: Path, member: PooledMember, reference, inputs,
                 tolerance: float) -> dict[str, float]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {name: t.numpy() for name, t in zip(INPUT_NAMES, inputs)}
    got = session.run(None, feeds)
    report = {}
    for head, ref, out in zip(member.schema.heads, reference, got):
        ref = ref.numpy()
        if head.classification:
            err = float(np.abs(out - ref).max())
        else:
            err = float((np.abs(out - ref) / np.maximum(np.abs(ref), 1.0)).max())
        report[head.name] = err
        if err > tolerance:
            raise SystemExit(f"{path}: {head.name} differs from the float32 checkpoint by {err:.2e}")
    return report


def schema_json(member: PooledMember) -> dict:
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
    ap.add_argument("--storage", choices=sorted(STORAGE), default="float32",
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
            member = PooledMember(a, storage=STORAGE[args.storage]).eval()
            this = schema_json(member)
            if schema is None:
                schema = this
            elif this != schema:
                raise SystemExit(f"{target['name']}: seed {m['seed']} has a schema of its own")
            inputs = probe_inputs(member)
            reference = reference_forward(a, inputs)
            path = tdir / f"seed_{m['seed']}.onnx"
            export_member(member, inputs, path)
            errors = check_member(path, member, reference, inputs, TOLERANCE[args.storage])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            members.append({"seed": m["seed"], "file": path.name, "bytes": path.stat().st_size,
                            "sha256": digest, "checkpoint_sha256": m["sha256"]})
            print(f"{target['name']} seed {m['seed']}: {path.stat().st_size / 1e6:.1f} MB, "
                  + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()), flush=True)
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
