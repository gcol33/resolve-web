"""One RESOLVE suite member as a plain torch module, read from its checkpoint.

The module takes what the browser assembles per plot and returns the target
columns in their original units:

    species_ids, genus_ids, family_ids  int64 (n, width), 0 = unknown / padding
    weights                             float32 (n, width), the pool weight of
                                        each kept species, 0 on padding
    has_cover                           float32 (n, 1), 1 when the input
                                        carried an abundance column
    continuous                          float32 (n, k), the raw continuous
                                        block with NaN where a value is missing

Filling and standardising the continuous block, pooling, the MLP and the
heads' inverse transforms all happen inside the module, so they travel with
the weights into the exported graph. Weights are stored in `storage`
(float16 for the browser) and computed with in float32.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

SPECIES_ENCODING_RANK_POOL = 3
ACTIVATIONS = {0: F.relu, 2: F.gelu, 3: F.silu, 4: torch.tanh}
NORM_LAYER_NORM = 1
TASK_CLASSIFICATION = 1
TRANSFORM_NONE, TRANSFORM_LOG1P = 0, 1
POOL_WEIGHTINGS = {0: "binary", 1: "abundance", 2: "log1p", 3: "norm"}


def strings(a: dict, key: str) -> list[str]:
    lengths = a[key + "_lengths"].tolist()
    data = bytes(a[key + "_bytes"].tolist())
    out, offset = [], 0
    for n in lengths:
        out.append(data[offset:offset + n].decode("utf-8"))
        offset += n
    return out


def text(a: dict, key: str) -> str:
    return bytes(a[key].tolist()).decode("utf-8") if a[key + "_len"].item() else ""


def scalar(a: dict, key: str):
    return a[key].item()


def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    module = torch.jit.load(path, map_location="cpu")
    a = dict(module.named_parameters())
    a.update(dict(module.named_buffers()))
    return a


@dataclass
class Head:
    name: str
    classification: bool
    transform: int
    classes: list[str] = field(default_factory=list)
    scaler: tuple[float, float] | None = None


@dataclass
class Schema:
    """What the browser needs to assemble a member's inputs."""

    species_vocab: list[str]
    genus_vocab: list[str]
    family_vocab: list[str]
    width: int
    pool_weighting: str
    has_coordinates: bool
    covariates: list[str]
    track_unknown_fraction: bool
    heads: list[Head]

    @property
    def continuous_columns(self) -> list[str]:
        cols = ["longitude", "latitude", "coordinates_missing"] if self.has_coordinates else []
        cols += self.covariates + [f"{c}_missing" for c in self.covariates]
        if self.track_unknown_fraction:
            cols.append("unknown_fraction")
        return cols


def read_schema(a: dict) -> Schema:
    if scalar(a, "species_encoding") != SPECIES_ENCODING_RANK_POOL:
        raise ValueError(f"species_encoding {scalar(a, 'species_encoding')} is not rank_pool")
    weighting = scalar(a, "schema_pool_weighting")
    if weighting not in POOL_WEIGHTINGS:
        raise ValueError(f"pool_weighting {weighting} has no browser implementation")
    if scalar(a, "schema_n_categoricals"):
        raise ValueError("categorical inputs have no browser implementation")
    if scalar(a, "schema_track_unknown_count"):
        raise ValueError("the unknown-species count input has no browser implementation")
    scalers = {text(a, f"target_scaler_{i}_name"): (a[f"target_scaler_{i}_mean"].item(),
                                                    a[f"target_scaler_{i}_scale"].item())
               for i in range(int(scalar(a, "n_target_scalers")))}
    heads = []
    for t in range(int(scalar(a, "schema_n_targets"))):
        name = text(a, f"schema_target_{t}_name")
        classification = scalar(a, f"schema_target_{t}_task") == TASK_CLASSIFICATION
        transform = int(scalar(a, f"schema_target_{t}_transform"))
        if transform not in (TRANSFORM_NONE, TRANSFORM_LOG1P):
            raise ValueError(f"target {name}: transform {transform} has no browser implementation")
        classes = ([text(a, f"schema_target_{t}_class_{k}")
                    for k in range(int(scalar(a, f"schema_target_{t}_n_class_names")))]
                   if classification else [])
        heads.append(Head(name, classification, transform, classes, scalers.get(name)))
    schema = Schema(
        species_vocab=strings(a, "schema_species_vocab"),
        genus_vocab=strings(a, "schema_genus_vocab"),
        family_vocab=strings(a, "schema_family_vocab"),
        width=int(scalar(a, "schema_pool_species_cap")),
        pool_weighting=POOL_WEIGHTINGS[weighting],
        has_coordinates=bool(scalar(a, "schema_has_coordinates")),
        covariates=[text(a, f"schema_covariate_{i}") for i in range(int(scalar(a, "schema_n_covariates")))],
        track_unknown_fraction=bool(scalar(a, "schema_track_unknown_fraction")),
        heads=heads,
    )
    n_continuous = a["continuous_mean"].numel()
    if len(schema.continuous_columns) != n_continuous:
        raise ValueError(f"continuous block holds {n_continuous} columns, the schema names "
                         f"{schema.continuous_columns}")
    return schema


class PooledMember(nn.Module):
    def __init__(self, a: dict[str, torch.Tensor], storage: torch.dtype = torch.float16):
        super().__init__()
        self.schema = read_schema(a)
        if scalar(a, "normalization") != NORM_LAYER_NORM or scalar(a, "use_residual"):
            raise ValueError("only the LayerNorm MLP without residuals has a browser implementation")
        if a["head_hidden_dims"].numel():
            raise ValueError("hidden head layers have no browser implementation")
        self.activation = ACTIVATIONS[int(scalar(a, "activation"))]

        def half(key: str) -> torch.Tensor:
            return a[key].detach().to(storage)

        self.register_buffer("fill", a["continuous_fill"].detach().float())
        self.register_buffer("mean", a["continuous_mean"].detach().float())
        self.register_buffer("scale", a["continuous_scale"].detach().float())
        for table in ("species", "genus", "family"):
            self.register_buffer(f"{table}_table", half(f"param_encoder.{table}_embedding.weight"))
        n_layers = a["hidden_dims"].numel()
        self.n_layers = n_layers
        for i in range(n_layers):
            lin, norm = 4 * i, 4 * i + 1
            self.register_buffer(f"w{i}", half(f"param_encoder.mlp.{lin}.weight"))
            self.register_buffer(f"b{i}", half(f"param_encoder.mlp.{lin}.bias"))
            self.register_buffer(f"g{i}", half(f"param_encoder.mlp.{norm}.inner.weight"))
            self.register_buffer(f"h{i}", half(f"param_encoder.mlp.{norm}.inner.bias"))
        for k, head in enumerate(self.schema.heads):
            self.register_buffer(f"head_w{k}", half(f"param_head_{head.name}.output.weight"))
            self.register_buffer(f"head_b{k}", half(f"param_head_{head.name}.output.bias"))

    def forward(self, species_ids, genus_ids, family_ids, weights, has_cover, continuous):
        block = torch.where(torch.isnan(continuous), self.fill.expand_as(continuous), continuous)
        block = (block - self.mean) / self.scale
        block = torch.where(self.scale <= 2e-8, torch.zeros_like(block), block)

        share = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        share = share.unsqueeze(-1)

        def pool(ids, table):
            return (table[ids].float() * share).sum(1)

        x = torch.cat([block, pool(species_ids, self.species_table), pool(genus_ids, self.genus_table),
                       pool(family_ids, self.family_table), has_cover], 1)
        for i in range(self.n_layers):
            x = F.linear(x, getattr(self, f"w{i}").float(), getattr(self, f"b{i}").float())
            x = F.layer_norm(x, (x.shape[1],), getattr(self, f"g{i}").float(),
                             getattr(self, f"h{i}").float(), eps=1e-5)
            x = self.activation(x)

        outputs = []
        for k, head in enumerate(self.schema.heads):
            y = F.linear(x, getattr(self, f"head_w{k}").float(), getattr(self, f"head_b{k}").float())
            if head.classification:
                outputs.append(torch.softmax(y, 1))
                continue
            y = y.squeeze(-1)
            if head.scaler is not None:
                y = y * head.scaler[1] + head.scaler[0]
            if head.transform == TRANSFORM_LOG1P:
                y = torch.expm1(torch.clamp(y, -88.0, 88.0))
            outputs.append(y)
        return tuple(outputs)
