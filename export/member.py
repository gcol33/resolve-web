"""One RESOLVE suite member as a plain torch module, read from its checkpoint.

The module takes what the browser assembles per plot and returns the target
columns in their original units:

    species_ids, genus_ids, family_ids  int64 (n, width), 0 = unknown / padding
    weights                             float32 (n, width), the pool weight of
                                        each kept species, 0 on padding
    mask                                float32 (n, width), 1 on each kept
                                        species, recognised or not, 0 on padding
    has_cover                           float32 (n, 1), 1 when the input
                                        carried an abundance column
    continuous                          float32 (n, k), the raw continuous
                                        block with NaN where a value is missing

Filling and standardising the continuous block, the species encoder (the
weighted pool, or the transformer's tokens, self-attention and attention
pooling), the MLP and the heads' inverse transforms all happen inside the module, so they travel with
the weights into the exported graph. Weights are stored in `storage`
(float16 for the browser) and computed with in float32.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import torch
import torch.nn.functional as F
from torch import nn

SPECIES_ENCODING_RANK_POOL, SPECIES_ENCODING_TRANSFORMER = 3, 4
TRANSFORMER_POOLING_ATTENTION = 0
AGGREGATION_ABUNDANCE = 0
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
    if scalar(a, "species_encoding") not in (SPECIES_ENCODING_RANK_POOL, SPECIES_ENCODING_TRANSFORMER):
        raise ValueError(f"species_encoding {scalar(a, 'species_encoding')} has no browser implementation")
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


class Member(nn.Module):
    """The continuous block, the MLP and the heads every member shares; a
    subclass supplies the species features that sit between the block and
    the MLP."""

    def __init__(self, a: dict[str, torch.Tensor], storage: torch.dtype):
        super().__init__()
        self.schema = read_schema(a)
        if scalar(a, "normalization") != NORM_LAYER_NORM or scalar(a, "use_residual"):
            raise ValueError("only the LayerNorm MLP without residuals has a browser implementation")
        if a["head_hidden_dims"].numel():
            raise ValueError("hidden head layers have no browser implementation")
        self.activation = ACTIVATIONS[int(scalar(a, "activation"))]
        self.a = a
        self.storage = storage

        self.register_buffer("fill", a["continuous_fill"].detach().float())
        self.register_buffer("mean", a["continuous_mean"].detach().float())
        self.register_buffer("scale", a["continuous_scale"].detach().float())
        n_layers = a["hidden_dims"].numel()
        self.n_layers = n_layers
        for i in range(n_layers):
            lin, norm = 4 * i, 4 * i + 1
            self.store(f"w{i}", f"param_encoder.mlp.{lin}.weight")
            self.store(f"b{i}", f"param_encoder.mlp.{lin}.bias")
            self.store(f"g{i}", f"param_encoder.mlp.{norm}.inner.weight")
            self.store(f"h{i}", f"param_encoder.mlp.{norm}.inner.bias")
        for k, head in enumerate(self.schema.heads):
            self.store(f"head_w{k}", f"param_head_{head.name}.output.weight")
            self.store(f"head_b{k}", f"param_head_{head.name}.output.bias")

    def store(self, name: str, key: str) -> None:
        self.register_buffer(name, self.a[key].detach().to(self.storage))

    def w(self, name: str) -> torch.Tensor:
        return getattr(self, name).float()

    def species_features(self, species_ids, genus_ids, family_ids, weights, mask, has_cover):
        raise NotImplementedError

    def forward(self, species_ids, genus_ids, family_ids, weights, mask, has_cover, continuous):
        block = torch.where(torch.isnan(continuous), self.fill.expand_as(continuous), continuous)
        block = (block - self.mean) / self.scale
        block = torch.where(self.scale <= 2e-8, torch.zeros_like(block), block)

        x = torch.cat([block, self.species_features(species_ids, genus_ids, family_ids,
                                                    weights, mask, has_cover)], 1)
        for i in range(self.n_layers):
            x = F.linear(x, self.w(f"w{i}"), self.w(f"b{i}"))
            x = F.layer_norm(x, (x.shape[1],), self.w(f"g{i}"), self.w(f"h{i}"), eps=1e-5)
            x = self.activation(x)

        outputs = []
        for k, head in enumerate(self.schema.heads):
            y = F.linear(x, self.w(f"head_w{k}"), self.w(f"head_b{k}"))
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


class PooledMember(Member):
    """The species, genus and family tables each averaged with the plot's
    normalised pool weights, then the has-cover flag."""

    def __init__(self, a: dict[str, torch.Tensor], storage: torch.dtype = torch.float16):
        super().__init__(a, storage)
        for table in ("species", "genus", "family"):
            self.store(f"{table}_table", f"param_encoder.{table}_embedding.weight")

    def species_features(self, species_ids, genus_ids, family_ids, weights, mask, has_cover):
        masked = weights * mask
        share = (masked / masked.sum(1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)

        def pool(ids, table):
            return (self.w(table)[ids] * share).sum(1)

        return torch.cat([pool(species_ids, "species_table"), pool(genus_ids, "genus_table"),
                          pool(family_ids, "family_table"), has_cover], 1)


def attend(query, tokens, padding, in_w, in_b, out_w, out_b, heads: int):
    """Multi-head attention of `query` (n, q, d) over `tokens` (n, t, d), each
    head scaled dot-product attention, keys where `padding` is true left out."""
    n, d = tokens.shape[0], tokens.shape[2]
    head_dim = d // heads
    wq, wk, wv = in_w.chunk(3)
    bq, bk, bv = in_b.chunk(3)

    def split(x, weight, bias):
        return F.linear(x, weight, bias).reshape(n, -1, heads, head_dim).transpose(1, 2)

    q, k, v = split(query, wq, bq), split(tokens, wk, bk), split(tokens, wv, bv)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
    out = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(n, -1, d)
    return F.linear(out, out_w, out_b)


class TransformerMember(Member):
    """One token per kept species (its species, genus and family embeddings
    plus its projected pool weight), post-norm self-attention layers with a
    GELU feed-forward, then a learned query attending over the tokens and a
    LayerNorm, then the has-cover flag."""

    def __init__(self, a: dict[str, torch.Tensor], storage: torch.dtype = torch.float16):
        super().__init__(a, storage)
        if scalar(a, "transformer_pooling") != TRANSFORMER_POOLING_ATTENTION:
            raise ValueError("only attention pooling has a browser implementation")
        if scalar(a, "schema_aggregation") != AGGREGATION_ABUNDANCE:
            raise ValueError("taxonomy pooled beside the attention has no browser implementation")
        self.heads = int(scalar(a, "n_heads"))
        self.n_attention_layers = int(scalar(a, "n_attention_layers"))
        e = "param_encoder."
        for table in ("species", "genus", "family"):
            self.store(f"{table}_table", f"{e}{table}_embedding.weight")
        self.store("weight_proj", f"{e}weight_proj.weight")
        for i in range(self.n_attention_layers):
            layer = f"{e}transformer_encoder.layers.{i}."
            for name, key in (("in_w", "self_attn.in_proj_weight"), ("in_b", "self_attn.in_proj_bias"),
                              ("out_w", "self_attn.out_proj.weight"), ("out_b", "self_attn.out_proj.bias"),
                              ("l1_w", "linear1.weight"), ("l1_b", "linear1.bias"),
                              ("l2_w", "linear2.weight"), ("l2_b", "linear2.bias"),
                              ("n1_g", "norm1.weight"), ("n1_b", "norm1.bias"),
                              ("n2_g", "norm2.weight"), ("n2_b", "norm2.bias")):
                self.store(f"layer{i}_{name}", layer + key)
        self.store("pool_query", f"{e}pool_query")
        for name, key in (("in_w", "in_proj_weight"), ("in_b", "in_proj_bias"),
                          ("out_w", "out_proj.weight"), ("out_b", "out_proj.bias")):
            self.store(f"pool_{name}", f"{e}pool_attn.{key}")
        self.store("pool_norm_g", f"{e}pool_norm.weight")
        self.store("pool_norm_b", f"{e}pool_norm.bias")

    def species_features(self, species_ids, genus_ids, family_ids, weights, mask, has_cover):
        tokens = (self.w("species_table")[species_ids] + self.w("genus_table")[genus_ids]
                  + self.w("family_table")[family_ids]
                  + weights.unsqueeze(-1) * self.w("weight_proj")[:, 0])
        # A plot with no species keeps its first position as a key, so the
        # attention softmax has something to attend to.
        valid = mask > 0
        empty = valid.logical_not().all(1, keepdim=True)
        first = torch.arange(valid.shape[1], device=valid.device).eq(0).unsqueeze(0)
        padding = valid.logical_or(empty.logical_and(first)).logical_not()
        d = tokens.shape[2]
        for i in range(self.n_attention_layers):
            p = f"layer{i}_"
            attended = attend(tokens, tokens, padding, self.w(p + "in_w"), self.w(p + "in_b"),
                              self.w(p + "out_w"), self.w(p + "out_b"), self.heads)
            tokens = F.layer_norm(tokens + attended, (d,), self.w(p + "n1_g"), self.w(p + "n1_b"), eps=1e-5)
            ff = F.linear(F.gelu(F.linear(tokens, self.w(p + "l1_w"), self.w(p + "l1_b"))),
                          self.w(p + "l2_w"), self.w(p + "l2_b"))
            tokens = F.layer_norm(tokens + ff, (d,), self.w(p + "n2_g"), self.w(p + "n2_b"), eps=1e-5)
        query = self.w("pool_query").reshape(1, 1, d).expand(tokens.shape[0], 1, d)
        pooled = attend(query, tokens, padding, self.w("pool_in_w"), self.w("pool_in_b"),
                        self.w("pool_out_w"), self.w("pool_out_b"), self.heads).squeeze(1)
        pooled = F.layer_norm(pooled, (d,), self.w("pool_norm_g"), self.w("pool_norm_b"), eps=1e-5)
        return torch.cat([pooled, has_cover], 1)


def member_from(a: dict[str, torch.Tensor], storage: torch.dtype = torch.float16) -> Member:
    """The member class the checkpoint's species encoder calls for."""
    if scalar(a, "species_encoding") == SPECIES_ENCODING_TRANSFORMER:
        return TransformerMember(a, storage)
    return PooledMember(a, storage)
