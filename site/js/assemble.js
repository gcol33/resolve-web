// Plot assemblages from a species table, and the model inputs of one target
// for a batch of plots, assembled as RESOLVE assembles them:
//
//   * a record's cover is a number; a missing or unreadable cover reads as 1
//     and a cover of 0 as `zeroAs`;
//   * the records of one species in one plot are one entry, at the place of
//     its first record, with the covers summed in float32;
//   * a species' genus and family are the smallest non-empty names its
//     records carry anywhere in the table;
//   * a name is looked up byte for byte; a name outside the target's
//     vocabulary encodes as unknown (0) and keeps its cover;
//   * a plot holding more species than the target's width keeps the ones with
//     the largest cover, ties by name, in plot order;
//   * a missing coordinate or covariate enters as NaN, and the model reads
//     its training mean beside a flag.

const encoder = new TextEncoder();

// Byte order of the UTF-8 encodings, the order the engine compares names in.
export function compareNames(a, b) {
  if (a === b) return 0;
  const x = encoder.encode(a);
  const y = encoder.encode(b);
  const n = Math.min(x.length, y.length);
  for (let i = 0; i < n; i += 1) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
}

const DECIMAL = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

// A decimal number, with a point or a single decimal comma; anything else
// reads as missing.
export function readNumber(cell) {
  if (cell === undefined || cell === null) return null;
  let s = String(cell).trim();
  if (!s.includes(".") && s.split(",").length === 2) s = s.replace(",", ".");
  if (!DECIMAL.test(s)) return null;
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
}

function keepSmallest(map, key, name) {
  if (!name) return;
  const current = map.get(key);
  if (current === undefined || compareNames(name, current) < 0) map.set(key, name);
}

// columns: { plot, species, abundance?, genus?, family?, latitude?, longitude?,
//            covariates: { <schema name>: <column> } }
// A plot's coordinates and covariates come from `header` when given, else
// from the first non-empty cell among its records in `species`.
export function buildPlots(species, columns, { header = null, headerColumns = columns, zeroAs = 1 } = {}) {
  const col = (table, name) => (name ? table.columns.get(name) : undefined);
  const plotCol = col(species, columns.plot);
  const nameCol = col(species, columns.species);
  if (!plotCol || !nameCol) throw new Error("the species table needs a plot and a species column");
  const coverCol = col(species, columns.abundance);
  const genusCol = col(species, columns.genus);
  const familyCol = col(species, columns.family);

  const index = new Map();
  const plots = [];
  const genusOf = new Map();
  const familyOf = new Map();
  for (let r = 0; r < species.nRows; r += 1) {
    const id = plotCol[r];
    let p = index.get(id);
    if (p === undefined) {
      p = plots.length;
      index.set(id, p);
      plots.push({ id, names: [], covers: [], position: new Map() });
    }
    const plot = plots[p];
    const name = nameCol[r] ?? "";
    let cover = coverCol ? readNumber(coverCol[r]) : null;
    cover = cover === null ? 1 : cover === 0 ? zeroAs : cover;
    keepSmallest(genusOf, name, genusCol ? genusCol[r] : "");
    keepSmallest(familyOf, name, familyCol ? familyCol[r] : "");
    const at = plot.position.get(name);
    if (at === undefined) {
      plot.position.set(name, plot.names.length);
      plot.names.push(name);
      plot.covers.push(Math.fround(cover));
    } else {
      plot.covers[at] = Math.fround(plot.covers[at] + Math.fround(cover));
    }
  }

  const site = header ?? species;
  const siteColumns = header ? headerColumns : columns;
  const sitePlot = col(site, siteColumns.plot);
  if (!sitePlot) throw new Error("the header table needs a plot column");
  const fields = {
    latitude: siteColumns.latitude,
    longitude: siteColumns.longitude,
    ...Object.fromEntries(Object.entries(siteColumns.covariates ?? {}).map(([k, v]) => [`cov:${k}`, v])),
  };
  for (const plot of plots) plot.site = {};
  for (const [key, name] of Object.entries(fields)) {
    const cells = col(site, name);
    if (!cells) continue;
    for (let r = 0; r < site.nRows; r += 1) {
      const p = index.get(sitePlot[r]);
      if (p === undefined || key in plots[p].site) continue;
      const v = readNumber(cells[r]);
      if (v !== null || header) plots[p].site[key] = v;
    }
  }
  for (const plot of plots) delete plot.position;
  return { plots, genusOf, familyOf, hasCover: Boolean(coverCol) };
}

function lookup(vocab) {
  const map = new Map();
  for (let i = 1; i < vocab.length; i += 1) map.set(vocab[i], i);
  return map;
}

const lookups = new WeakMap();

function vocabularies(schema) {
  let v = lookups.get(schema);
  if (!v) {
    v = {
      species: lookup(schema.species_vocab),
      genus: lookup(schema.genus_vocab),
      family: lookup(schema.family_vocab),
    };
    lookups.set(schema, v);
  }
  return v;
}

const WEIGHTINGS = {
  binary: (v) => v.map(() => 1),
  abundance: (v) => v,
  log1p: (v) => v.map((x) => Math.log1p(x)),
  norm: (v) => {
    const total = v.reduce((a, b) => a + b, 0);
    return v.map((x) => (total > 0 ? x / total : 0));
  },
};

// The entries a plot keeps under `width`, as indices in plot order.
function kept(plot, width) {
  const n = plot.names.length;
  const all = [...Array(n).keys()];
  if (n <= width) return all;
  const ranked = all.sort((a, b) => plot.covers[b] - plot.covers[a] || compareNames(plot.names[a], plot.names[b]));
  return ranked.slice(0, width).sort((a, b) => a - b);
}

// The inputs of one target's members for `plotIndices` of `assembled`.
export function encodeBatch(assembled, plotIndices, schema) {
  const vocab = vocabularies(schema);
  const { plots, genusOf, familyOf, hasCover } = assembled;
  const n = plotIndices.length;
  let widest = 1;
  for (const p of plotIndices) widest = Math.max(widest, plots[p].names.length);
  const width = Math.max(1, Math.min(schema.width, widest));
  const columns = schema.continuous_columns;
  const k = columns.length;

  const speciesIds = new BigInt64Array(n * width);
  const genusIds = new BigInt64Array(n * width);
  const familyIds = new BigInt64Array(n * width);
  const weights = new Float32Array(n * width);
  const continuous = new Float32Array(n * k).fill(NaN);
  const recognised = new Float64Array(n);

  const weigh = WEIGHTINGS[schema.pool_weighting];
  for (let b = 0; b < n; b += 1) {
    const plot = plots[plotIndices[b]];
    const values = plot.covers.map((c) => (hasCover ? c : 1));
    const plotWeights = weigh(values);
    let total = 0;
    let unknown = 0;
    plot.names.forEach((name, i) => {
      total += values[i];
      if (!vocab.species.has(name)) unknown += values[i];
    });
    recognised[b] = total > 0 ? 1 - unknown / total : 0;
    const keep = kept(plot, width);
    keep.forEach((i, j) => {
      const name = plot.names[i];
      const at = b * width + j;
      speciesIds[at] = BigInt(vocab.species.get(name) ?? 0);
      genusIds[at] = BigInt(vocab.genus.get(genusOf.get(name) ?? "") ?? 0);
      familyIds[at] = BigInt(vocab.family.get(familyOf.get(name) ?? "") ?? 0);
      weights[at] = plotWeights[i];
    });

    const site = plot.site;
    const row = b * k;
    let c = 0;
    if (schema.has_coordinates) {
      const lon = site.longitude ?? null;
      const lat = site.latitude ?? null;
      continuous[row] = lon ?? NaN;
      continuous[row + 1] = lat ?? NaN;
      continuous[row + 2] = lon === null || lat === null ? 1 : 0;
      c = 3;
    }
    const nCov = schema.covariates.length;
    schema.covariates.forEach((name, j) => {
      const v = site[`cov:${name}`] ?? null;
      continuous[row + c + j] = v ?? NaN;
      continuous[row + c + nCov + j] = v === null ? 1 : 0;
    });
    c += 2 * nCov;
    if (schema.track_unknown_fraction) continuous[row + c] = total > 0 ? unknown / total : 0;
  }
  return {
    width,
    recognised,
    feeds: {
      species_ids: [speciesIds, [n, width]],
      genus_ids: [genusIds, [n, width]],
      family_ids: [familyIds, [n, width]],
      weights: [weights, [n, width]],
      has_cover: [new Float32Array(n).fill(hasCover ? 1 : 0), [n, 1]],
      continuous: [continuous, [n, k]],
    },
  };
}
