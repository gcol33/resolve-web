// Predicting a suite's targets for assembled plots with onnxruntime-web.
//
// `ort` is the onnxruntime-web module; `fetchBytes(path)` returns a file of
// the suite directory as an ArrayBuffer and `fetchJson(path)` as parsed JSON,
// so the same code reads from a web server and, in tests, from disk.

import { encodeBatch } from "./assemble.js";
import { combine } from "./combine.js";

export async function loadSuite(fetchJson) {
  const index = await fetchJson("suite.json");
  const targets = new Map();
  for (const entry of index.targets) targets.set(entry.name, entry);
  return { index, targets, fetchJson };
}

export async function loadTarget(suite, name) {
  const entry = suite.targets.get(name);
  if (!entry) throw new Error(`the suite has no target '${name}'`);
  if (!entry.loaded) entry.loaded = suite.fetchJson(entry.path);
  return entry.loaded;
}

function directory(path) {
  const cut = path.lastIndexOf("/");
  return cut < 0 ? "" : path.slice(0, cut + 1);
}

// Positions 0..n-1 of `plotIndices` in batches. Plots are ordered by the
// number of species they keep under `cap`, so a batch is padded to the width
// of plots like its own rather than to the widest plot of the upload (ties
// keep the input order); a batch holds at most `batchSize` plots and at most
// `tokenBudget` positions once padded.
function cut(assembled, plotIndices, cap, batchSize, tokenBudget) {
  const width = plotIndices.map((p) => Math.max(1, Math.min(cap, assembled.plots[p].names.length)));
  const order = [...plotIndices.keys()].sort((a, b) => width[a] - width[b] || a - b);
  const batches = [];
  let rows = [];
  for (const i of order) {
    if (rows.length && (rows.length === batchSize || (rows.length + 1) * width[i] > tokenBudget)) {
      batches.push(rows);
      rows = [];
    }
    rows.push(i);
  }
  if (rows.length) batches.push(rows);
  return batches;
}

// One target for `plotIndices` of `assembled`: the combined columns, the
// per-seed values and the share of each plot's cover the target recognises.
export async function predictTarget(ort, suite, name, assembled, plotIndices, options = {}) {
  const { fetchBytes, batchSize = 512, tokenBudget = 8192, sessionOptions = { executionProviders: ["wasm"] }, onProgress = () => {} } = options;
  const target = await loadTarget(suite, name);
  const n = plotIndices.length;
  const batches = [];
  for (const rows of cut(assembled, plotIndices, target.schema.width, batchSize, tokenBudget)) {
    batches.push({ rows, encoded: encodeBatch(assembled, rows.map((i) => plotIndices[i]), target.schema) });
  }
  const heads = target.schema.heads;
  const members = [];
  const base = directory(suite.targets.get(name).path);
  for (const [m, member] of target.members.entries()) {
    const bytes = await fetchBytes(base + member.file);
    const session = await ort.InferenceSession.create(new Uint8Array(bytes), sessionOptions);
    const outputs = heads.map((head) => new Float32Array(n * (head.classification ? head.classes.length : 1)));
    let done = 0;
    for (const { rows, encoded } of batches) {
      const feeds = {};
      for (const key of session.inputNames) {
        const [data, dims] = encoded.feeds[key];
        feeds[key] = new ort.Tensor(data instanceof BigInt64Array ? "int64" : "float32", data, dims);
      }
      const result = await session.run(feeds);
      heads.forEach((head, h) => {
        const width = head.classification ? head.classes.length : 1;
        const values = result[head.name].data;
        rows.forEach((row, b) => outputs[h].set(values.subarray(b * width, (b + 1) * width), row * width));
      });
      done += rows.length;
      onProgress({ target: name, member: m, members: target.members.length, done, total: n });
    }
    await session.release();
    members.push(outputs);
  }
  const recognised = new Float64Array(n);
  for (const { rows, encoded } of batches) rows.forEach((row, b) => { recognised[row] = encoded.recognised[b]; });
  const combined = combine(target, members, n);
  return { ...combined, recognised, seeds: target.members.map((m) => m.seed), seedValues: combined.seeds };
}
