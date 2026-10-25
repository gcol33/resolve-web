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

// One target for `plotIndices` of `assembled`: the combined columns, the
// per-seed values and the share of each plot's cover the target recognises.
export async function predictTarget(ort, suite, name, assembled, plotIndices, options = {}) {
  const { fetchBytes, batchSize = 512, onProgress = () => {} } = options;
  const target = await loadTarget(suite, name);
  const n = plotIndices.length;
  const batches = [];
  for (let start = 0; start < n; start += batchSize) {
    batches.push({ start, encoded: encodeBatch(assembled, plotIndices.slice(start, start + batchSize), target.schema) });
  }
  const heads = target.schema.heads;
  const members = [];
  const base = directory(suite.targets.get(name).path);
  for (const [m, member] of target.members.entries()) {
    const bytes = await fetchBytes(base + member.file);
    const session = await ort.InferenceSession.create(new Uint8Array(bytes), { executionProviders: ["wasm"] });
    const outputs = heads.map((head) => new Float32Array(n * (head.classification ? head.classes.length : 1)));
    for (const { start, encoded } of batches) {
      const feeds = {};
      for (const key of session.inputNames) {
        const [data, dims] = encoded.feeds[key];
        feeds[key] = new ort.Tensor(data instanceof BigInt64Array ? "int64" : "float32", data, dims);
      }
      const result = await session.run(feeds);
      heads.forEach((head, h) => {
        const width = head.classification ? head.classes.length : 1;
        outputs[h].set(result[head.name].data, start * width);
      });
      onProgress({ target: name, member: m, members: target.members.length, done: start + encoded.recognised.length, total: n });
    }
    await session.release();
    members.push(outputs);
  }
  const recognised = new Float64Array(n);
  for (const { start, encoded } of batches) recognised.set(encoded.recognised, start);
  const combined = combine(target, members, n);
  return { ...combined, recognised, seeds: target.members.map((m) => m.seed), seedValues: combined.seeds };
}
