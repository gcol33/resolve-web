// Run the site's prediction path in Node over a suite directory on disk.
//
//   node tests/run_suite.mjs <suite dir> <species.csv> <header.csv> <out.csv> [targets] [n plots]
//
// Columns are the suite manifest's input names. The output holds one row per
// plot: its id, every combined column, each seed's value and the recognised
// share of its cover, under the names SuitePredictor gives them.

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-web";

import { buildPlots } from "../site/js/assemble.js";
import { readTable } from "../site/js/csv.js";
import { loadSuite, predictTarget } from "../site/js/predict.js";

const [suiteDir, speciesPath, headerPath, outPath, targetList = "", limit = ""] = process.argv.slice(2);
ort.env.wasm.numThreads = 1;

const fetchJson = async (p) => JSON.parse(await readFile(path.join(suiteDir, p), "utf8"));
const fetchBytes = async (p) => (await readFile(path.join(suiteDir, p))).buffer;

const suite = await loadSuite(fetchJson);
const inputs = suite.index.inputs;
const columns = {
  plot: inputs.plot_id, species: inputs.species, abundance: inputs.abundance,
  genus: inputs.genus, family: inputs.family, latitude: inputs.latitude, longitude: inputs.longitude,
  covariates: Object.fromEntries(inputs.covariates.map((c) => [c, c])),
};
let header = readTable(await readFile(headerPath, "utf8"));
const species = readTable(await readFile(speciesPath, "utf8"));
const assembled = buildPlots(species, columns, { header, zeroAs: inputs.zero_abundance_as });
let indices = [...assembled.plots.keys()];
if (limit) indices = indices.slice(0, Number(limit));

const names = targetList ? targetList.split(",") : [...suite.targets.keys()];
const out = [["plot_id", indices.map((p) => assembled.plots[p].id)]];
for (const name of names) {
  const t0 = Date.now();
  const result = await predictTarget(ort, suite, name, assembled, indices, { fetchBytes });
  out.push(...result.columns);
  result.seeds.forEach((seed, m) => {
    const v = result.seedValues[m];
    if (Array.isArray(v) && v.length === 2 && ArrayBuffer.isView(v[0])) {
      out.push([`${name}_seed${seed}_latitude`, v[0]], [`${name}_seed${seed}_longitude`, v[1]]);
    } else {
      out.push([`${name}_seed${seed}`, v]);
    }
  });
  out.push([`${name}_recognised`, result.recognised]);
  console.log(`${name}: ${indices.length} plots in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
}

const quote = (v) => {
  const s = typeof v === "number" ? (Number.isNaN(v) ? "" : String(v)) : String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
};
const lines = [out.map(([k]) => quote(k)).join(",")];
for (let i = 0; i < indices.length; i += 1) lines.push(out.map(([, v]) => quote(v[i])).join(","));
await writeFile(outPath, lines.join("\n") + "\n");
