// Run the site's prediction path in Node over a suite directory on disk.
//
//   node tests/run_suite.mjs <suite dir> <species.csv> <header.csv> <out.csv> [targets] [n plots]
//
// Columns are the suite manifest's input names. The output holds one row per
// plot: its id, every combined column, each seed's value and the recognised
// share of its cover, under the names SuitePredictor gives them.
// ORT_NUM_THREADS sets the number of wasm threads (default 1).

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-web";

import { predictTables, toCsv } from "./suite_table.mjs";

const [suiteDir, speciesPath, headerPath, outPath, targetList = "", limit = ""] = process.argv.slice(2);
ort.env.wasm.numThreads = Number(process.env.ORT_NUM_THREADS ?? 1);

const fetchJson = async (p) => JSON.parse(await readFile(path.join(suiteDir, p), "utf8"));
const fetchBytes = async (p) => (await readFile(path.join(suiteDir, p))).buffer;

const { columns, seconds, n } = await predictTables(ort, fetchJson, fetchBytes,
  await readFile(speciesPath, "utf8"), await readFile(headerPath, "utf8"),
  { targets: targetList ? targetList.split(",") : [], limit: Number(limit) || 0 });
for (const [name, s] of Object.entries(seconds)) console.log(`${name}: ${n} plots in ${s.toFixed(1)} s`);
await writeFile(outPath, toCsv(columns));
