// The site's prediction path over whole tables, shared by the Node runner
// (run_suite.mjs) and the browser bench (browser_bench.html).
//
// The output holds one column per combined value, each seed's value and the
// recognised share of each plot's cover, under the names SuitePredictor gives
// them, after a plot_id column.

import { buildPlots } from "../site/js/assemble.js";
import { readTable } from "../site/js/csv.js";
import { loadSuite, predictTarget } from "../site/js/predict.js";

export async function predictTables(ort, fetchJson, fetchBytes, speciesText, headerText, options = {}) {
  const { targets = [], limit = 0, predictOptions = {} } = options;
  const suite = await loadSuite(fetchJson);
  const inputs = suite.index.inputs;
  const columns = {
    plot: inputs.plot_id, species: inputs.species, abundance: inputs.abundance,
    genus: inputs.genus, family: inputs.family, latitude: inputs.latitude, longitude: inputs.longitude,
    covariates: Object.fromEntries(inputs.covariates.map((c) => [c, c])),
  };
  const assembled = buildPlots(readTable(speciesText), columns,
    { header: readTable(headerText), zeroAs: inputs.zero_abundance_as });
  let indices = [...assembled.plots.keys()];
  if (limit) indices = indices.slice(0, limit);

  const names = targets.length ? targets : [...suite.targets.keys()];
  const out = [["plot_id", indices.map((p) => assembled.plots[p].id)]];
  const seconds = {};
  for (const name of names) {
    const t0 = performance.now();
    const result = await predictTarget(ort, suite, name, assembled, indices, { fetchBytes, ...predictOptions });
    seconds[name] = (performance.now() - t0) / 1000;
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
  }
  return { columns: out, seconds, n: indices.length };
}

export function toCsv(columns) {
  const quote = (v) => {
    const s = typeof v === "number" ? (Number.isNaN(v) ? "" : String(v)) : String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  const n = columns[0][1].length;
  const lines = [columns.map(([k]) => quote(k)).join(",")];
  for (let i = 0; i < n; i += 1) lines.push(columns.map(([, v]) => quote(v[i])).join(","));
  return lines.join("\n") + "\n";
}
