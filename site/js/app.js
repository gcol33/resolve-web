import { buildPlots } from "./assemble.js";
import { readTable } from "./csv.js";
import { loadSuite, loadTarget, predictTarget } from "./predict.js";

// The model repository at a fixed revision, so a page always runs the models
// it was checked against; `?models=<url>` reads them from elsewhere.
const MODEL_REPO = "https://huggingface.co/gcol33/resolve-models/resolve";
const MODEL_REVISION = "533b13022146d7ac62ab09456a148b8318f89f24";
const MODELS = new URLSearchParams(location.search).get("models") ?? `${MODEL_REPO}/${MODEL_REVISION}`;
// Plots with coordinates read the context suite; the rest the species-only one.
const SUITES = { context: "V_species_context", species: "V_minimal" };
const TARGET_LABELS = {
  eunis: "EUNIS habitat",
  altitude: "Altitude (m)",
  slope: "Slope (°)",
  area: "Plot area (m²)",
  year: "Survey year",
  location: "Location",
};
const FIELDS = [
  { key: "plot", label: "Plot", table: "both", required: true,
    guesses: ["plotobservationid", "plot", "plot_id", "plotid", "releve", "relevé", "releve_id", "site"] },
  { key: "species", label: "Species", table: "species", required: true,
    guesses: ["wfo_taxon", "species", "taxon", "taxon_name", "scientific_name", "scientificname", "name"] },
  { key: "abundance", label: "Cover (%)", table: "species",
    guesses: ["cover %", "cover", "cover_percent", "abundance", "percentage_cover"] },
  { key: "genus", label: "Genus", table: "species", guesses: ["wfo_genus", "genus"] },
  { key: "family", label: "Family", table: "species", guesses: ["wfo_family", "family"] },
  { key: "latitude", label: "Latitude", table: "header", guesses: ["latitude", "lat", "decimallatitude", "y"] },
  { key: "longitude", label: "Longitude", table: "header", guesses: ["longitude", "lon", "long", "lng", "decimallongitude", "x"] },
  { key: "Year", label: "Year", table: "header", covariate: true, guesses: ["year", "survey_year"] },
  { key: "Month", label: "Month", table: "header", covariate: true, guesses: ["month", "survey_month"] },
];

const $ = (id) => document.getElementById(id);
const state = { species: null, header: null, suites: {}, result: null };

const fetchJsonFrom = (base) => async (path) => {
  const response = await fetch(`${base}/${path}`);
  if (!response.ok) throw new Error(`${base}/${path}: ${response.status}`);
  return response.json();
};
const fetchBytesFrom = (base) => async (path) => {
  const response = await fetch(`${base}/${path}`);
  if (!response.ok) throw new Error(`${base}/${path}: ${response.status}`);
  return response.arrayBuffer();
};

async function suite(name) {
  if (!state.suites[name]) state.suites[name] = loadSuite(fetchJsonFrom(`${MODELS}/${name}`));
  return state.suites[name];
}

function guess(table, field) {
  if (!table) return "";
  const lower = new Map(table.names.map((n) => [n.toLowerCase(), n]));
  for (const g of field.guesses) if (lower.has(g)) return lower.get(g);
  return "";
}

function select(id, table, value, optional) {
  const s = document.createElement("select");
  s.id = id;
  if (optional) s.append(new Option("(none)", ""));
  for (const name of table.names) s.append(new Option(name, name));
  s.value = value;
  s.addEventListener("change", summarise);
  return s;
}

function renderMapping() {
  const box = $("mapping-fields");
  box.replaceChildren();
  for (const field of FIELDS) {
    const tables = field.table === "both" ? ["species", "header"] : [tableOf(field, "species")];
    for (const which of tables) {
      const table = state[which];
      if (!table) continue;
      const id = `map-${which}-${field.key}`;
      const label = document.createElement("label");
      label.htmlFor = id;
      label.textContent = which === "header" && field.table === "both" ? `${field.label} (plot table)` : field.label;
      box.append(label, select(id, table, guess(table, field), !field.required));
    }
  }
  $("mapping").hidden = false;
}

function mapped(which, key) {
  const el = $(`map-${which}-${key}`);
  return el ? el.value : "";
}

// The table a field is read from: plot-level fields from the plot table when
// one is given.
function tableOf(field, which) {
  if (field.table === "both") return which;
  if (field.table === "header") return state.header ? "header" : "species";
  return "species";
}

// The column mapping of one table, as buildPlots takes it.
function columnsFor(which) {
  const out = { covariates: {} };
  for (const field of FIELDS) {
    if (tableOf(field, which) !== which) continue;
    const value = mapped(which, field.key);
    if (!value) continue;
    if (field.covariate) out.covariates[field.key] = value;
    else out[field.key] = value;
  }
  return out;
}

function assemble() {
  const columns = columnsFor("species");
  const headerColumns = state.header ? columnsFor("header") : columns;
  return buildPlots(state.species, columns, { header: state.header, headerColumns, zeroAs: 1 });
}

function summarise() {
  try {
    const assembled = assemble();
    const n = assembled.plots.length;
    const located = assembled.plots.filter((p) => p.site.latitude != null && p.site.longitude != null).length;
    const records = assembled.plots.reduce((a, p) => a + p.names.length, 0);
    $("input-summary").textContent = `${n.toLocaleString()} plots, ${records.toLocaleString()} species entries; `
      + `${located.toLocaleString()} with coordinates.`;
    $("run").disabled = n === 0;
    state.assembled = assembled;
  } catch (e) {
    $("input-summary").textContent = e.message;
    $("run").disabled = true;
  }
}

async function readFile(input) {
  const file = input.files[0];
  return file ? readTable(await file.text()) : null;
}

async function onFiles() {
  state.species = await readFile($("species-file"));
  state.header = await readFile($("header-file"));
  if (!state.species) return;
  renderMapping();
  summarise();
}

function renderTargets() {
  const list = $("target-list");
  for (const [name, label] of Object.entries(TARGET_LABELS)) {
    const wrap = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = name;
    box.checked = true;
    wrap.append(box, ` ${label}`);
    list.append(wrap);
  }
}

function chosenTargets() {
  return [...$("target-list").querySelectorAll("input:checked")].map((b) => b.value);
}

// For one target, the plots each suite answers: a target that reads
// coordinates goes to the species-only suite for plots without them.
async function routes(name, assembled) {
  const context = await suite(SUITES.context);
  const target = context.targets.has(name) ? await loadTarget(context, name) : null;
  const all = [...assembled.plots.keys()];
  if (!target) return [[SUITES.species, all]];
  if (!target.schema.has_coordinates) return [[SUITES.context, all]];
  const located = (p) => assembled.plots[p].site.latitude != null && assembled.plots[p].site.longitude != null;
  return [[SUITES.context, all.filter(located)], [SUITES.species, all.filter((p) => !located(p))]];
}

async function run() {
  const assembled = state.assembled;
  const n = assembled.plots.length;
  const out = new Map([["plot_id", assembled.plots.map((p) => p.id)]]);
  const progress = $("progress");
  progress.hidden = false;
  $("run").disabled = true;
  const targets = chosenTargets();
  try {
    for (const [t, name] of targets.entries()) {
      out.set(`${name}_model`, new Array(n).fill(""));
      for (const [suiteName, plots] of await routes(name, assembled)) {
        if (!plots.length) continue;
        const s = await suite(suiteName);
        const result = await predictTarget(globalThis.ort, s, name, assembled, plots, {
          fetchBytes: fetchBytesFrom(`${MODELS}/${suiteName}`),
          onProgress: ({ member, members, done, total }) => {
            progress.value = (t + (member + done / total) / members) / targets.length;
            $("status").textContent = `${TARGET_LABELS[name]}: model ${member + 1} of ${members}`;
          },
        });
        const put = (column, values) => {
          if (!out.has(column)) out.set(column, new Array(n).fill(null));
          const col = out.get(column);
          plots.forEach((p, i) => { col[p] = values[i]; });
        };
        for (const [column, values] of result.columns) put(column, values);
        put(`${name}_recognised`, result.recognised);
        put(`${name}_model`, plots.map(() => suiteName));
      }
    }
    state.result = out;
    renderResults(out);
    $("status").textContent = `Done: ${n.toLocaleString()} plots.`;
  } catch (e) {
    $("status").textContent = `Stopped: ${e.message}`;
  } finally {
    progress.hidden = true;
    $("run").disabled = false;
  }
}

function format(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isNaN(v) ? "" : Number.isInteger(v) ? String(v) : v.toPrecision(6);
  return String(v);
}

const PREVIEW_ROWS = 50;

function renderResults(out) {
  const table = $("preview");
  const keys = [...out.keys()].filter((k) => !k.includes("_prob_"));
  const head = document.createElement("tr");
  for (const k of keys) {
    const th = document.createElement("th");
    th.textContent = k;
    head.append(th);
  }
  const rows = [head];
  const n = out.get("plot_id").length;
  for (let i = 0; i < Math.min(n, PREVIEW_ROWS); i += 1) {
    const tr = document.createElement("tr");
    for (const k of keys) {
      const td = document.createElement("td");
      td.textContent = format(out.get(k)[i]);
      tr.append(td);
    }
    rows.push(tr);
  }
  table.replaceChildren(...rows);
  $("preview-note").textContent = n > PREVIEW_ROWS
    ? `The first ${PREVIEW_ROWS} of ${n.toLocaleString()} plots; the download holds all of them, with the class probabilities.`
    : "The download adds the class probabilities.";
  $("results").hidden = false;
}

function download() {
  const out = state.result;
  const keys = [...out.keys()];
  const quote = (v) => {
    const s = format(v);
    return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  const lines = [keys.map(quote).join(",")];
  const n = out.get("plot_id").length;
  for (let i = 0; i < n; i += 1) lines.push(keys.map((k) => quote(out.get(k)[i])).join(","));
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "resolve_predictions.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

globalThis.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
renderTargets();
$("species-file").addEventListener("change", onFiles);
$("header-file").addEventListener("change", onFiles);
$("run").addEventListener("click", run);
$("download").addEventListener("click", download);
