import { buildPlots, resolveName } from "./assemble.js";
import { readTable, tableFromColumns } from "./csv.js";
import { loadSuite, loadTarget, predictTarget } from "./predict.js";

// The model repository at a fixed revision, so a page always runs the models
// it was checked against; `?models=<url>` reads them from elsewhere.
const MODEL_REPO = "https://huggingface.co/gcol33/resolve-models/resolve";
const MODEL_REVISION = "b4e1e8a543b3f339395ff00772e1d533b86553b0";
const MODELS = new URLSearchParams(location.search).get("models") ?? `${MODEL_REPO}/${MODEL_REVISION}`;
// Plots with coordinates read the context suite; the rest the species-only one.
const SUITES = { context: "V_species_context", species: "V_minimal" };
const TARGETS = {
  eunis: "EUNIS habitat",
  altitude: "Altitude (m)",
  slope: "Slope (°)",
  area: "Plot area (m²)",
  year: "Survey year",
  location: "Location",
};
const EUNIS = {
  M: "Marine habitats",
  N: "Coastal habitats",
  P: "Inland surface waters",
  Q: "Wetlands",
  R: "Grasslands and lands dominated by forbs, mosses or lichens",
  S: "Heathland, scrub and tundra",
  T: "Forests and other wooded land",
  U: "Inland habitats with no or little soil",
  V: "Vegetated man-made habitats",
};
// The columns an uploaded table can carry, and the names each is found by.
const FIELDS = [
  { key: "plot", label: "Plot", table: "both", need: "required",
    guesses: ["PlotObservationID", "plot", "plot_id", "plotid", "releve", "relevé", "releve_id", "site"] },
  { key: "species", label: "Species", table: "species", need: "required",
    guesses: ["WFO_TAXON", "species", "taxon", "taxon_name", "scientific_name", "scientificName", "name"] },
  { key: "abundance", label: "Cover (%)", table: "species", need: "optional; every species counts the same without it",
    guesses: ["Cover %", "cover", "cover_percent", "abundance", "percentage_cover"] },
  { key: "genus", label: "Genus", table: "species", need: "optional; taken from the species name without it",
    guesses: ["WFO_GENUS", "genus"] },
  { key: "family", label: "Family", table: "species", need: "optional; taken from the species name without it",
    guesses: ["WFO_FAMILY", "family"] },
  { key: "latitude", label: "Latitude", table: "header", need: "optional; decimal degrees (WGS 84)",
    guesses: ["Latitude", "lat", "decimalLatitude", "y"] },
  { key: "longitude", label: "Longitude", table: "header", need: "optional; decimal degrees (WGS 84)",
    guesses: ["Longitude", "lon", "long", "lng", "decimalLongitude", "x"] },
  { key: "Year", label: "Year", table: "header", covariate: true, need: "optional",
    guesses: ["Year", "survey_year"] },
  { key: "Month", label: "Month", table: "header", covariate: true, need: "optional; 1 to 12",
    guesses: ["Month", "survey_month"] },
];

const $ = (id) => document.getElementById(id);
const state = { species: null, header: null, suites: {}, taxonomy: null, result: null };

async function fetchOk(path) {
  const response = await fetch(`${MODELS}/${path}`);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response;
}

function suite(name) {
  if (!state.suites[name]) state.suites[name] = loadSuite(async (p) => (await fetchOk(`${name}/${p}`)).json());
  return state.suites[name];
}

function taxonomy() {
  if (!state.taxonomy) state.taxonomy = fetchOk("taxonomy.json").then((r) => r.json());
  return state.taxonomy;
}

// ---------------------------------------------------------------- one plot

// A line's species name and cover: the cover is a number at the end of the
// line, after a tab, semicolon, comma or space.
function parseLine(line) {
  const m = line.match(/^(.*?)[\s;,]+([0-9]+(?:[.,][0-9]+)?)\s*%?\s*$/);
  return m ? { name: m[1], cover: m[2] } : { name: line, cover: "" };
}

function pastedLines() {
  return $("paste").value.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).map(parseLine);
}

async function checkPaste() {
  const tax = await taxonomy();
  const lines = pastedLines();
  const unknown = lines.filter((l) => !resolveName(l.name, tax).known).map((l) => l.name.trim());
  $("paste-check").textContent = !lines.length ? ""
    : unknown.length === 0 ? `All ${lines.length} names recognised.`
    : `${lines.length - unknown.length} of ${lines.length} names recognised; not recognised: ${unknown.join(", ")}. `
      + "Unrecognised species still count towards the plot's cover but carry no identity.";
}

async function pastedPlot() {
  const tax = await taxonomy();
  const lines = pastedLines();
  if (!lines.length) throw new Error("the species list is empty");
  const hasCover = lines.some((l) => l.cover !== "");
  const n = lines.length;
  const repeat = (v) => new Array(n).fill(v);
  const columns = {
    plot: repeat("pasted plot"),
    species: lines.map((l) => resolveName(l.name, tax).name),
    latitude: repeat($("paste-lat").value),
    longitude: repeat($("paste-lon").value),
    Year: repeat($("paste-year").value),
    Month: repeat($("paste-month").value),
  };
  if (hasCover) columns.cover = lines.map((l) => l.cover);
  const table = tableFromColumns(columns);
  return buildPlots(table, {
    plot: "plot", species: "species", abundance: hasCover ? "cover" : undefined,
    latitude: "latitude", longitude: "longitude", covariates: { Year: "Year", Month: "Month" },
  }, { taxonomy: tax });
}

// ---------------------------------------------------------------- many plots

function guess(table, field) {
  if (!table) return "";
  const lower = new Map(table.names.map((n) => [n.toLowerCase(), n]));
  for (const g of field.guesses) if (lower.has(g.toLowerCase())) return lower.get(g.toLowerCase());
  return "";
}

function renderGuide() {
  const rows = [["Column", "Found under", "Needed"]];
  for (const f of FIELDS) rows.push([f.label, f.guesses.join(", "), f.need]);
  $("column-guide").replaceChildren(...rows.map((cells, i) => {
    const tr = document.createElement("tr");
    for (const c of cells) {
      const cell = document.createElement(i === 0 ? "th" : "td");
      cell.textContent = c;
      tr.append(cell);
    }
    return tr;
  }));
}

// The table a field is read from: plot-level fields from the plot table when
// one is given.
function tableOf(field, which) {
  if (field.table === "both") return which;
  if (field.table === "header") return state.header ? "header" : "species";
  return "species";
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
      const s = document.createElement("select");
      s.id = id;
      if (field.need !== "required") s.append(new Option("(none)", ""));
      for (const name of table.names) s.append(new Option(name, name));
      s.value = guess(table, field);
      s.addEventListener("change", summarise);
      box.append(label, s);
    }
  }
  $("mapping").hidden = false;
}

function columnsFor(which) {
  const out = { covariates: {} };
  for (const field of FIELDS) {
    if (tableOf(field, which) !== which) continue;
    const value = $(`map-${which}-${field.key}`)?.value;
    if (!value) continue;
    if (field.covariate) out.covariates[field.key] = value;
    else out[field.key] = value;
  }
  return out;
}

async function uploadedPlots() {
  const tax = await taxonomy();
  const columns = columnsFor("species");
  const species = state.species;
  const resolved = species.columns.get(columns.species).map((n) => resolveName(n, tax).name);
  const table = { ...species, columns: new Map(species.columns).set(columns.species, resolved) };
  const headerColumns = state.header ? columnsFor("header") : columns;
  return buildPlots(table, columns, { header: state.header, headerColumns, zeroAs: 1, taxonomy: tax });
}

async function summarise() {
  try {
    const assembled = await uploadedPlots();
    const plots = assembled.plots;
    const located = plots.filter(located_).length;
    const entries = plots.reduce((a, p) => a + p.names.length, 0);
    const tax = await taxonomy();
    const known = plots.reduce((a, p) => a + p.names.filter((n) => n in tax.taxa).length, 0);
    $("input-summary").textContent = `${plots.length.toLocaleString()} plots, ${entries.toLocaleString()} species `
      + `entries (${((100 * known) / Math.max(entries, 1)).toFixed(1)}% of names recognised), `
      + `${located.toLocaleString()} plots with coordinates.`;
    $("run-upload").disabled = plots.length === 0;
  } catch (e) {
    $("input-summary").textContent = e.message;
    $("run-upload").disabled = true;
  }
}

async function onFiles() {
  const read = async (input) => (input.files[0] ? readTable(await input.files[0].text()) : null);
  state.species = await read($("species-file"));
  state.header = await read($("header-file"));
  if (!state.species) return;
  renderMapping();
  await summarise();
}

// ---------------------------------------------------------------- predicting

function located_(plot) {
  return plot.site.latitude != null && plot.site.longitude != null;
}

function renderTargets() {
  for (const [name, label] of Object.entries(TARGETS)) {
    const wrap = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = name;
    box.checked = true;
    wrap.append(box, ` ${label}`);
    $("target-list").append(wrap);
  }
}

// For one target, the plots each suite answers: a target that reads
// coordinates goes to the species-only suite for plots without them.
async function routes(name, assembled) {
  const context = await suite(SUITES.context);
  const all = [...assembled.plots.keys()];
  if (!context.targets.has(name)) return [[SUITES.species, all]];
  const target = await loadTarget(context, name);
  if (!target.schema.has_coordinates) return [[SUITES.context, all]];
  const with_ = all.filter((p) => located_(assembled.plots[p]));
  return [[SUITES.context, with_], [SUITES.species, all.filter((p) => !located_(assembled.plots[p]))]];
}

async function predict(assembled) {
  const n = assembled.plots.length;
  const out = new Map([["plot_id", assembled.plots.map((p) => p.id)]]);
  const targets = [...$("target-list").querySelectorAll("input:checked")].map((b) => b.value);
  const progress = $("progress");
  for (const [t, name] of targets.entries()) {
    for (const [suiteName, plots] of await routes(name, assembled)) {
      if (!plots.length) continue;
      const result = await predictTarget(globalThis.ort, await suite(suiteName), name, assembled, plots, {
        fetchBytes: async (p) => (await fetchOk(`${suiteName}/${p}`)).arrayBuffer(),
        onProgress: ({ member, members, done, total }) => {
          progress.value = (t + (member + done / total) / members) / targets.length;
          $("status").textContent = `${TARGETS[name]}: model ${member + 1} of ${members}`;
        },
      });
      const put = (column, values) => {
        if (!out.has(column)) out.set(column, new Array(n).fill(null));
        const col = out.get(column);
        plots.forEach((p, i) => { col[p] = values[i]; });
      };
      for (const [column, values] of result.columns) put(column, values);
      put(`${name}_model`, plots.map(() => suiteName));
    }
  }
  return out;
}

async function run(source) {
  const buttons = [$("run-paste"), $("run-upload")];
  buttons.forEach((b) => { b.disabled = true; });
  $("results").hidden = false;
  $("progress").hidden = false;
  $("preview").replaceChildren();
  $("preview-note").textContent = "";
  $("download").hidden = true;
  try {
    const assembled = source === "paste" ? await pastedPlot() : await uploadedPlots();
    const out = await predict(assembled);
    state.result = out;
    if (source === "paste") renderOne(out);
    else renderMany(out);
    $("status").textContent = source === "paste" ? ""
      : `Done: ${assembled.plots.length.toLocaleString()} plots.`;
    $("download").hidden = false;
  } catch (e) {
    $("status").textContent = `Stopped: ${e.message}`;
  } finally {
    $("progress").hidden = true;
    $("run-paste").disabled = false;
    $("run-upload").disabled = !state.species;
  }
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- results

function format(v, digits = 6) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isNaN(v) ? "" : Number.isInteger(v) ? String(v) : v.toPrecision(digits);
  return String(v);
}

function row(cells, header = false) {
  const tr = document.createElement("tr");
  for (const c of cells) {
    const cell = document.createElement(header ? "th" : "td");
    cell.textContent = c;
    tr.append(cell);
  }
  return tr;
}

const round = (v, step) => (v === null || Number.isNaN(v) ? "" : String(Math.round(v / step) * step));

function renderOne(out) {
  const get = (k) => out.get(k)?.[0];
  const rows = [row(["", "Prediction", "Spread over the five models"], true)];
  if (out.has("eunis")) {
    const code = get("eunis");
    rows.push(row(["EUNIS habitat", `${code}: ${EUNIS[code] ?? ""}`,
      `probability ${format(get(`eunis_prob_${code}`), 2)}`]));
  }
  const regression = [["altitude", "Altitude", " m", 1], ["slope", "Slope", "°", 0.1],
    ["area", "Plot area", " m²", 1], ["year", "Survey year", "", 1]];
  for (const [key, label, unit, step] of regression) {
    if (!out.has(key)) continue;
    rows.push(row([label, `${Number(round(get(key), step)).toLocaleString()}${unit}`,
      `± ${Number(round(get(`${key}_sd`), step)).toLocaleString()}${unit}`]));
  }
  if (out.has("location_latitude")) {
    const lat = get("location_latitude");
    const lon = get("location_longitude");
    rows.push(row(["Location", `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"}, `
      + `${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}`, ""]));
  }
  $("preview").replaceChildren(...rows);
  const models = [...out.keys()].filter((k) => k.endsWith("_model")).map((k) => get(k));
  $("preview-note").textContent = models.includes(SUITES.species)
    ? "Without coordinates the species-only models were used."
    : "";
}

const PREVIEW_ROWS = 50;

function renderMany(out) {
  const keys = [...out.keys()].filter((k) => !k.includes("_prob_") && !k.endsWith("_model"));
  const n = out.get("plot_id").length;
  const rows = [row(keys, true)];
  for (let i = 0; i < Math.min(n, PREVIEW_ROWS); i += 1) rows.push(row(keys.map((k) => format(out.get(k)[i]))));
  $("preview").replaceChildren(...rows);
  $("preview-note").textContent = n > PREVIEW_ROWS
    ? `The first ${PREVIEW_ROWS} of ${n.toLocaleString()} plots. The download holds all of them, with the class probabilities and the models each plot used.`
    : "The download adds the class probabilities and the models each plot used.";
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
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  a.download = "resolve_predictions.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

globalThis.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
renderTargets();
renderGuide();
checkPaste();
$("paste").addEventListener("input", checkPaste);
$("species-file").addEventListener("change", onFiles);
$("header-file").addEventListener("change", onFiles);
$("run-paste").addEventListener("click", () => run("paste"));
$("run-upload").addEventListener("click", () => run("upload"));
$("download").addEventListener("click", download);
