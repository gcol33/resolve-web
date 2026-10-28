# RESOLVE plot predictor

Predict a vegetation plot's EUNIS habitat, altitude, slope, plot area, survey
year and location from its species list, in the browser. The page runs the
released [RESOLVE](https://github.com/gcol33/resolve) model suites with
onnxruntime-web, five models per target combined as the engine combines them;
the uploaded tables are read locally and never sent anywhere.

## Layout

- `site/`: the page. `js/csv.js` reads the tables, `js/assemble.js` builds
  each target's inputs as the engine does, `js/predict.js` runs the members
  and `js/combine.js` combines them. `coi.js` is a service worker that makes
  the page cross-origin isolated (`js/isolate.js` registers it), so the
  models run on up to eight threads; where the browser offers WebGPU they
  run on the GPU.
- `export/`: turns a released suite into the files the page reads: per
  target, `target.json` (vocabularies, inputs, heads, combine rule) and one
  ONNX graph per member.
- `tests/`: runs the page's prediction path in Node and compares it with the
  engine's `SuitePredictor` on the same plots.

## Building the models

```
python export/export_suite.py <suite dir> site/models/<suite name>
```

Each exported graph is checked against the same member in torch before it is
written. `--storage float16` halves the files and `--storage int8` (blocks of
the largest matrices, float16 for the rest) takes them to under a third, at
the cost of moving the predictions; on 2,000 NPMS plots, against float32,
float16 changed no EUNIS label and altitude by at most 7.5 m, int8 changed
altitude by 1 m on average and up to 104 m.
The page reads `site/models/V_species_context` for plots with coordinates and
`site/models/V_minimal` for plots without.

## Checking against the engine

```
cd tests && npm install && cd ..
node tests/run_suite.mjs site/models/V_species_context plots.csv header.csv site.csv
python tests/compare_engine.py <suite dir> plots.csv header.csv site.csv
```

`compare_engine.py` needs a Python with `resolve_core`.
`tests/browser_bench.py` runs the same prediction path in headless Edge
(`--ep webgpu` for the GPU, `--threads N` for the CPU) and writes the same
table; `tests/browser_bench.py --site <models dir>` checks that the page
becomes cross-origin isolated and predicts the example plot. On 2,000 plots of the
UK National Plant Monitoring Scheme every EUNIS label agrees with the engine,
probabilities differ by less than 1e-6 and regressions by less than 5e-4 of
their value.
