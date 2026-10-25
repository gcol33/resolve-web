# RESOLVE plot predictor

Predict a vegetation plot's EUNIS habitat, altitude, slope, plot area, survey
year and location from its species list, in the browser. The page runs the
released [RESOLVE](https://github.com/gcol33/resolve) model suites with
onnxruntime-web, five models per target combined as the engine combines them;
the uploaded tables are read locally and never sent anywhere.

## Layout

- `site/`: the page. `js/csv.js` reads the tables, `js/assemble.js` builds
  each target's inputs as the engine does, `js/predict.js` runs the members
  and `js/combine.js` combines them.
- `export/`: turns a released suite into the files the page reads: per
  target, `target.json` (vocabularies, inputs, heads, combine rule) and one
  ONNX graph per member.
- `tests/`: runs the page's prediction path in Node and compares it with the
  engine's `SuitePredictor` on the same plots.

## Building the models

```
python export/export_suite.py <suite dir> site/models/<suite name>
```

Each exported graph is checked against its checkpoint before it is written.
The page reads `site/models/V_species_context` for plots with coordinates and
`site/models/V_minimal` for plots without.

## Checking against the engine

```
cd tests && npm install && cd ..
node tests/run_suite.mjs site/models/V_species_context plots.csv header.csv site.csv
python tests/compare_engine.py <suite dir> plots.csv header.csv site.csv
```

`compare_engine.py` needs a Python with `resolve_core`. On 2,000 plots of the
UK National Plant Monitoring Scheme every EUNIS label agrees with the engine,
probabilities differ by less than 1e-6 and regressions by less than 5e-4 of
their value.
