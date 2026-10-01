# AI-NWP Regional WeatherBench (Southeast Asia) 

Evaluates AI weather forecast models against ERA5 and GPM over a Southeast
Asia domain, across three complementary benchmarks:

- **Traditional** -- elementwise NRMSE of geopotential, temperature,
  specific humidity, and wind vector against ERA5.
- **Dynamic** -- derived-variable metrics (MSE convergence/divergence,
  vorticity convergence, surface temperature energy convergence, HKE
  spectrum) plus GPM-based precipitation skill, designed to reveal whether
  a model's fields are physically coherent, not just individually accurate.
- **Combined** -- unweighted mean across all Traditional + Dynamic +
  precipitation metrics, one score per model per lead time.

Precipitation skill uses the Fractions Skill Score (FSS) at four
thresholds and four window sizes. Window sizes are computed at runtime
from your truth grid's actual resolution to represent consistent
real-world distances (~56/139/278/556 km) regardless of native grid
spacing, so `window_gridpts` in the FSS output is not a fixed constant
across models -- it reflects each model's own precip truth resolution,
and scores stay physically comparable across models even when their
truth sources differ in resolution.

Ships with precomputed results for five models: **PanguWeather,
FourCastNet, AIFS, GraphCast, Aurora**. Supports adding your own model
through a JSON config, computed live against your own truth data.

## Usage

    python weatherbench.py --mode traditional --period full_year
    python weatherbench.py --mode dynamic --period 12cases
    python weatherbench.py --mode combined --period im
    python weatherbench.py --mode traditional --period full_year --add_model_config my_model.json

`--period`: `full_year`, `12cases`, `im` (Inter-Monsoon), `sw` (Southwest
Monsoon), `ne` (Northeast Monsoon).

`--add_model_config` is repeatable -- pass it multiple times to add
several models to the same plot alongside the 5 precomputed ones. Figures
are saved to `figures/`.

## Adding your own model

Requires a JSON config. `name` and `forecast_dir` are always required;
`pl_truth` / `sfc_truth` / `precip_truth` are each required only if the
mode/metric you're running actually needs them (`pl_truth` for
traditional, dynamic, and the HKE spectrum; `sfc_truth` for dynamic's
surface temperature energy metric; `precip_truth` for dynamic's
precipitation column). There is no default truth source -- every truth
type you use must be explicitly given a directory and variable mapping.

See `examples/` for three complete, working configs:
- `example_grib_model.json` -- a grib model with MLP-derived precipitation
  (`precip_forecast_dir` override)
- `example_netcdf_model.json` -- a netCDF model with native precipitation
  and a custom `pressure_dim`
- `example_needs_rh_to_q_model.json` -- a model whose files only contain
  relative humidity, not specific humidity directly (`needs_rh_to_q`)

### ModelConfig fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Display name for this model |
| `forecast_dir` | yes | Directory containing forecast files |
| `filename_template` | yes* | `{date}`, `{hour}`, `{lead}` placeholders, e.g. `model_{date}_{hour}-out-{lead}.grib` |
| `pressure_dim` | no | Pressure-level dimension name in your forecast files. Default `isobaricInhPa` |
| `sfc_var_style` | no | Auto-inferred from file format if unset |
| `precip_var` | no | Variable name for precipitation. Default `tp` |
| `precip_unit_scale` | no | Multiplier to convert your precip units to mm. Default `1000.0` (assumes native meters) |
| `precip_forecast_dir` | no | Override -- use a different directory for precipitation only (e.g. a separately MLP-derived precip product). Defaults to `forecast_dir` |
| `precip_forecast_filename_template` | no | Filename pattern for `precip_forecast_dir`. Defaults to `filename_template` |
| `needs_rh_to_q` | no | Set `true` if your forecast files contain relative humidity (`r`) instead of specific humidity (`q`) -- converted automatically via the Bolton (1980) approximation |
| `truth_source_label` | no | Display label for your `pl_truth`/`sfc_truth` source, shown in each row's label on Dynamic/Combined figures. Default `"ERA5"` |
| `precip_truth_source_label` | no | Display label for your `precip_truth` source, shown alongside `truth_source_label` (e.g. `"ERA5 / GPM"`). Default `"GPM"` |
| `pl_truth` | conditional | TruthSpec for pressure-level truth (z, t, q, u, v) |
| `sfc_truth` | conditional | TruthSpec for surface truth (u10, v10, t2m) |
| `precip_truth` | conditional | TruthSpec for precipitation truth |

*`filename_template` is technically optional but required in practice for
any mode that reads forecast files.

### TruthSpec fields

| Field | Required | Description |
|---|---|---|
| `dir` | yes | Directory containing truth files |
| `variables` | yes | Mapping of our internal variable key to `{"name": "...", "unit": "..."}` |
| `filename_template` | no | `{date}`, `{hour}` placeholders. Defaults to our own naming convention if unset (`era5_pl_{date}_{hour}` for `pl_truth`, `era5_sfc_{date}_{hour}_regridded` for `sfc_truth`, `gpm_acc_{date}_{hour}` for `precip_truth`) |
| `t2m_dir` | `sfc_truth` only | t2m sometimes lives in a separate directory from u10/v10 |
| `t2m_filename_template` | `sfc_truth` only | Filename pattern for `t2m_dir` |

If your truth data is a single unified source (one file per timestamp
containing everything), set the same `dir` and `filename_template` across
`pl_truth`, `sfc_truth`, and `precip_truth` -- every lookup will converge
on the same file.

### Allowed units

Only truth values are unit-converted; forecast values are always trusted
as-is in their native units.

| Variable | Allowed units |
|---|---|
| `z` | `geopotential`, `m`, `gpm` |
| `t`, `t2m` | `K`, `C` |
| `q` | `kg/kg`, `g/kg` |
| `u`, `v`, `u10`, `v10` | `m/s`, `km/h` |
| `precip` | `mm`, `m_precip`, `kg/m2` |

## Output structure

    npz/                 Cached Traditional + Dynamic + Spectra results, one file per model/period
    fss/                 Cached precipitation FSS results, one file per model/period
    figures/             Generated plots. The clean, precomputed-only plots
                          (traditional/dynamic/combined, all 5 periods) are
                          version-controlled as the repo's reference figures;
                          anything including a live-added model is regenerated
                          locally and not tracked
    reference_values/    Cached truth-derived normalization constants, one per distinct truth path
    spectra_reference/   Cached ERA5 per-lead spectra, one per distinct truth path
                          (built on first use -- not pre-shipped, even for our own default truth)
    examples/            Example model configs
    scripts/weatherbench.py   The tool itself

Any live-added model using a truth path we haven't seen before will
trigger a one-time build of that path's reference values and (if running
Dynamic or Combined) ERA5 spectra cache -- the latter can take a
meaningful amount of time on first use for a full year of data.

## Requirements

See `requirements.txt`. Requires both `.grib` and `.nc` forecast/truth
support -- `cfgrib` handles the former, `xarray`'s built-in engines the
latter.
