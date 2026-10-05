#!/usr/bin/env python3
"""
weatherbench.py -- single-script WeatherBench (Traditional / Dynamic / Combined).

BUILD STATUS: All three modes complete (precomputed + live add-your-own-model,
including per-lead HKE spectra and GPM-based precip FSS).

Usage:
  python weatherbench.py --mode traditional --period full_year
  python weatherbench.py --mode dynamic --period 12cases --add_model_config mymodel.json
  python weatherbench.py --mode combined --period full_year --add_model_config mymodel.json

Adding your own model requires a JSON config. There is no default truth
source -- every truth type your requested metrics need must be explicitly
specified; there is no fallback or shortcut. Truth files must follow this
repo's filename convention (era5_pl_YYYY-MM-DD_HH, era5_sfc_YYYY-MM-DD_HH
_regridded, era5_sfc_YYYY-MM-DD_HH_t2m_regridded, gpm_acc_YYYY-MM-DD_HH --
each as either .grib or .nc); only the directory path is yours to choose.
If your truth source is a single unified dataset, point all three truth
dirs at the same path (see the docstring in resolve_pl_truth etc. for
current limitations on this).

mymodel.json fields ("name" and "forecast_dir" are always required;
pl_truth/sfc_truth/precip_truth are each required only if you request a
mode/metric that needs them -- pl_truth for traditional, dynamic, and
spectra; sfc_truth for dynamic's surface_temperature_energy; precip_truth
for dynamic's precip column. truth_source_label/precip_truth_source_label
are optional display strings shown in each row's label on Dynamic/Combined
figures -- e.g. "ModelName (ERA5 / GPM)" -- and default to "ERA5"/"GPM" if
omitted):
  {
    "name": "mymodel",
    "forecast_dir": "/path/to/forecasts",
    "filename_template": "mymodel_{date}_{hour}-out-{lead}.grib",
    "pressure_dim": "isobaricInhPa",
    "truth_source_label": "ERA5",
    "precip_truth_source_label": "GPM",
    "pl_truth": {
      "dir": "/path/to/your/pl/truth",
      "variables": {
        "z": {"name": "z", "unit": "geopotential"},
        "t": {"name": "t", "unit": "K"},
        "q": {"name": "q", "unit": "kg/kg"},
        "u": {"name": "u", "unit": "m/s"},
        "v": {"name": "v", "unit": "m/s"}
      }
    },
    "sfc_truth": {
      "dir": "/path/to/your/sfc/truth",
      "variables": {
        "u10": {"name": "u10", "unit": "m/s"},
        "v10": {"name": "v10", "unit": "m/s"},
        "t2m": {"name": "t2m", "unit": "K"}
      }
    },
    "precip_truth": {
      "dir": "/path/to/your/precip/truth",
      "variables": {
        "precip": {"name": "precipitation", "unit": "mm"}
      }
    }
  }

See ALLOWED_UNITS below for the full list of accepted units per variable --
config loading validates these immediately and fails fast on anything
unsupported, before any computation starts.
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
import json
import re
import logging
logging.getLogger("cfgrib").setLevel(logging.ERROR)
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch
import math
import warnings
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
NPZ_DIR = REPO / "npz"
FSS_DIR = REPO / "fss"
REF_DIR = REPO / "reference_values"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

PRECOMPUTED_MODELS = ["pangu", "fourcastnet", "aifs", "graphcast", "aurora"]
PRECOMPUTED_TRUTH_LABEL = "ERA5"
PRECOMPUTED_PRECIP_TRUTH_LABEL = "GPM"
MODEL_DISPLAY = {"pangu": "PanguWeather", "fourcastnet": "FourcastNet", "aifs": "AIFS",
                  "graphcast": "Graphcast", "aurora": "Aurora"}

DOMAIN = {"lat_min": -10, "lat_max": 25, "lon_min": 90, "lon_max": 140}
LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]
LEADS = [12, 24, 36, 48]
GRAVITY = 9.80665
TRAD_VARS = [("z", 500), ("t", 850), ("q", 700), ("wind", 850)]
TRAD_DISPLAY = {"z": "Geopotential\n500 hPa", "t": "Temperature\n850 hPa",
                 "q": "Specific Humidity\n700 hPa", "wind": "Wind Vector\n850 hPa"}

MONSOON_PERIODS = {"Inter-monsoon": [4, 5, 10, 11], "Southwest-monsoon": [6, 7, 8, 9], "Northeast-monsoon": [12, 1, 2, 3]}
TWELVE_CASE_DATES = {"2024-01-24_00", "2024-02-06_00", "2024-03-06_00", "2024-04-12_00", "2024-05-04_00",
                      "2024-06-21_12", "2024-07-12_06", "2024-08-09_00", "2024-09-17_06", "2024-10-14_00",
                      "2024-11-16_00", "2024-12-23_00"}
PERIOD_CHOICES = {"full_year": "full_year", "12cases": "12cases",
                   "im": "Inter-monsoon", "sw": "Southwest-monsoon", "ne": "Northeast-monsoon"}

TARGET_UNITS = {"z": "geopotential", "t": "K", "q": "kg/kg", "u": "m/s", "v": "m/s",
                "u10": "m/s", "v10": "m/s", "t2m": "K", "precip": "mm"}

UNIT_CONVERSIONS = {
    ("K", "K"): lambda x: x,
    ("C", "K"): lambda x: x + 273.15,
    ("kg/kg", "kg/kg"): lambda x: x,
    ("g/kg", "kg/kg"): lambda x: x / 1000.0,
    ("m/s", "m/s"): lambda x: x,
    ("km/h", "m/s"): lambda x: x / 3.6,
    ("geopotential", "geopotential"): lambda x: x,       # matches forecast's assumed native units
    ("m", "geopotential"): lambda x: x * GRAVITY,        # height -> geopotential
    ("gpm", "geopotential"): lambda x: x * GRAVITY,      # geopotential meters -> geopotential
    ("mm", "mm"): lambda x: x,
    ("m_precip", "mm"): lambda x: x * 1000.0,
    ("kg/m2", "mm"): lambda x: x,
}

def get_truth_labels(model_key, add_configs):
    if model_key in PRECOMPUTED_MODELS:
        return PRECOMPUTED_TRUTH_LABEL, PRECOMPUTED_PRECIP_TRUTH_LABEL
    cfg = add_configs.get(model_key)
    if cfg is not None:
        return cfg.truth_source_label, cfg.precip_truth_source_label
    return PRECOMPUTED_TRUTH_LABEL, PRECOMPUTED_PRECIP_TRUTH_LABEL

def convert_units(arr, from_unit, to_unit):
    if from_unit is None or from_unit == to_unit:
        return arr
    key = (from_unit, to_unit)
    if key not in UNIT_CONVERSIONS:
        raise ValueError(f"No known unit conversion from '{from_unit}' to '{to_unit}'. "
                          f"Known source units: {[k[0] for k in UNIT_CONVERSIONS if k[1] == to_unit]}")
    return UNIT_CONVERSIONS[key](arr)

@dataclass
class VarSpec:
    name: str            # variable name as it appears in the user's file
    unit: str = None     # None -> assume already in our target unit, no conversion

@dataclass
class TruthSpec:
    dir: str
    variables: dict = field(default_factory=dict)
    filename_template: str = None            # {date}, {hour} placeholders; extension auto-detected
    t2m_dir: str = None                      # sfc_truth only
    t2m_filename_template: str = None        # sfc_truth only

def load_truth_spec(d, default_filename_template=None, default_t2m_filename_template=None):
    if d is None:
        return None
    variables = {k: VarSpec(**v) for k, v in d.get("variables", {}).items()}
    return TruthSpec(
        dir=d["dir"],
        variables=variables,
        filename_template=d.get("filename_template", default_filename_template),
        t2m_dir=d.get("t2m_dir"),
        t2m_filename_template=d.get("t2m_filename_template", default_t2m_filename_template),
    )

def get_truth_var(ds, internal_key, spec, level=None):
    var_spec = spec.variables[internal_key]
    da = ds[var_spec.name]
    if level is not None:
        da = da.sel(isobaricInhPa=level)   # truth pressure-dim name assumed "isobaricInhPa" always
    return convert_units(da.values, var_spec.unit, TARGET_UNITS[internal_key])

def glob_truth_files(truth_dir, prefix, suffix_extra=""):
    files = list(Path(truth_dir).glob(f"{prefix}_*{suffix_extra}.grib")) + \
            list(Path(truth_dir).glob(f"{prefix}_*{suffix_extra}.nc"))
    return sorted(files)

def open_truth_file(fpath, filter_pl=True):
    fpath = Path(fpath)
    if fpath.suffix == ".grib":
        kwargs = {"indexpath": ""}
        if filter_pl:
            kwargs["filter_by_keys"] = {"typeOfLevel": "isobaricInhPa"}
        return xr.open_dataset(fpath, engine="cfgrib", backend_kwargs=kwargs)
    return xr.open_dataset(fpath)

def build_truth_file_regex(filename_template):
    pattern = re.escape(filename_template)
    pattern = pattern.replace(re.escape("{date}"), r"(\d{4}-\d{2}-\d{2})")
    pattern = pattern.replace(re.escape("{hour}"), r"(\d{2})")
    pattern += r"\.(?:grib|nc)$"
    return re.compile(pattern)

def glob_truth_files_by_template(truth_dir, filename_template):
    regex = build_truth_file_regex(filename_template)
    return sorted(f for f in Path(truth_dir).iterdir() if regex.match(f.name))

def find_truth_file_by_template(truth_dir, filename_template, date_str, hour_str):
    base = filename_template.format(date=date_str, hour=hour_str)
    for ext in [".grib", ".nc"]:
        candidate = Path(truth_dir) / f"{base}{ext}"
        if candidate.exists():
            return candidate
    return None

@dataclass
class ModelConfig:
    name: str
    forecast_dir: str
    filename_template: str = None
    pressure_dim: str = "isobaricInhPa"
    sfc_var_style: str = None
    precip_var: str = "tp"
    precip_unit_scale: float = 1000.0
    precip_forecast_dir: str = None
    precip_forecast_filename_template: str = None
    truth_source_label: str = "ERA5"
    precip_truth_source_label: str = "GPM"
    pl_truth: TruthSpec = None
    sfc_truth: TruthSpec = None
    precip_truth: TruthSpec = None
    needs_rh_to_q: bool = False

def sci_round(v, decimals=1):
    if v == 0 or np.isnan(v):
        return v
    exponent = int(np.floor(np.log10(abs(v))))
    mantissa = round(v / (10 ** exponent), decimals)
    if abs(mantissa) >= 10:
        mantissa /= 10
        exponent += 1
    return mantissa * (10 ** exponent)

ALLOWED_UNITS = {
    "z": ["geopotential", "m", "gpm"],
    "t": ["K", "C"],
    "q": ["kg/kg", "g/kg"],
    "u": ["m/s", "km/h"],
    "v": ["m/s", "km/h"],
    "u10": ["m/s", "km/h"],
    "v10": ["m/s", "km/h"],
    "t2m": ["K", "C"],
    "precip": ["mm", "m_precip", "kg/m2"],
}

def validate_truth_spec(spec, spec_name, model_name):
    if spec is None:
        return
    for key, var_spec in spec.variables.items():
        if key not in ALLOWED_UNITS:
            raise ValueError(f"Model '{model_name}': unknown variable key '{key}' in {spec_name}. "
                              f"Valid keys: {list(ALLOWED_UNITS.keys())}")
        if var_spec.unit is not None and var_spec.unit not in ALLOWED_UNITS[key]:
            raise ValueError(f"Model '{model_name}': unit '{var_spec.unit}' for '{key}' in {spec_name} "
                              f"is not supported. Allowed units for '{key}': {ALLOWED_UNITS[key]}")

def load_model_config(path):
    with open(path) as f:
        d = json.load(f)
    d["pl_truth"] = load_truth_spec(d.get("pl_truth"), default_filename_template="era5_pl_{date}_{hour}")
    d["sfc_truth"] = load_truth_spec(d.get("sfc_truth"),
                                       default_filename_template="era5_sfc_{date}_{hour}_regridded",
                                       default_t2m_filename_template="era5_sfc_{date}_{hour}_t2m_regridded")
    d["precip_truth"] = load_truth_spec(d.get("precip_truth"), default_filename_template="gpm_acc_{date}_{hour}")
    config = ModelConfig(**d)
    validate_truth_spec(config.pl_truth, "pl_truth", config.name)
    validate_truth_spec(config.sfc_truth, "sfc_truth", config.name)
    validate_truth_spec(config.precip_truth, "precip_truth", config.name)
    return config

def resolve_pl_truth(config):
    if config.pl_truth is None:
        raise ValueError(
            f"Model '{config.name}' has no pl_truth set. Please provide pl_truth "
            f"(a 'dir' and a 'variables' mapping for z, t, q, u, v -- name + unit for "
            f"each) in your model config, the same way you provide 'forecast_dir'. "
            f"Truth files must still follow our naming convention (era5_pl_YYYY-MM-DD_HH, "
            f"either .grib or .nc)."
        )
    return config.pl_truth

def resolve_sfc_truth(config):
    if config.sfc_truth is None:
        raise ValueError(
            f"Model '{config.name}' has no sfc_truth set (needed for Surface Temperature "
            f"Energy Convergence). Please provide sfc_truth (a 'dir' and a 'variables' "
            f"mapping for u10, v10, t2m -- name + unit for each) in your model config."
        )
    return config.sfc_truth

def resolve_precip_truth(config):
    if config.precip_truth is None:
        raise ValueError(
            f"Model '{config.name}' has no precip_truth set. Please provide precip_truth "
            f"(a 'dir' and a 'variables' mapping for precip -- name + unit) in your model "
            f"config. If your other truth already contains precip, point this at the same "
            f"directory."
        )
    return config.precip_truth

def resolve_precip_forecast_dir(config):
    return config.precip_forecast_dir or config.forecast_dir

def resolve_precip_forecast_filename_template(config):
    return config.precip_forecast_filename_template or config.filename_template

def rh_to_specific_humidity(rh, t, p):
    """Bolton (1980) approximation, verbatim from the original per-model
    engine -- confirmed correct via the 80/80 exact match against Zach's
    published FourCastNet numbers."""
    t_c = t - 273.15
    e_s = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e = rh / 100.0 * e_s
    q = 0.622 * e / (p - 0.378 * e)
    return q

def get_forecast_pl_var(ds_fc, var, level, config):
    """Reads a pressure-level variable from a forecast dataset. Handles
    the one confirmed real-world case: models (FourCastNet) that only
    provide relative humidity (shortName 'r'), not specific humidity
    directly."""
    if var == "q" and config.needs_rh_to_q:
        r = ds_fc["r"].sel({config.pressure_dim: level}).values
        t = ds_fc["t"].sel({config.pressure_dim: level}).values
        return rh_to_specific_humidity(r, t, level)
    return ds_fc[var].sel({config.pressure_dim: level}).values

SFC_TRUTH_PATTERN = re.compile(r"era5_sfc_(\d{4}-\d{2}-\d{2})_(\d{2})_regridded\.(?:grib|nc)")

def find_t2m_file(sfc_truth, date_str, hour_str):
    t2m_dir = sfc_truth.t2m_dir or sfc_truth.dir
    return find_truth_file_by_template(t2m_dir, sfc_truth.t2m_filename_template, date_str, hour_str)
# ---- universal domain clipping -- no needs_lat_flip needed, works for any native order ----
def clip_domain(ds):
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    ds = ds.sortby(lat_name).sortby(lon_name)
    return ds.sel({lat_name: slice(DOMAIN["lat_min"], DOMAIN["lat_max"]),
                    lon_name: slice(DOMAIN["lon_min"], DOMAIN["lon_max"])})

def parse_truth_datetime(fname):
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})", fname)
    if not m:
        raise ValueError(f"Could not parse datetime from {fname}")
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}:00")

def filter_truth_files(truth_files, mode, season=None):
    if mode == "full_year":
        return truth_files
    elif mode == "monsoon":
        months = MONSOON_PERIODS[season]
        return [f for f in truth_files if parse_truth_datetime(f.name).month in months]
    elif mode == "12cases":
        return [f for f in truth_files if parse_truth_datetime(f.name).strftime("%Y-%m-%d_%H") in TWELVE_CASE_DATES]
    raise ValueError(f"Unknown mode: {mode}")

def suffix_to_mode_season(suffix):
    if suffix == "full_year": return "full_year", None
    if suffix == "12cases": return "12cases", None
    return "monsoon", suffix

# ---- live Traditional RMSE computation (mirrors traditional_weatherbench_generalized.py) ----
def compute_rmse_for_file(truth_file, config):
    truth_time = parse_truth_datetime(truth_file.name)
    pl_truth = resolve_pl_truth(config)
    results = {}
    try:
        ds_truth = clip_domain(open_truth_file(truth_file))
    except Exception:
        return None

    for lead in LEAD_TIMES:
        init_time = truth_time - pd.Timedelta(hours=lead)
        fc_path = Path(config.forecast_dir) / config.filename_template.format(
            date=init_time.strftime("%Y-%m-%d"), hour=init_time.strftime("%H"), lead=lead)
        if not fc_path.exists():
            continue
        try:
            ds_fc = clip_domain(open_pl(fc_path, config))
        except Exception:
            continue

        for var in ["z", "t", "q", "u", "v"]:
            if var not in pl_truth.variables:
                continue
            level = {"z": 500, "t": 850, "q": 700, "u": 850, "v": 850}[var]
            try:
                truth_v = get_truth_var(ds_truth, var, pl_truth, level=level)
                fc_v = get_forecast_pl_var(ds_fc, var, level, config)
                rmse = float(np.sqrt(np.nanmean((truth_v - fc_v) ** 2)))
                truth_mean = float(np.nanmean(truth_v))
                results.setdefault(var, {}).setdefault(level, {})[lead] = {"rmse": rmse, "truth_mean": truth_mean}
            except Exception:
                continue

        if "u" in pl_truth.variables and "v" in pl_truth.variables:
            try:
                u_t = get_truth_var(ds_truth, "u", pl_truth, level=850)
                v_t = get_truth_var(ds_truth, "v", pl_truth, level=850)
                u_f = ds_fc["u"].sel({config.pressure_dim: 850}).values
                v_f = ds_fc["v"].sel({config.pressure_dim: 850}).values
                rmse_u = float(np.sqrt(np.nanmean((u_t - u_f) ** 2)))
                rmse_v = float(np.sqrt(np.nanmean((v_t - v_f) ** 2)))
                rmse_wind = float(np.sqrt(rmse_u ** 2 + rmse_v ** 2))
                mean_u, mean_v = float(np.nanmean(u_t)), float(np.nanmean(v_t))
                truth_mean_wind = float(np.sqrt(mean_u ** 2 + mean_v ** 2))
                results.setdefault("wind", {}).setdefault(850, {})[lead] = {"rmse": rmse_wind, "truth_mean": truth_mean_wind}
            except Exception:
                pass
    return results

def aggregate_and_normalize(all_results, ref):
    combined_rmse = {}
    for r in all_results:
        if r is None: continue
        for var, levels in r.items():
            for level, leads in levels.items():
                for lead, vals in leads.items():
                    combined_rmse.setdefault(var, {}).setdefault(level, {}).setdefault(lead, []).append(vals["rmse"])
    out = {}
    for var, level in TRAD_VARS:
        out[var] = {}
        for lt in LEADS:
            vals = combined_rmse.get(var, {}).get(level, {}).get(lt)
            if not vals:
                out[var][lt] = np.nan
                continue
            raw = float(np.mean(vals))
            out[var][lt] = normalize_trad(var, raw, ref)
    return out

def normalize_trad(var, raw, ref):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)): return np.nan
    if var == "z": return (raw / GRAVITY) / (ref["z"][500] / GRAVITY)
    if var == "t": return raw / ref["t"][850]
    if var == "q": return (raw * 1000) / (ref["q"][700] * 1000)
    if var == "wind": return raw / ref["wind"][850]
    return raw

def compute_truth_reference(pl_truth):
    truth_files = glob_truth_files_by_template(pl_truth.dir, pl_truth.filename_template)
    print(f"Computing reference values over {len(truth_files)} truth files...")
    sums = {v: 0.0 for v in ["z", "t", "q"]}
    counts = {v: 0 for v in ["z", "t", "q"]}
    wind_sum, wind_n = 0.0, 0
    levels = {"z": 500, "t": 850, "q": 700}

    for f in truth_files:
        try:
            ds = clip_domain(open_truth_file(f))
        except Exception:
            continue
        for var, level in levels.items():
            if var not in pl_truth.variables: continue
            vals = get_truth_var(ds, var, pl_truth, level=level)
            valid = vals[~np.isnan(vals)]
            sums[var] += valid.sum(); counts[var] += valid.size
        if "u" in pl_truth.variables and "v" in pl_truth.variables:
            mean_u = np.nanmean(get_truth_var(ds, "u", pl_truth, level=850))
            mean_v = np.nanmean(get_truth_var(ds, "v", pl_truth, level=850))
            wind_sum += np.sqrt(mean_u**2 + mean_v**2); wind_n += 1

    ref = {var: {level: (sums[var]/counts[var] if counts[var] else np.nan)} for var, level in levels.items()}
    ref["wind"] = {850: (wind_sum/wind_n if wind_n else np.nan)}
    return ref

def get_reference_values(pl_truth):
    if pl_truth is None:
        path = REF_DIR / "era5_traditional_truth_mean.npz"
        if not path.exists():
            raise FileNotFoundError("Default reference values not found -- run setup first.")
        return np.load(path, allow_pickle=True)["truth_mean"].item()

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", pl_truth.dir).strip("_")
    path = REF_DIR / f"era5_traditional_truth_mean_{safe_name}.npz"
    if path.exists():
        print(f"  Using cached reference values -> {path}")
        return np.load(path, allow_pickle=True)["truth_mean"].item()

    ref = compute_truth_reference(pl_truth)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(path, truth_mean=ref)
    print(f"  Saved new reference values -> {path}")
    return ref

def get_traditional_data(model_key, period, add_configs):
    if model_key in PRECOMPUTED_MODELS:
        path = NPZ_DIR / f"era5_vs_{model_key}_combined_{period}.npz"
        if not path.exists():
            print(f"MISSING: {path}")
            return {v: {lt: np.nan for lt in LEADS} for v, _ in TRAD_VARS}
        d = np.load(path, allow_pickle=True)
        trad_raw = d["traditional"].item()
        ref = get_reference_values(None)
        return {var: {lt: normalize_trad(var, trad_raw.get(var, {}).get(level, {}).get(lt, np.nan), ref)
                       for lt in LEADS} for var, level in TRAD_VARS}
    else:
        config = add_configs[model_key]
        pl_truth = resolve_pl_truth(config)
        out_path = NPZ_DIR / f"era5_vs_{model_key}_combined_{period}.npz"
        ref = get_reference_values(pl_truth)

        if out_path.exists():
            d = np.load(out_path, allow_pickle=True)
            if "traditional" in d:
                print(f"  Using cached Traditional data for {config.name} ({period}) -> {out_path}")
                trad_raw = d["traditional"].item()
                return {var: {lt: normalize_trad(var, trad_raw.get(var, {}).get(level, {}).get(lt, np.nan), ref)
                               for lt in LEADS} for var, level in TRAD_VARS}

        mode, season = suffix_to_mode_season(period)
        truth_files = filter_truth_files(glob_truth_files_by_template(pl_truth.dir, pl_truth.filename_template), mode, season)
        print(f"  Computing live Traditional for {config.name} ({period}): {len(truth_files)} truth files")
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(compute_rmse_for_file, tf, config) for tf in truth_files]
            all_results = [f.result() for f in as_completed(futs)]

        combined_rmse = {}
        for r in all_results:
            if r is None: continue
            for var, levels in r.items():
                for level, leads in levels.items():
                    for lead, vals in leads.items():
                        combined_rmse.setdefault(var, {}).setdefault(level, {}).setdefault(lead, []).append(vals["rmse"])
        raw = {var: {level: {lt: float(np.mean(v)) for lt, v in leads.items()} for level, leads in levels.items()}
               for var, levels in combined_rmse.items()}

        existing = {}
        if out_path.exists():
            d = np.load(out_path, allow_pickle=True)
            existing = {k: d[k].item() for k in d.files}
        existing["traditional"] = raw
        NPZ_DIR.mkdir(exist_ok=True)
        np.savez(out_path, **existing)
        print(f"  Saved -> {out_path}")

        return aggregate_and_normalize(all_results, ref)

# ---- plotting (unchanged logic, now driven by get_traditional_data for any model) ----
def plot_traditional(model_keys, add_configs, period, outname):
    suffix = PERIOD_CHOICES[period]
    matrices = {}
    for m in model_keys:
        data = get_traditional_data(m, suffix, add_configs)
        matrices[m] = np.array([[data[v][lt] for lt in LEADS] for v, _ in TRAD_VARS])

    all_matrices = [matrices[m] for m in model_keys]
    n_rows, n_vars = len(model_keys), len(TRAD_VARS)

    best_mask = [np.zeros_like(m, dtype=bool) for m in all_matrices]
    second_mask = [np.zeros_like(m, dtype=bool) for m in all_matrices]
    for var_idx in range(n_vars):
        for lead_idx in range(len(LEADS)):
            col = np.array([mat[var_idx, lead_idx] for mat in all_matrices])
            if np.all(np.isnan(col)): continue
            col_r = np.array([sci_round(x) for x in col])
            sorted_idx = np.argsort(np.nan_to_num(col_r, nan=np.inf))
            best_val = col_r[sorted_idx[0]]
            tied = [i for i in range(n_rows) if not np.isnan(col_r[i]) and col_r[i] == best_val]
            for idx in tied: best_mask[idx][var_idx, lead_idx] = True
            if len(tied) == 1:
                second_val = None
                for idx in sorted_idx:
                    if not np.isnan(col_r[idx]) and col_r[idx] != best_val:
                        second_val = col_r[idx]; break
                if second_val is not None:
                    for idx in range(n_rows):
                        if not np.isnan(col_r[idx]) and col_r[idx] == second_val:
                            second_mask[idx][var_idx, lead_idx] = True

    combined = np.vstack(all_matrices)
    vmin, vmax = np.nanmin(combined), np.nanmax(combined)
    cmap = plt.get_cmap("Reds")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def sci_format(v, decimals=1):
        if v == 0 or np.isnan(v): return "0.0e0"
        exponent = int(np.floor(np.log10(abs(v))))
        mantissa = v / (10 ** exponent)
        if round(mantissa, decimals) >= 10:
            mantissa /= 10; exponent += 1
        return f"{mantissa:.{decimals}f}e{exponent}"

    gap = 0.1
    fig, axes = plt.subplots(n_rows, n_vars, figsize=(n_vars * 3.5, n_rows * 0.8),
                              sharex=True, sharey=True, gridspec_kw={"wspace": 0.15, "hspace": 0.1})
    if n_rows == 1: axes = np.array([axes])
    if n_vars == 1: axes = axes.reshape(n_rows, 1)

    for i, m in enumerate(model_keys):
        matrix = all_matrices[i]
        for j, ax in enumerate(axes[i]):
            for spine in ax.spines.values(): spine.set_visible(False)
            for k, lt in enumerate(LEADS):
                v = matrix[j, k]
                xl, xr = k + gap/2, k + 1 - gap/2
                if np.isnan(v):
                    ax.fill_between([xl, xr], 0, 1, color=(0.95,0.95,0.95,1.0))
                    ax.text((xl+xr)/2, 0.5, "NaN", ha="center", va="center", fontsize=8, color="black")
                    continue
                ax.fill_between([xl, xr], 0, 1, color=cmap(norm(v)))
                ax.text((xl+xr)/2, 0.5, sci_format(v), ha="center", va="center", fontsize=7.5, color="black")
                if best_mask[i][j, k]:
                    ax.add_patch(Rectangle((xl+0.04, 0.04), 0.89, 0.89, fill=False, edgecolor="green", linewidth=1.5))
                if second_mask[i][j, k]:
                    ax.add_patch(Rectangle((xl+0.08, 0.08), 0.83, 0.83, fill=False, edgecolor="blue", linewidth=1.5))
            if i == 0:
                ax.set_title(TRAD_DISPLAY[TRAD_VARS[j][0]] + " NRMSE", fontsize=9, pad=10)
            if i == n_rows - 1:
                ax.set_xticks(np.arange(len(LEADS)) + 0.5)
                ax.set_xticklabels(LEADS, fontsize=8)
                ax.set_xlabel("Lead (h)", fontsize=8)
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_yticks([]); ax.set_xlim(0, len(LEADS)); ax.set_ylim(0, 1)
        pl_sfc_label, _ = get_truth_labels(m, add_configs)
        display = f"{MODEL_DISPLAY.get(m, m)}\n({pl_sfc_label})"
        axes[i, 0].set_ylabel(display, rotation=0, ha="right", va="center", fontsize=10, labelpad=20)

    legend_handles = [Patch(edgecolor="green", facecolor="none", linewidth=1.5, label="Best"),
                       Patch(edgecolor="blue", facecolor="none", linewidth=1.5, label="Second Best")]
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.2, extend="max")
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(2.0, 0), bbox_transform=cbar.ax.transAxes, fontsize=9, frameon=False)
    cbar.set_label("Normalised RMSE", fontsize=9)
    cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{(vmin+vmax)/2:.2f}", f"{vmax:.2f}"])

    out_path = FIG_DIR / outname
    csv_rows = []
    for m in model_keys:
        for idx, (v, _) in enumerate(TRAD_VARS):
            row = {"model": MODEL_DISPLAY.get(m, m), "variable": TRAD_DISPLAY[v].replace("\n", " ")}
            row.update({f"lead_{lt}h": matrices[m][idx, k] for k, lt in enumerate(LEADS)})
            csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(out_path.with_suffix(".csv"), index=False)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

# ============================================================
# DYNAMIC MODE
# ============================================================
DYN_VARS = ["mse_convergence", "mse_divergence", "vorticity_convergence", "surface_temperature_energy"]
DYN_DISPLAY = {
    "mse_convergence": "MSE Convergence\n850 hPa",
    "mse_divergence": "MSE Divergence\n200 hPa",
    "vorticity_convergence": "Vorticity Convergence\n850 hPa",
    "surface_temperature_energy": "* Surface Temperature\nEnergy Convergence",
}
PRESSURE_LEVELS_MSE = [850, 200]
CP, G, LV, RE = 1004, 9.81, 2.5e6, 6371000

def compute_mse(t, z, q, cp=CP, g=G, lv=LV):
    return cp * t + g * (z / g) + lv * q

def safe_gradient(arr, axis, spacing):
    arr_copy = np.where(np.isnan(arr), np.nan_to_num(arr, nan=0.0), arr)
    return np.gradient(arr_copy, axis=axis) / spacing

def compute_mse_vorticity_convergence(u, v, lats, lons, mse):
    dlat = np.deg2rad(np.gradient(lats))
    dlon = np.deg2rad(np.gradient(lons))
    coslat = np.clip(np.cos(np.deg2rad(lats)), 1e-3, 1.0)
    zeta = (np.gradient(v, axis=-1) / (dlon[None, :] * RE * coslat[:, None])
            - np.gradient(u, axis=-2) / (dlat[:, None] * RE))
    mse_dudx = np.gradient(u * mse, axis=-1) / (dlon[None, :] * RE * coslat[:, None])
    mse_dvdy = np.gradient(v * mse, axis=-2) / (dlat[:, None] * RE)
    nan_mask = np.isnan(v * zeta * mse)
    clean_array = np.where(nan_mask, 0.0, v * zeta * mse)
    zeta_dvdy = np.gradient(clean_array, axis=-2) / (dlat[:, None] * RE)
    zeta_dvdy[nan_mask] = np.nan
    mse_div_F = mse_dudx + mse_dvdy
    zeta_dudx = np.gradient(u * mse * zeta, axis=-1) / (dlon[None, :] * RE * coslat[:, None])
    zeta_div_F = zeta_dudx + zeta_dvdy
    mse_convergence = np.where(mse_div_F < 0, mse_div_F, np.nan)
    mse_divergence = np.where(mse_div_F > 0, mse_div_F, np.nan)
    zeta_convergence = np.where(zeta_div_F < 0, zeta_div_F, np.nan)
    return mse_convergence, mse_divergence, zeta_convergence

def compute_t2m_energy(u10, v10, lats, lons, t2m, cp=CP, density_sfc=1.2):
    cp_t2m = cp * t2m * density_sfc
    dlat = np.deg2rad(np.gradient(lats))
    dlon = np.deg2rad(np.gradient(lons))
    coslat = np.clip(np.cos(np.deg2rad(lats)), 1e-3, 1.0)
    dudx = np.gradient(u10 * cp_t2m, axis=-1) / (dlon[None, :] * RE * coslat[:, None])
    dvdy = np.gradient(v10 * cp_t2m, axis=-2) / (dlat[:, None] * RE)
    return dudx + dvdy

def compute_scalar_rmse(f, t):
    return np.sqrt(np.nanmean((f.ravel() - t.ravel()) ** 2))

def infer_sfc_var_style(config):
    if config.sfc_var_style is not None:
        return config.sfc_var_style
    fmt = "grib" if Path(config.filename_template).suffix == ".grib" else "netcdf"
    return "cfgrib" if fmt == "grib" else "raw"

def open_pl(fpath, config):
    if Path(fpath).suffix == ".grib":
        return xr.open_dataset(fpath, engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}, "indexpath": ""})
    return xr.open_dataset(fpath)

def process_dyn_pl_file(truth_file, config):
    truth_time = parse_truth_datetime(truth_file.name)
    pl_truth = resolve_pl_truth(config)
    result = {v: {lt: [] for lt in LEADS} for v in ["mse_convergence", "mse_divergence", "vorticity_convergence"]}
    try:
        ds_truth = clip_domain(open_truth_file(truth_file))
    except Exception:
        return result

    for lead in LEADS:
        init_time = truth_time - pd.Timedelta(hours=lead)
        fc_path = Path(config.forecast_dir) / config.filename_template.format(
            date=init_time.strftime("%Y-%m-%d"), hour=init_time.strftime("%H"), lead=lead)
        if not fc_path.exists(): continue
        try:
            ds_fc = clip_domain(open_pl(fc_path, config))
        except Exception:
            continue
        for level in PRESSURE_LEVELS_MSE:
            try:
                tu = get_truth_var(ds_truth, "u", pl_truth, level=level)
                tv = get_truth_var(ds_truth, "v", pl_truth, level=level)
                tt = get_truth_var(ds_truth, "t", pl_truth, level=level)
                tz = get_truth_var(ds_truth, "z", pl_truth, level=level)
                tq = get_truth_var(ds_truth, "q", pl_truth, level=level)
                fu = ds_fc["u"].sel({config.pressure_dim: level}).values
                fv = ds_fc["v"].sel({config.pressure_dim: level}).values
                ft = ds_fc["t"].sel({config.pressure_dim: level}).values
                fz = ds_fc["z"].sel({config.pressure_dim: level}).values
                fq = get_forecast_pl_var(ds_fc, "q", level, config)
                mse_t = compute_mse(tt, tz, tq)
                mse_f = compute_mse(ft, fz, fq)
                conv_t, div_t, zeta_t = compute_mse_vorticity_convergence(tu, tv, ds_truth.latitude.values, ds_truth.longitude.values, mse_t)
                conv_f, div_f, zeta_f = compute_mse_vorticity_convergence(fu, fv, ds_fc.latitude.values, ds_fc.longitude.values, mse_f)
                if level == 850:
                    result["mse_convergence"][lead].append(compute_scalar_rmse(conv_f, conv_t))
                    result["vorticity_convergence"][lead].append(compute_scalar_rmse(zeta_f, zeta_t))
                if level == 200:
                    result["mse_divergence"][lead].append(compute_scalar_rmse(div_f, div_t))
            except Exception:
                continue
    return result

def process_dyn_sfc_file(sfc_file, config, sfc_truth, sfc_style):
    result = {lt: [] for lt in LEADS}
    try:
        valid_time = parse_truth_datetime(sfc_file.name)
    except ValueError:
        return result
    date_str, hour_str = valid_time.strftime("%Y-%m-%d"), valid_time.strftime("%H")

    t2m_file = find_t2m_file(sfc_truth, date_str, hour_str)
    if t2m_file is None:
        return result
    try:
        truth_sfc = clip_domain(open_truth_file(sfc_file, filter_pl=False))
        truth_t2m = clip_domain(open_truth_file(t2m_file, filter_pl=False))
        u10_t = get_truth_var(truth_sfc, "u10", sfc_truth)
        v10_t = get_truth_var(truth_sfc, "v10", sfc_truth)
        t2m_t = get_truth_var(truth_t2m, "t2m", sfc_truth)
    except Exception:
        return result
    lats, lons = truth_sfc.latitude.values, truth_sfc.longitude.values
    energy_t = compute_t2m_energy(u10_t, v10_t, lats, lons, t2m_t)

    for lead in LEADS:
        fc_valid = valid_time - pd.Timedelta(hours=lead)
        fc_path = Path(config.forecast_dir) / config.filename_template.format(
            date=fc_valid.strftime("%Y-%m-%d"), hour=fc_valid.strftime("%H"), lead=lead)
        if not fc_path.exists(): continue
        try:
            if sfc_style == "cfgrib":
                fc_sfc = clip_domain(xr.open_dataset(fc_path, engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"shortName": ["10u", "10v"], "typeOfLevel": "heightAboveGround", "level": 10}, "indexpath": ""}))
                fc_t2m = clip_domain(xr.open_dataset(fc_path, engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}, "indexpath": ""}))
                u10_f, v10_f, t2m_f = fc_sfc["u10"].values, fc_sfc["v10"].values, fc_t2m["t2m"].values
                fc_lats, fc_lons = fc_sfc.latitude.values, fc_sfc.longitude.values
            else:
                fc = clip_domain(xr.open_dataset(fc_path))
                u10_f, v10_f, t2m_f = fc["10u"].values, fc["10v"].values, fc["2t"].values
                fc_lats, fc_lons = fc.latitude.values, fc.longitude.values
            if not (np.array_equal(lats, fc_lats) and np.array_equal(lons, fc_lons)):
                continue
            energy_f = compute_t2m_energy(u10_f, v10_f, fc_lats, fc_lons, t2m_f)
            result[lead].append(compute_scalar_rmse(energy_f, energy_t))
        except Exception:
            continue
    return result

def compute_dynamic_reference(pl_truth, sfc_truth):
    truth_files = glob_truth_files_by_template(pl_truth.dir, pl_truth.filename_template)
    print(f"Computing Dynamic reference values over {len(truth_files)} pl files...")
    mse_conv_vals, mse_div_vals, vort_vals = [], [], []
    for f in truth_files:
        try:
            ds = clip_domain(open_truth_file(f))
        except Exception:
            continue
        for level in PRESSURE_LEVELS_MSE:
            try:
                u = get_truth_var(ds, "u", pl_truth, level=level)
                v = get_truth_var(ds, "v", pl_truth, level=level)
                t = get_truth_var(ds, "t", pl_truth, level=level)
                z = get_truth_var(ds, "z", pl_truth, level=level)
                q = get_truth_var(ds, "q", pl_truth, level=level)
                mse = compute_mse(t, z, q)
                conv, div, zeta = compute_mse_vorticity_convergence(u, v, ds.latitude.values, ds.longitude.values, mse)
                if level == 850:
                    if not np.all(np.isnan(conv)): mse_conv_vals.append(np.nanmean(np.abs(conv)))
                    if not np.all(np.isnan(zeta)): vort_vals.append(np.nanmean(np.abs(zeta)))
                if level == 200:
                    if not np.all(np.isnan(div)): mse_div_vals.append(np.nanmean(np.abs(div)))
            except Exception:
                continue

    ste_vals = []
    if sfc_truth is not None:
        sfc_files = glob_truth_files_by_template(sfc_truth.dir, sfc_truth.filename_template)
        sfc_files = [f for f in sfc_files if "_t2m_" not in f.name]
        print(f"Computing STE reference over {len(sfc_files)} sfc files...")
        for sf in sfc_files:
            try:
                valid_time = parse_truth_datetime(sf.name)
            except ValueError:
                continue
            date_str, hour_str = valid_time.strftime("%Y-%m-%d"), valid_time.strftime("%H")
            t2m_file = find_t2m_file(sfc_truth, date_str, hour_str)
            if t2m_file is None: continue
            try:
                truth_sfc = clip_domain(open_truth_file(sf, filter_pl=False))
                truth_t2m = clip_domain(open_truth_file(t2m_file, filter_pl=False))
                u10 = get_truth_var(truth_sfc, "u10", sfc_truth)
                v10 = get_truth_var(truth_sfc, "v10", sfc_truth)
                t2m = get_truth_var(truth_t2m, "t2m", sfc_truth)
                energy = compute_t2m_energy(u10, v10, truth_sfc.latitude.values, truth_sfc.longitude.values, t2m)
                ste_vals.append(abs(np.nanmean(energy)))
            except Exception:
                continue

    return {
        "mse_convergence": float(np.mean(mse_conv_vals)) if mse_conv_vals else np.nan,
        "mse_divergence": float(np.mean(mse_div_vals)) if mse_div_vals else np.nan,
        "vorticity_convergence": float(np.mean(vort_vals)) if vort_vals else np.nan,
        "surface_temperature_energy": float(np.mean(ste_vals)) if ste_vals else np.nan,
    }

def get_dynamic_reference_values(pl_truth, sfc_truth):
    if pl_truth is None:
        path = REF_DIR / "era5_dynamic_truth_mean.npz"
        if not path.exists():
            raise FileNotFoundError("Default Dynamic reference values not found -- run setup first.")
        return np.load(path, allow_pickle=True)["truth_mean"].item()

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", pl_truth.dir).strip("_")
    path = REF_DIR / f"era5_dynamic_truth_mean_{safe_name}.npz"
    if path.exists():
        print(f"  Using cached Dynamic reference values -> {path}")
        return np.load(path, allow_pickle=True)["truth_mean"].item()

    ref = compute_dynamic_reference(pl_truth, sfc_truth)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(path, truth_mean=ref)
    print(f"  Saved new Dynamic reference values -> {path}")
    return ref

def get_dynamic_data(model_key, period, add_configs):
    if model_key in PRECOMPUTED_MODELS:
        path = NPZ_DIR / f"era5_vs_{model_key}_combined_{period}.npz"
        if not path.exists():
            print(f"MISSING: {path}")
            return {v: {lt: np.nan for lt in LEADS} for v in DYN_VARS}
        d = np.load(path, allow_pickle=True)
        dyn_raw = d["dynamic"].item()
        ref = get_dynamic_reference_values(None, None)
        return {v: {lt: (float(np.nanmean(dyn_raw.get(v, {}).get(lt, []))) / ref[v]
                          if dyn_raw.get(v, {}).get(lt) else np.nan) for lt in LEADS} for v in DYN_VARS}

    config = add_configs[model_key]
    pl_truth = resolve_pl_truth(config)
    sfc_truth = resolve_sfc_truth(config)
    out_path = NPZ_DIR / f"era5_vs_{model_key}_combined_{period}.npz"
    ref = get_dynamic_reference_values(pl_truth, sfc_truth)

    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        if "dynamic" in d:
            print(f"  Using cached Dynamic data for {config.name} ({period}) -> {out_path}")
            dyn_raw = d["dynamic"].item()
            return {v: {lt: (float(np.nanmean(dyn_raw.get(v, {}).get(lt, []))) / ref[v]
                              if dyn_raw.get(v, {}).get(lt) else np.nan) for lt in LEADS} for v in DYN_VARS}

    mode, season = suffix_to_mode_season(period)
    pl_files = filter_truth_files(glob_truth_files_by_template(pl_truth.dir, pl_truth.filename_template), mode, season)
    print(f"  Computing live Dynamic (pl) for {config.name} ({period}): {len(pl_files)} truth files")
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_dyn_pl_file, tf, config) for tf in pl_files]
        pl_results = [f.result() for f in as_completed(futs)]
    combined_pl = {v: {lt: [] for lt in LEADS} for v in ["mse_convergence", "mse_divergence", "vorticity_convergence"]}
    for r in pl_results:
        for v in combined_pl:
            for lt in LEADS:
                combined_pl[v][lt].extend(r[v][lt])

    sfc_files = glob_truth_files_by_template(sfc_truth.dir, sfc_truth.filename_template)
    sfc_files = filter_truth_files(sfc_files, mode, season) if mode != "full_year" else sfc_files
    sfc_style = infer_sfc_var_style(config)
    print(f"  Computing live Dynamic (sfc) for {config.name} ({period}): {len(sfc_files)} truth files")
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_dyn_sfc_file, sf, config, sfc_truth, sfc_style) for sf in sfc_files]
        sfc_result = {lt: [] for lt in LEADS}
        for f in as_completed(futs):
            r = f.result()
            for lt in LEADS: sfc_result[lt].extend(r[lt])

    raw = dict(combined_pl)
    raw["surface_temperature_energy"] = sfc_result

    existing = {}
    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        existing = {k: d[k].item() for k in d.files}
    existing["dynamic"] = raw
    NPZ_DIR.mkdir(exist_ok=True)
    np.savez(out_path, **existing)
    print(f"  Saved -> {out_path}")

    return {v: {lt: (float(np.nanmean(raw[v][lt])) / ref[v] if raw[v][lt] else np.nan) for lt in LEADS} for v in DYN_VARS}

from scores.spatial import fss_2d_single_field

PRECIP_DOMAIN = {"lat_min": -12, "lat_max": 23, "lon_min": 92, "lon_max": 127}
PRECIP_THRESHOLDS = [5, 10, 20, 50]
KM_PER_DEGREE = 111.2
DEFAULT_PRECIP_WINDOWS_GRIDPTS = [1]
DEFAULT_PRECIP_RESOLUTION_DEG = 0.25

def km_to_gridpoints(km, grid_spacing_deg):
    km_per_gridpt = KM_PER_DEGREE * grid_spacing_deg
    return int(math.floor(km / km_per_gridpt + 0.5))

def get_precip_windows_for_resolution(grid_spacing_deg):
    window_km_values = [w * DEFAULT_PRECIP_RESOLUTION_DEG * KM_PER_DEGREE for w in DEFAULT_PRECIP_WINDOWS_GRIDPTS]
    return [km_to_gridpoints(km, grid_spacing_deg) for km in window_km_values]
PRECIP_LEAD_TIMES = [6, 12, 18, 24, 30, 36, 42, 48]
TWELVE_CASE_DATES_PRECIP = {pd.Timestamp(f"{d.split('_')[0]} {d.split('_')[1]}:00") for d in [
    '2024-01-24_12','2024-02-06_12','2024-03-06_12','2024-04-12_06',
    '2024-05-04_06','2024-06-22_00','2024-07-12_06','2024-08-09_06',
    '2024-09-17_12','2024-10-14_06','2024-11-16_06','2024-12-23_12']}

def find_precip_truth_file(precip_truth, valid_time):
    date_str, hour_str = valid_time.strftime("%Y-%m-%d"), valid_time.strftime("%H")
    return find_truth_file_by_template(precip_truth.dir, precip_truth.filename_template, date_str, hour_str)

def load_precip_truth(fpath, precip_truth):
    ds = xr.open_dataset(str(fpath)) if Path(fpath).suffix != ".grib" else open_truth_file(fpath, filter_pl=False)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    ds = ds.sel({lat_name: slice(PRECIP_DOMAIN["lat_min"], PRECIP_DOMAIN["lat_max"]),
                 lon_name: slice(PRECIP_DOMAIN["lon_min"], PRECIP_DOMAIN["lon_max"])})
    ds = ds.sortby(lat_name, ascending=False)
    arr = get_truth_var(ds, "precip", precip_truth)
    return arr, ds[lat_name].values, ds[lon_name].values

def load_precip_forecast(fpath, config, target_lat=None, target_lon=None):
    if Path(fpath).suffix == ".grib":
        ds = xr.open_dataset(str(fpath), engine="cfgrib", backend_kwargs={"indexpath": ""})
    else:
        ds = xr.open_dataset(str(fpath))
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    ds = ds.sortby(lat_name).sortby(lon_name)
    if target_lat is not None and target_lon is not None:
        ds = ds.interp({lat_name: target_lat, lon_name: target_lon}, method="linear")
    else:
        ds = ds.sel({lat_name: slice(PRECIP_DOMAIN["lat_min"], PRECIP_DOMAIN["lat_max"]),
                     lon_name: slice(PRECIP_DOMAIN["lon_min"], PRECIP_DOMAIN["lon_max"])})
    ds = ds.sortby(lat_name, ascending=False)
    tp = ds[config.precip_var].squeeze().values.astype(np.float32) * config.precip_unit_scale
    return np.clip(tp, 0, None)

def build_filename_regex(template):
    pattern = re.escape(template)
    pattern = pattern.replace(re.escape("{date}"), r"(\d{4}-\d{2}-\d{2})")
    pattern = pattern.replace(re.escape("{hour}"), r"(\d{2})")
    pattern = pattern.replace(re.escape("{lead}"), r"(\d+)")
    return re.compile(pattern)

def save_fss_csv(store, outpath):
    rows = []
    for (lead, thr, win), vals in sorted(store.items()):
        rows.append({"lead_time_hr": lead, "threshold_mm": thr,
                     "window_gridpts": win, "n_samples": len(vals), "fss_avg": np.mean(vals)})
    pd.DataFrame(rows).to_csv(outpath, index=False)
    print(f"  Saved -> {outpath}")

def load_fss_penalty(csv_path):
    df = pd.read_csv(str(csv_path))
    df = df[df["threshold_mm"].isin(PRECIP_THRESHOLDS)]
    avg_by_lead = df.groupby("lead_time_hr")["fss_avg"].mean()
    return {lt: float(1 - avg_by_lead[lt]) for lt in LEADS if lt in avg_by_lead}

def compute_fss_for_config(config, mode, season=None):
    precip_truth = resolve_precip_truth(config)
    precip_dir = resolve_precip_forecast_dir(config)
    precip_template = resolve_precip_forecast_filename_template(config)
    regex = build_filename_regex(precip_template)
    all_files = sorted(Path(precip_dir).glob("*"))
    fss_store = {}
    processed = skipped = 0
    precip_windows = None

    for f in all_files:
        m = regex.match(f.name)
        if not m: continue
        date_str, hour_str, lead_str = m.groups()
        lead = int(lead_str)
        if lead not in PRECIP_LEAD_TIMES: continue
        init_time = pd.Timestamp(f"{date_str} {hour_str}:00")
        valid_time = init_time + pd.Timedelta(hours=lead)

        if mode == "12cases":
            if valid_time not in TWELVE_CASE_DATES_PRECIP: continue
        elif mode == "monsoon":
            if valid_time.month not in MONSOON_PERIODS[season]: continue

        truth_file = find_precip_truth_file(precip_truth, valid_time)
        if truth_file is None:
            skipped += 1; continue
        try:
            obs_arr, truth_lat, truth_lon = load_precip_truth(truth_file, precip_truth)
            if precip_windows is None:
                grid_spacing_deg = abs(float(truth_lat[1]) - float(truth_lat[0]))
                precip_windows = get_precip_windows_for_resolution(grid_spacing_deg)
                print(f"  Truth grid spacing: {grid_spacing_deg:.4f} deg -> precip windows (gridpoints): {precip_windows}")
            fc_arr = load_precip_forecast(f, config)
            if fc_arr.shape != obs_arr.shape:
                fc_arr = load_precip_forecast(f, config, target_lat=truth_lat, target_lon=truth_lon)
            if fc_arr.shape != obs_arr.shape:
                skipped += 1; continue
            for thr in PRECIP_THRESHOLDS:
                for win in precip_windows:
                    fss_val = float(fss_2d_single_field(fcst=fc_arr, obs=obs_arr,
                                                         event_threshold=thr, window_size=(win, win)))
                    fss_store.setdefault((lead, thr, win), []).append(fss_val)
            processed += 1
        except Exception as e:
            print(f"  Error {f.name}: {e}"); skipped += 1

    print(f"  Precip: processed {processed}, skipped {skipped}")
    return fss_store

def get_precip_data(model_key, period, add_configs):
    suffix = PERIOD_CHOICES[period]
    if model_key in PRECOMPUTED_MODELS:
        path = FSS_DIR / f"fss_{model_key}_{suffix}.csv"
        if not path.exists():
            print(f"MISSING: {path}")
            return {lt: np.nan for lt in LEADS}
        return load_fss_penalty(path)

    config = add_configs[model_key]
    path = FSS_DIR / f"fss_{model_key}_{suffix}.csv"
    if path.exists():
        print(f"  Using cached precip for {config.name} ({period}) -> {path}")
        return load_fss_penalty(path)

    mode, season = suffix_to_mode_season(suffix)
    print(f"  Computing live precip for {config.name} ({period})")
    fss_store = compute_fss_for_config(config, mode, season)
    FSS_DIR.mkdir(exist_ok=True)
    save_fss_csv(fss_store, path)
    return load_fss_penalty(path)

# ---- spectra ----
SPECTRA_LEADS = [12, 24, 36, 48]
SPECTRA_PRESSURES = [850, 700, 200]
SPECTRA_REF_DIR = REPO / "spectra_reference"

def compute_hke(u, v):
    u = np.squeeze(u)
    v = np.squeeze(v)
    return 0.5 * (u**2 + v**2)

def hke_spectrum_2d(hke, lat, lon):
    dy_deg, dx_deg = abs(lat[1]-lat[0]), abs(lon[1]-lon[0])
    meters_deg = 2 * np.pi * 6371220.0 / 360.0
    dx, dy = dx_deg * meters_deg, dy_deg * meters_deg
    H = np.fft.fftn(hke)
    H_power = np.fft.fftshift(np.abs(H) ** 2, axes=(0, 1))
    ny, nx = hke.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy))
    return H_power, kx, ky

def radial_average_spectrum_2d(H_power, kx, ky):
    if max(kx) > max(ky): return H_power.mean(axis=1), ky
    return H_power.mean(axis=0), kx

def compute_spectrum_for_file(fpath, level, config):
    ds = clip_domain(open_pl(fpath, config))
    u = ds["u"].sel({config.pressure_dim: level}).values
    v = ds["v"].sel({config.pressure_dim: level}).values
    hke = compute_hke(u, v)
    H_power, kx, ky = hke_spectrum_2d(hke, ds.latitude.values, ds.longitude.values)
    H_rad, _ = radial_average_spectrum_2d(H_power, kx, ky)
    return H_rad

def compute_era5_spectrum_for_file(truth_file, level, pl_truth):
    ds = clip_domain(open_truth_file(truth_file))
    u = get_truth_var(ds, "u", pl_truth, level=level)
    v = get_truth_var(ds, "v", pl_truth, level=level)
    hke = compute_hke(u, v)
    H_power, kx, ky = hke_spectrum_2d(hke, ds.latitude.values, ds.longitude.values)
    H_rad, _ = radial_average_spectrum_2d(H_power, kx, ky)
    return H_rad

def ensure_era5_spectra_cached(pl_truth):
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", pl_truth.dir).strip("_")
    cache_dir = SPECTRA_REF_DIR / f"era5_{safe_name}"

    if cache_dir.exists() and any(cache_dir.iterdir()):
        return cache_dir

    print(f"  Computing ERA5 spectra reference for {pl_truth.dir} (full year, one-time)...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    truth_files = glob_truth_files_by_template(pl_truth.dir, pl_truth.filename_template)
    for tf in truth_files:
        valid_str = parse_truth_datetime(tf.name).strftime("%Y-%m-%d_%H")
        for level in SPECTRA_PRESSURES:
            outname = cache_dir / f"HKE_era5_{valid_str}_{level}.npz"
            if outname.exists(): continue
            try:
                H_rad = compute_era5_spectrum_for_file(tf, level, pl_truth)
                np.savez_compressed(outname, H_rad=H_rad)
            except Exception:
                continue
    return cache_dir

def compute_spectra_reference(pl_truth):
    truth_files = glob_truth_files_by_template(pl_truth.dir, pl_truth.filename_template)
    era5_cache = ensure_era5_spectra_cached(pl_truth)
    print(f"Computing Spectra (HKE) reference values over {len(truth_files)} truth files...")
    sums = {level: None for level in SPECTRA_PRESSURES}
    counts = {level: 0 for level in SPECTRA_PRESSURES}
    for tf in truth_files:
        valid_str = parse_truth_datetime(tf.name).strftime("%Y-%m-%d_%H")
        for level in SPECTRA_PRESSURES:
            era5_path = era5_cache / f"HKE_era5_{valid_str}_{level}.npz"
            if not era5_path.exists(): continue
            H_rad = np.load(era5_path)["H_rad"].astype(np.float64)
            sums[level] = H_rad if sums[level] is None else sums[level] + H_rad
            counts[level] += 1
    return {level: (float(np.mean(sums[level] / counts[level])) if counts[level] else np.nan)
            for level in SPECTRA_PRESSURES}

def get_spectra_reference_values(pl_truth):
    if pl_truth is None:
        path = REF_DIR / "era5_spectra_truth_mean.npz"
        if not path.exists():
            raise FileNotFoundError("Default Spectra reference values not found -- run setup first.")
        return np.load(path, allow_pickle=True)["truth_mean"].item()

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", pl_truth.dir).strip("_")
    path = REF_DIR / f"era5_spectra_truth_mean_{safe_name}.npz"
    if path.exists():
        print(f"  Using cached Spectra reference values -> {path}")
        return np.load(path, allow_pickle=True)["truth_mean"].item()

    ref = compute_spectra_reference(pl_truth)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(path, truth_mean=ref)
    print(f"  Saved new Spectra reference values -> {path}")
    return ref

def _compute_one_spectrum(args):
    fpath, level, era5_path, config = args
    try:
        model_rad = compute_spectrum_for_file(fpath, level, config)
        era5_rad = np.load(era5_path)["H_rad"].astype(np.float64)
        return model_rad, era5_rad
    except Exception:
        return None

def compute_spectra_for_model(config, period):
    """Live compute. Caches RAW (unnormalized) per-lead, per-level RMSE as
    'spectra_by_level_raw' into era5_vs_{config.name}_combined_{period}.npz.
    Normalization (division by log10(ref[level]) per level, then averaged
    across levels) happens at READ time in get_spectra_data /
    get_spectra_data_by_level -- matching the Traditional/Dynamic convention,
    so the reference can be recomputed later without redoing this expensive
    per-file FFT computation."""
    suffix = PERIOD_CHOICES[period]
    pl_truth = resolve_pl_truth(config)
    out_path = NPZ_DIR / f"era5_vs_{config.name}_combined_{suffix}.npz"

    era5_cache = ensure_era5_spectra_cached(pl_truth)
    regex = build_filename_regex(config.filename_template)
    all_files = sorted(Path(config.forecast_dir).glob("*"))
    mode, season = suffix_to_mode_season(suffix)

    jobs = []
    for f in all_files:
        m = regex.match(f.name)
        if not m: continue
        date_str, hour_str, lead_str = m.groups()
        lead = int(lead_str)
        if lead not in SPECTRA_LEADS: continue
        init_time = pd.Timestamp(f"{date_str} {hour_str}:00")
        valid_time = init_time + pd.Timedelta(hours=lead)
        if mode == "12cases":
            if valid_time.strftime("%Y-%m-%d_%H") not in TWELVE_CASE_DATES: continue
        elif mode == "monsoon":
            if valid_time.month not in MONSOON_PERIODS[season]: continue
        for level in SPECTRA_PRESSURES:
            era5_path = era5_cache / f"HKE_era5_{valid_time.strftime('%Y-%m-%d_%H')}_{level}.npz"
            if not era5_path.exists(): continue
            jobs.append(((lead, level), f, level, era5_path))

    print(f"  {config.name} ({period}): {len(jobs)} (file, level) pairs to compute")
    sums, era5_sums, counts = {}, {}, {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_compute_one_spectrum, (fpath, level, era5_path, config)): key
                for key, fpath, level, era5_path in jobs}
        for fut in as_completed(futs):
            key = futs[fut]
            res = fut.result()
            if res is None: continue
            model_rad, era5_rad = res
            if key not in sums:
                sums[key] = np.zeros_like(model_rad); era5_sums[key] = np.zeros_like(era5_rad); counts[key] = 0
            sums[key] += model_rad; era5_sums[key] += era5_rad; counts[key] += 1

    print(f"  Spectra computed for {config.name} ({period})")
    by_level_raw = {}
    for lt in LEADS:
        level_raw = {}
        for level in SPECTRA_PRESSURES:
            key = (lt, level)
            n = counts.get(key, 0)
            if n == 0: continue
            model_mean = sums[key] / n
            era5_mean = era5_sums[key] / n
            level_raw[level] = float(np.sqrt(np.mean((era5_mean - model_mean) ** 2)))
        by_level_raw[lt] = level_raw

    existing = {}
    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        existing = {k: d[k].item() for k in d.files}
    existing["spectra_by_level_raw"] = by_level_raw
    NPZ_DIR.mkdir(exist_ok=True)
    np.savez(out_path, **existing)
    print(f"  Saved -> {out_path}")
    return by_level_raw


def _normalize_spectra_by_level(by_level_raw, ref):
    return {lt: {level: raw / ref[level] for level, raw in levels.items()}
            for lt, levels in by_level_raw.items()}


def get_spectra_data_by_level(model_key, period, add_configs=None):
    suffix = PERIOD_CHOICES[period]
    path = NPZ_DIR / f"era5_vs_{model_key}_combined_{suffix}.npz"
    if not path.exists():
        print(f"MISSING: {path}")
        return {lt: {lvl: np.nan for lvl in SPECTRA_PRESSURES} for lt in LEADS}
    d = np.load(path, allow_pickle=True)
    if "spectra_by_level_raw" not in d:
        raise KeyError(f"{path} has no 'spectra_by_level_raw' yet -- regenerate this model's spectra first.")
    raw = {int(lt): {int(lvl): float(v) for lvl, v in levels.items()}
           for lt, levels in d["spectra_by_level_raw"].item().items()}
    if model_key in PRECOMPUTED_MODELS:
        ref = get_spectra_reference_values(None)
    else:
        config = add_configs[model_key]
        ref = get_spectra_reference_values(resolve_pl_truth(config))
    return _normalize_spectra_by_level(raw, ref)


def get_spectra_data(model_key, period, add_configs):
    suffix = PERIOD_CHOICES[period]
    if model_key in PRECOMPUTED_MODELS:
        path = NPZ_DIR / f"era5_vs_{model_key}_combined_{suffix}.npz"
        if not path.exists():
            print(f"MISSING: {path}")
            return {lt: np.nan for lt in LEADS}
        d = np.load(path, allow_pickle=True)
        if "spectra_by_level_raw" not in d:
            print(f"MISSING spectra_by_level_raw in: {path}")
            return {lt: np.nan for lt in LEADS}
        raw = {int(lt): {int(lvl): float(v) for lvl, v in levels.items()}
               for lt, levels in d["spectra_by_level_raw"].item().items()}
        ref = get_spectra_reference_values(None)
        by_level = _normalize_spectra_by_level(raw, ref)
        return {lt: (float(np.mean(list(levels.values()))) if levels else np.nan) for lt, levels in by_level.items()}

    config = add_configs[model_key]
    pl_truth = resolve_pl_truth(config)
    out_path = NPZ_DIR / f"era5_vs_{model_key}_combined_{suffix}.npz"
    ref = get_spectra_reference_values(pl_truth)

    if out_path.exists():
        d = np.load(out_path, allow_pickle=True)
        if "spectra_by_level_raw" in d:
            print(f"  Using cached spectra for {config.name} ({period}) -> {out_path}")
            raw = {int(lt): {int(lvl): float(v) for lvl, v in levels.items()}
                   for lt, levels in d["spectra_by_level_raw"].item().items()}
            by_level = _normalize_spectra_by_level(raw, ref)
            return {lt: (float(np.mean(list(levels.values()))) if levels else np.nan) for lt, levels in by_level.items()}

    by_level_raw = compute_spectra_for_model(config, period)
    by_level = _normalize_spectra_by_level(by_level_raw, ref)
    return {lt: (float(np.mean(list(levels.values()))) if levels else np.nan) for lt, levels in by_level.items()}

def plot_dynamic(model_keys, add_configs, period, outname):
    suffix = PERIOD_CHOICES[period]
    DERIVED_MODELS = {"pangu", "fourcastnet", "aurora"}
    COLUMN_ORDER = DYN_VARS + ["spectra", "precip"]
    matrices = {}
    for m in model_keys:
        dyn = get_dynamic_data(m, suffix, add_configs)
        spectra = get_spectra_data(m, period, add_configs)
        precip = get_precip_data(m, period, add_configs)
        rows = [[dyn[v][lt] for lt in LEADS] for v in DYN_VARS]
        rows.append([spectra[lt] for lt in LEADS])
        rows.append([precip.get(lt, np.nan) for lt in LEADS])
        matrices[m] = np.array(rows)

    all_matrices = [matrices[m] for m in model_keys]
    n_rows, n_vars = len(model_keys), len(COLUMN_ORDER)
    spectra_row = len(DYN_VARS)
    precip_row = len(DYN_VARS) + 1

    all_non_precip = [x for m in model_keys for row_idx in range(spectra_row) for x in matrices[m][row_idx] if not np.isnan(x)]
    global_scale = max(all_non_precip) if all_non_precip else 1.0
    for m in model_keys:
        matrices[m][spectra_row] = [x * global_scale if not np.isnan(x) else x for x in matrices[m][spectra_row]]
        matrices[m][precip_row] = [x * global_scale if not np.isnan(x) else x for x in matrices[m][precip_row]]

    best_mask = [np.zeros_like(m, dtype=bool) for m in all_matrices]
    second_mask = [np.zeros_like(m, dtype=bool) for m in all_matrices]
    for var_idx in range(n_vars):
        for lead_idx in range(len(LEADS)):
            col = np.array([mat[var_idx, lead_idx] for mat in all_matrices])
            if np.all(np.isnan(col)): continue
            col_r = np.where(np.isnan(col), np.nan, np.round(col, 2))
            sorted_idx = np.argsort(np.nan_to_num(col_r, nan=np.inf))
            best_val = col_r[sorted_idx[0]]
            tied = [i for i in range(n_rows) if not np.isnan(col_r[i]) and col_r[i] == best_val]
            for idx in tied: best_mask[idx][var_idx, lead_idx] = True
            if len(tied) == 1:
                second_val = None
                for idx in sorted_idx:
                    if not np.isnan(col_r[idx]) and col_r[idx] != best_val:
                        second_val = col_r[idx]; break
                if second_val is not None:
                    for idx in range(n_rows):
                        if not np.isnan(col_r[idx]) and col_r[idx] == second_val:
                            second_mask[idx][var_idx, lead_idx] = True

    combined = np.vstack(all_matrices)
    combined = np.where(combined <= 0, 1e-3, combined)
    vmin, vmax = np.nanmin(combined), np.nanmax(combined)
    cmap = plt.get_cmap("Reds")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    gap = 0.1
    fig, axes = plt.subplots(n_rows, n_vars, figsize=(n_vars * 3.5, n_rows * 0.8),
                              sharex=True, sharey=True, gridspec_kw={"wspace": 0.15, "hspace": 0.1})
    if n_rows == 1: axes = np.array([axes])
    if n_vars == 1: axes = axes.reshape(n_rows, 1)

    for i, m in enumerate(model_keys):
        matrix = all_matrices[i]
        for j, ax in enumerate(axes[i]):
            for spine in ax.spines.values(): spine.set_visible(False)
            for k, lt in enumerate(LEADS):
                v = matrix[j, k]
                xl, xr = k + gap/2, k + 1 - gap/2
                if np.isnan(v):
                    ax.fill_between([xl, xr], 0, 1, color=(0.95,0.95,0.95,1.0))
                    ax.text((xl+xr)/2, 0.5, "NaN", ha="center", va="center", fontsize=8, color="black")
                    continue
                ax.fill_between([xl, xr], 0, 1, color=cmap(norm(v)))
                label_text = f"{v:.2f}"
                if j == precip_row and m in DERIVED_MODELS:
                    label_text += "\u2020"
                ax.text((xl+xr)/2, 0.5, label_text, ha="center", va="center", fontsize=7.5, color="black")
                if best_mask[i][j, k]:
                    ax.add_patch(Rectangle((xl+0.04, 0.04), 0.89, 0.89, fill=False, edgecolor="green", linewidth=1.5))
                if second_mask[i][j, k]:
                    ax.add_patch(Rectangle((xl+0.08, 0.08), 0.83, 0.83, fill=False, edgecolor="blue", linewidth=1.5))
            if i == 0:
                title = "HKE Spectrum NRMSE\n(scaled, 200/700/850 avg)" if j == spectra_row else \
                         "6-hourly Acc. Precip.\nFSS Score (scaled)" if j == precip_row else \
                         DYN_DISPLAY[DYN_VARS[j]] + " NRMSE"
                ax.set_title(title, fontsize=9, pad=10)
            if i == n_rows - 1:
                ax.set_xticks(np.arange(len(LEADS)) + 0.5)
                ax.set_xticklabels(LEADS, fontsize=8)
                ax.set_xlabel("Lead (h)", fontsize=8)
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_yticks([]); ax.set_xlim(0, len(LEADS)); ax.set_ylim(0, 1)
        pl_sfc_label, precip_label = get_truth_labels(m, add_configs)
        row_label = f"{MODEL_DISPLAY.get(m, m)}\n({pl_sfc_label} / {precip_label})"
        axes[i, 0].set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=10, labelpad=20)

    legend_handles = [Patch(edgecolor="green", facecolor="none", linewidth=1.5, label="Best"),
                       Patch(edgecolor="blue", facecolor="none", linewidth=1.5, label="Second Best")]
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.2, extend="max")
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(2.0, 0), bbox_transform=cbar.ax.transAxes, fontsize=9, frameon=False)
    cbar.set_label("Normalised Score", fontsize=9)
    cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{(vmin+vmax)/2:.2f}", f"{vmax:.2f}"])

    out_path = FIG_DIR / outname
    var_labels = [DYN_DISPLAY[v].replace("\n", " ") for v in DYN_VARS] + \
                 ["HKE Spectrum NRMSE (scaled)", "Precipitation FSS (scaled)"]
    csv_rows = []
    for m in model_keys:
        for idx, label in enumerate(var_labels):
            row = {"model": MODEL_DISPLAY.get(m, m), "variable": label}
            row.update({f"lead_{lt}h": matrices[m][idx, k] for k, lt in enumerate(LEADS)})
            csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(out_path.with_suffix(".csv"), index=False)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

# ============================================================
# COMBINED MODE
# ============================================================
def compute_combined_scores(model_keys, period, add_configs):
    """Traditional's 4 + Dynamic's 4, unweighted-averaged, with spectra AND
    precip each scaled by global_scale (max of the base pool: Traditional's 4
    + Dynamic's 4, excluding spectra and precip) before being folded in --
    matches the manuscript's approved no-log + spectra-scaled convention."""
    suffix = PERIOD_CHOICES[period]
    all_metrics = {}
    for m in model_keys:
        trad = get_traditional_data(m, suffix, add_configs)
        dyn = get_dynamic_data(m, suffix, add_configs)
        spectra = get_spectra_data(m, period, add_configs)
        precip = get_precip_data(m, period, add_configs)
        metrics = {}
        for var, _ in TRAD_VARS:
            metrics[f"trad_{var}"] = trad[var]
        for var in DYN_VARS:
            metrics[f"dyn_{var}"] = dyn[var]
        all_metrics[m] = {"base": metrics, "spectra": spectra, "precip": precip}

    all_base = []
    for m in model_keys:
        for metric, lead_dict in all_metrics[m]["base"].items():
            for lt in LEADS:
                v = lead_dict.get(lt, np.nan)
                if not (isinstance(v, float) and np.isnan(v)):
                    all_base.append(float(v))
    global_scale = max(all_base) if all_base else 1.0
    print(f"  [{period}] Global scale (base pool, spectra+precip excluded): {global_scale:.4f}")

    scores = {}
    for m in model_keys:
        scores[m] = {}
        for lt in LEADS:
            vals = []
            for metric, lead_dict in all_metrics[m]["base"].items():
                v = lead_dict.get(lt, np.nan)
                if not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            spectra_v = all_metrics[m]["spectra"].get(lt, np.nan)
            if not (isinstance(spectra_v, float) and np.isnan(spectra_v)):
                vals.append(spectra_v * global_scale)
            precip_v = all_metrics[m]["precip"].get(lt, np.nan)
            if not (isinstance(precip_v, float) and np.isnan(precip_v)):
                vals.append(precip_v * global_scale)
            scores[m][lt] = float(np.mean(vals)) if vals else np.nan
    return scores

def plot_combined(model_keys, add_configs, period, outname):
    scores = compute_combined_scores(model_keys, period, add_configs)
    n_rows = len(model_keys)
    matrix = np.array([[scores[m][lt] for lt in LEADS] for m in model_keys])

    best_mask = np.zeros_like(matrix, dtype=bool)
    second_mask = np.zeros_like(matrix, dtype=bool)
    for lead_idx in range(len(LEADS)):
        col = matrix[:, lead_idx]
        if np.all(np.isnan(col)): continue
        col_r = np.where(np.isnan(col), np.nan, np.round(col, 2))
        sorted_idx = np.argsort(np.nan_to_num(col_r, nan=np.inf))
        best_val = col_r[sorted_idx[0]]
        tied = [i for i in range(n_rows) if not np.isnan(col_r[i]) and col_r[i] == best_val]
        for idx in tied: best_mask[idx, lead_idx] = True
        if len(tied) == 1:
            second_val = None
            for idx in sorted_idx:
                if not np.isnan(col_r[idx]) and col_r[idx] != best_val:
                    second_val = col_r[idx]; break
            if second_val is not None:
                for idx in range(n_rows):
                    if not np.isnan(col_r[idx]) and col_r[idx] == second_val:
                        second_mask[idx, lead_idx] = True

    valid = matrix[~np.isnan(matrix)]
    vmin, vmax = (float(np.min(valid)), float(np.max(valid))) if len(valid) else (0.0, 1.0)
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    gap = 0.1
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, n_rows * 0.9), sharex=True, sharey=True)
    if n_rows == 1: axes = np.array([axes])

    for i, m in enumerate(model_keys):
        ax = axes[i]
        for spine in ax.spines.values(): spine.set_visible(False)
        for k, lt in enumerate(LEADS):
            v = matrix[i, k]
            xl, xr = k + gap/2, k + 1 - gap/2
            if np.isnan(v):
                ax.fill_between([xl, xr], 0, 1, color=(0.95,0.95,0.95,1.0))
                ax.text((xl+xr)/2, 0.5, "NaN", ha="center", va="center", fontsize=8, color="black")
                continue
            ax.fill_between([xl, xr], 0, 1, color=cmap(norm(v)))
            ax.text((xl+xr)/2, 0.5, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")
            if best_mask[i, k]:
                ax.add_patch(Rectangle((xl+0.04, 0.04), 0.89, 0.89, fill=False, edgecolor="red", linewidth=1.5))
            if second_mask[i, k]:
                ax.add_patch(Rectangle((xl+0.08, 0.08), 0.83, 0.83, fill=False, edgecolor="orange", linewidth=1.5))
        if i == 0:
            ax.set_title("Combined Score", fontsize=9, pad=10)
        if i == n_rows - 1:
            ax.set_xticks(np.arange(len(LEADS)) + 0.5)
            ax.set_xticklabels(LEADS, fontsize=8)
            ax.set_xlabel("Lead (h)", fontsize=8)
        else:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_yticks([]); ax.set_xlim(0, len(LEADS)); ax.set_ylim(0, 1)
        pl_sfc_label, precip_label = get_truth_labels(m, add_configs)
        row_label = f"{MODEL_DISPLAY.get(m, m)}\n({pl_sfc_label} / {precip_label})"
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=10, labelpad=20)

    legend_handles = [Patch(edgecolor="red", facecolor="none", linewidth=1.5, label="Best"),
                       Patch(edgecolor="orange", facecolor="none", linewidth=1.5, label="Second Best")]
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.15)
    cbar.set_label("Combined Score", fontsize=9)
    cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", f"{(vmin+vmax)/2:.2f}", f"{vmax:.2f}"])
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), bbox_transform=fig.transFigure, fontsize=9, frameon=False)

    out_path = FIG_DIR / outname
    csv_rows = []
    for i, m in enumerate(model_keys):
        row = {"model": MODEL_DISPLAY.get(m, m)}
        row.update({f"lead_{lt}h": matrix[i, k] for k, lt in enumerate(LEADS)})
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(out_path.with_suffix(".csv"), index=False)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WeatherBench: Traditional / Dynamic / Combined")
    parser.add_argument("--mode", required=True, choices=["traditional", "dynamic", "combined"])
    parser.add_argument("--period", required=True, choices=list(PERIOD_CHOICES.keys()))
    parser.add_argument("--add_model_config", action="append", default=[],
                         help="Path to a JSON config for a model to add (repeatable)")
    args = parser.parse_args()

    model_keys = list(PRECOMPUTED_MODELS)
    add_configs = {}
    for path in args.add_model_config:
        cfg = load_model_config(path)
        model_keys.append(cfg.name)
        add_configs[cfg.name] = cfg

    if args.mode == "traditional":
        plot_traditional(model_keys, add_configs, args.period, f"traditional_{args.period}.png")
    elif args.mode == "dynamic":
        plot_dynamic(model_keys, add_configs, args.period, f"dynamic_{args.period}.png")
    elif args.mode == "combined":
        plot_combined(model_keys, add_configs, args.period, f"combined_{args.period}.png")
