"""Check feature and score drift for Best_BB20_Reversion_Buy_SimOnly.

Reference window defaults to the training years in the model meta file
(2019-2023). The current window defaults to the latest XAUUSDm year CSV
that is not in that training set.

The compared population matches the rows the classifier was fit on, as far
as features allow: finite feature rows whose BB20 reversion direction is buy.
The TP/SL label filter is left out because it looks into the future.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT
    / "models"
    / "Best_BB20_Reversion_Buy_SimOnly"
    / "Best_BB20_Reversion_Buy_SimOnly_model.joblib"
)
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "1M"
REPORT_DIR = ROOT / "monitoring" / "reports"
REFERENCE_DIR = ROOT / "monitoring" / "reference"
VENDOR_SCJ = ROOT / "vendor" / "scj"

DUPLICATE_FEATURES = ("bb20_pos", "bb20_width_r", "bb20_z")
PSI_BINS = 10
PSI_WARNING = 0.10
PSI_ALERT = 0.25
KS_WARNING = 0.10
KS_ALERT = 0.20
EPS = 1e-6


def _import_cycle_builder():
    vendor = str(VENDOR_SCJ)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from method.CycleFeatureBuilder import create_cycle_features

    return create_cycle_features


def parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(piece.strip()) for piece in part.split("-", 1)]
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    return sorted(dict.fromkeys(years))


def _status(psi: float, ks: float) -> str:
    if psi >= PSI_ALERT or ks >= KS_ALERT:
        return "alert"
    if psi >= PSI_WARNING or ks >= KS_WARNING:
        return "warning"
    return "ok"


def _quantile_edges(values: np.ndarray, n_bins: int = PSI_BINS) -> np.ndarray | None:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return None
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = _quantile_edges(reference)
    if edges is None:
        keys = np.unique(np.concatenate([reference, current]))
        ref_counts = np.array([(reference == key).sum() for key in keys], dtype=float)
        cur_counts = np.array([(current == key).sum() for key in keys], dtype=float)
    else:
        ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
        cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    ref_share = ref_counts / ref_counts.sum()
    cur_share = cur_counts / max(cur_counts.sum(), 1.0)
    ref_share = np.clip(ref_share, EPS, None)
    cur_share = np.clip(cur_share, EPS, None)
    ref_share = ref_share / ref_share.sum()
    cur_share = cur_share / cur_share.sum()
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    grid = np.sort(np.unique(np.concatenate([reference, current])))
    ref_cdf = np.searchsorted(reference, grid, side="right") / len(reference)
    cur_cdf = np.searchsorted(current, grid, side="right") / len(current)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def load_meta(model_path: Path) -> dict:
    meta_path = model_path.with_name(model_path.name.replace("_model.joblib", "_meta.json"))
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing model meta: {meta_path}")
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    features = list(meta["features"])
    if len(features) != len(set(features)):
        raise ValueError("Meta feature list contains duplicate names; expected the logical list.")
    meta["features"] = features
    meta["meta_path"] = str(meta_path)
    return meta


def _year_csv(data_dir: Path, year: int) -> Path:
    path = data_dir / f"XAUUSDm_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing OHLC file: {path}")
    return path


def available_years(data_dir: Path) -> list[int]:
    years = []
    for path in data_dir.glob("XAUUSDm_*.csv"):
        suffix = path.stem.replace("XAUUSDm_", "")
        if suffix.isdigit():
            years.append(int(suffix))
    return sorted(years)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_bb20_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the BB20 reversion columns the saved model was trained on."""
    create_cycle_features = _import_cycle_builder()
    frame = raw.reset_index(drop=True)
    dates = pd.to_datetime(frame["dates"], errors="coerce")
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)

    body_abs = (close - open_).abs()
    body_range = body_abs.rolling(200, min_periods=50).max() - body_abs.rolling(200, min_periods=50).min()
    fallback_scale = (high - low).rolling(200, min_periods=50).mean()
    norm_scale = body_range.replace(0, np.nan).fillna(fallback_scale).replace(0, np.nan)

    out = pd.DataFrame(index=frame.index)
    out["dates"] = dates
    out["hour"] = dates.dt.hour
    out["open"] = open_
    out["high"] = high
    out["low"] = low
    out["close"] = close
    out["norm_scale"] = norm_scale
    out["hour_sin"] = np.sin(2 * np.pi * dates.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dates.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    out["range_r"] = _safe_div(high - low, norm_scale)
    out["body_r"] = _safe_div(close - open_, norm_scale)
    out["body_abs_r"] = out["body_r"].abs()
    candle_range = (high - low).replace(0, np.nan)
    out["upper_wick_ratio"] = (high - np.maximum(open_, close)) / candle_range
    out["lower_wick_ratio"] = (np.minimum(open_, close) - low) / candle_range
    out["wick_balance"] = out["lower_wick_ratio"] - out["upper_wick_ratio"]

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = true_range.rolling(14).mean()
    out["atr_rank"] = out["ATR14"].rolling(2000, min_periods=200).rank(pct=True)

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = upper - lower
    out["bb20_pos__base"] = _safe_div(close - lower, width)
    out["bb20_width_r__base"] = _safe_div(width, norm_scale)
    out["bb20_z__base"] = _safe_div(close - mid, std)
    out["bb20_pos"] = (close - lower) / width.replace(0, np.nan)
    out["bb20_width_r"] = width / norm_scale.replace(0, np.nan)
    out["bb20_width_rank"] = width.rolling(2000, min_periods=200).rank(pct=True)
    out["bb20_z"] = (close - mid) / std.replace(0, np.nan)
    out["bb20_dist_mid_r"] = (close - mid) / norm_scale.replace(0, np.nan)
    out["bb20_dist_upper_r"] = (close - upper) / norm_scale.replace(0, np.nan)
    out["bb20_dist_lower_r"] = (close - lower) / norm_scale.replace(0, np.nan)
    out["bb20_squeeze"] = (out["bb20_width_rank"] <= 0.20).astype(float)
    out["bb20_break_upper"] = (close > upper).astype(float)
    out["bb20_break_lower"] = (close < lower).astype(float)
    out["bb20_reversion_raw"] = np.where(out["bb20_pos"] <= 0.20, 1, np.where(out["bb20_pos"] >= 0.80, -1, 0))
    out["bb20_width_signal"] = np.where(out["bb20_width_rank"] >= 0.50, 1, -1)
    out["bb20_mid"] = mid
    out["bb20_upper"] = upper
    out["bb20_lower"] = lower

    pieces = [out]
    pieces.append(
        create_cycle_features(
            out,
            "signal_col",
            prefix="bb20_reversion",
            signal_col="bb20_reversion_raw",
            reference_col="bb20_mid",
            norm_col="norm_scale",
        )
    )
    pieces.append(
        create_cycle_features(
            out,
            "price_line_cross",
            prefix="bb20_lower",
            price_col="close",
            line_col="bb20_lower",
            reference_col="bb20_lower",
            norm_col="norm_scale",
        )
    )
    pieces.append(
        create_cycle_features(
            out,
            "price_line_cross",
            prefix="bb20_upper",
            price_col="close",
            line_col="bb20_upper",
            reference_col="bb20_upper",
            norm_col="norm_scale",
        )
    )
    pieces.append(
        create_cycle_features(
            out,
            "signal_col",
            prefix="bb20_width",
            signal_col="bb20_width_signal",
            reference_col="bb20_mid",
            norm_col="norm_scale",
        )
    )
    built = pd.concat(pieces, axis=1)
    built = built.loc[:, ~built.columns.duplicated()].replace([np.inf, -np.inf], np.nan)
    return built


def load_year_features(data_dir: Path, year: int) -> pd.DataFrame:
    raw = pd.read_csv(_year_csv(data_dir, year), usecols=["dates", "open", "high", "low", "close"])
    print(f"Building BB20 features for {year}: {len(raw):,} candles", flush=True)
    frame = build_bb20_frame(raw)
    frame["year"] = year
    return frame


def select_population(frame: pd.DataFrame, features: list[str], population: str) -> pd.DataFrame:
    selected = frame
    if population == "session":
        selected = selected[(selected["hour"] >= 7) & (selected["hour"] <= 22)]
    elif population == "train_like":
        signal = selected["bb20_reversion_signal_dir"]
        selected = selected[signal == 1]
    elif population != "all":
        raise ValueError(f"Unsupported population: {population}")
    valid = np.isfinite(selected[features].to_numpy(dtype=float)).all(axis=1)
    return selected.loc[valid].reset_index(drop=True)


def model_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """81 columns: each duplicated BB20 name contributes the base copy, then the family copy."""
    columns = []
    for name in features:
        if name in DUPLICATE_FEATURES:
            columns.append(frame[f"{name}__base"].to_numpy(dtype=float))
        columns.append(frame[name].to_numpy(dtype=float))
    matrix = np.column_stack(columns)
    expected = len(features) + len(DUPLICATE_FEATURES)
    if matrix.shape[1] != expected:
        raise RuntimeError(f"Expected {expected} model columns, got {matrix.shape[1]}")
    return matrix


def sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indexes = np.sort(rng.choice(len(frame), size=max_rows, replace=False))
    return frame.iloc[indexes].reset_index(drop=True)


def _reference_cache_path() -> Path:
    return REFERENCE_DIR / "bb20_reversion_reference.parquet"


def _reference_meta_path() -> Path:
    return REFERENCE_DIR / "bb20_reversion_reference_meta.json"


def _cache_key(args, meta: dict, years: list[int]) -> dict:
    return {
        "model": str(args.model.resolve()),
        "features": meta["features"],
        "years": years,
        "population": args.population,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "buy_threshold": float(meta.get("buy_threshold", 0.52)),
    }


def load_or_build_reference(args, meta: dict, model) -> tuple[pd.DataFrame, dict]:
    years = args.reference_years or [int(year) for year in meta["train_years"]]
    key = _cache_key(args, meta, years)
    cache = _reference_cache_path()
    cache_meta = _reference_meta_path()
    if cache.exists() and cache_meta.exists() and not args.refresh_reference:
        saved = json.loads(cache_meta.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            print(f"Loaded reference cache: {cache}", flush=True)
            return pd.read_parquet(cache), saved

    parts = [select_population(load_year_features(args.data_dir, year), meta["features"], args.population) for year in years]
    reference = sample_frame(pd.concat(parts, ignore_index=True), args.max_rows, args.seed)
    reference = score_frame(reference, meta["features"], model, float(meta.get("buy_threshold", 0.52)))
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["year", "buy_score", *meta["features"]]
    reference[keep].to_parquet(cache, index=False)
    payload = {"key": key, "rows": int(len(reference)), "saved_at": datetime.now(timezone.utc).isoformat()}
    cache_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved reference cache: {cache} ({len(reference):,} rows)", flush=True)
    return reference, payload


def score_frame(frame: pd.DataFrame, features: list[str], model, buy_threshold: float) -> pd.DataFrame:
    scored = frame.copy()
    matrix = model_matrix(scored, features)
    if matrix.shape[1] != int(model.n_features_in_):
        raise RuntimeError(
            f"Model expects {model.n_features_in_} inputs, matrix has {matrix.shape[1]}"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        scored["buy_score"] = model.predict_proba(matrix)[:, 1]
    scored["above_threshold"] = scored["buy_score"] >= buy_threshold
    return scored


def compare_feature(reference: np.ndarray, current: np.ndarray) -> dict:
    psi = population_stability_index(reference, current)
    ks = ks_statistic(reference, current)
    return {
        "psi": psi,
        "ks": ks,
        "ref_mean": float(np.nanmean(reference)),
        "cur_mean": float(np.nanmean(current)),
        "ref_std": float(np.nanstd(reference)),
        "cur_std": float(np.nanstd(current)),
        "status": _status(psi, ks),
    }


def build_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], buy_threshold: float) -> dict:
    rows = []
    for name in features:
        row = {"feature": name, **compare_feature(reference[name].to_numpy(float), current[name].to_numpy(float))}
        rows.append(row)
    rows.sort(key=lambda item: (-(item["psi"] if np.isfinite(item["psi"]) else -1), item["feature"]))
    score = compare_feature(reference["buy_score"].to_numpy(float), current["buy_score"].to_numpy(float))
    score["feature"] = "buy_score"
    score["buy_threshold"] = buy_threshold
    score["ref_above_threshold"] = float(reference["above_threshold"].mean())
    score["cur_above_threshold"] = float(current["above_threshold"].mean())
    worst = "ok"
    for item in [*rows, score]:
        if item["status"] == "alert":
            worst = "alert"
            break
        if item["status"] == "warning":
            worst = "warning"
    return {"status": worst, "features": rows, "score": score}


def write_report(report: dict, stem: str) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{stem}.json"
    csv_path = REPORT_DIR / f"{stem}.csv"
    latest_csv = REPORT_DIR / "bb20_data_drift_latest.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = pd.DataFrame(report["features"])
    score_row = {key: report["score"].get(key) for key in table.columns}
    table = pd.concat([pd.DataFrame([score_row]), table], ignore_index=True)
    table.to_csv(csv_path, index=False)
    table.to_csv(latest_csv, index=False)
    return json_path, csv_path


def print_report(report: dict) -> None:
    print("")
    print(f"Drift status: {report['status']}")
    print(
        "Reference {ref_years} ({ref_rows:,} rows) vs current {cur_years} ({cur_rows:,} rows), population={population}".format(
            **report
        )
    )
    score = report["score"]
    print(
        "buy_score PSI={psi:.4f} KS={ks:.4f} status={status} "
        "P(score>={buy_threshold:.2f}) ref={ref_above_threshold:.3f} current={cur_above_threshold:.3f}".format(**score)
    )
    drifted = [row for row in report["features"] if row["status"] != "ok"]
    print(f"Features: {len(report['features'])} checked, {len(drifted)} warning/alert")
    preview = drifted[:15] if drifted else report["features"][:8]
    print(f"{'feature':<42} {'psi':>8} {'ks':>8} {'ref_mean':>12} {'cur_mean':>12} status")
    for row in preview:
        print(
            f"{row['feature']:<42} {row['psi']:8.4f} {row['ks']:8.4f} "
            f"{row['ref_mean']:12.4f} {row['cur_mean']:12.4f} {row['status']}"
        )


def self_test() -> None:
    rng = np.random.default_rng(0)
    same = rng.normal(size=5000)
    identical = population_stability_index(same, same.copy())
    shifted = population_stability_index(same, same + 2.0)
    ks_same = ks_statistic(same, same.copy())
    ks_shift = ks_statistic(same, same + 2.0)
    if not identical < 0.02:
        raise AssertionError(f"identical PSI too high: {identical}")
    if not shifted > PSI_ALERT:
        raise AssertionError(f"shifted PSI too low: {shifted}")
    if not ks_same < 0.01:
        raise AssertionError(f"identical KS too high: {ks_same}")
    if not ks_shift > KS_ALERT:
        raise AssertionError(f"shifted KS too low: {ks_shift}")
    raw = pd.DataFrame(
        {
            "dates": pd.date_range("2024-01-02 07:00:00", periods=2500, freq="min"),
            "open": np.linspace(2300, 2320, 2500),
            "high": np.linspace(2300.4, 2320.4, 2500),
            "low": np.linspace(2299.6, 2319.6, 2500),
            "close": np.linspace(2300.2, 2320.2, 2500) + np.sin(np.linspace(0, 40, 2500)),
        }
    )
    built = build_bb20_frame(raw)
    required = ["bb20_reversion_signal_dir", "bb20_lower_bar", "bb20_width_touch_ref_count_clip", "atr_rank"]
    missing = [name for name in required if name not in built.columns]
    if missing:
        raise AssertionError(f"feature builder missed {missing}")
    print(
        "self-test ok "
        f"psi_same={identical:.4f} psi_shift={shifted:.4f} "
        f"ks_same={ks_same:.4f} ks_shift={ks_shift:.4f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BB20 reversion buy-model data drift.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference-years", type=parse_years, default=None, help="Example: 2019-2023")
    parser.add_argument("--current-years", type=parse_years, default=None, help="Example: 2026 or 2025-2026")
    parser.add_argument("--population", choices=["train_like", "session", "all"], default="train_like")
    parser.add_argument("--max-rows", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    meta = load_meta(args.model)
    print(f"Loading model: {args.model}", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        model = joblib.load(args.model)

    years_on_disk = available_years(args.data_dir)
    reference_years = args.reference_years or [int(year) for year in meta["train_years"]]
    if args.current_years:
        current_years = args.current_years
    else:
        held_out = [year for year in years_on_disk if year not in reference_years]
        if not held_out:
            raise RuntimeError(f"No current year CSV outside {reference_years} in {args.data_dir}")
        current_years = [held_out[-1]]

    reference, _cache = load_or_build_reference(args, meta, model)
    current_parts = [
        select_population(load_year_features(args.data_dir, year), meta["features"], args.population)
        for year in current_years
    ]
    current = sample_frame(pd.concat(current_parts, ignore_index=True), args.max_rows, args.seed + 1)
    threshold = float(meta.get("buy_threshold", 0.52))
    current = score_frame(current, meta["features"], model, threshold)
    if "above_threshold" not in reference.columns:
        reference["above_threshold"] = reference["buy_score"] >= threshold

    report = build_report(reference, current, meta["features"], threshold)
    report.update(
        {
            "model": str(args.model.resolve()),
            "strategy_name": meta.get("strategy_name"),
            "population": args.population,
            "ref_years": reference_years,
            "cur_years": current_years,
            "ref_rows": int(len(reference)),
            "cur_rows": int(len(current)),
            "thresholds": {
                "psi_warning": PSI_WARNING,
                "psi_alert": PSI_ALERT,
                "ks_warning": KS_WARNING,
                "ks_alert": KS_ALERT,
            },
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, csv_path = write_report(report, f"bb20_data_drift_{stamp}")
    print_report(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 1 if report["status"] == "alert" else 0


if __name__ == "__main__":
    raise SystemExit(main())
