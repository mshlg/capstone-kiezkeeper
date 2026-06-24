# %% [markdown]
# # Stage 6 — Monthly slopes  (analysis_ready_data monthly panels -> feature matrix)
#
# For every PLR and every share/quotient variable, fit a linear trend over the 36
# monthly observations and report the SLOPE (per year) as a feature for the models.
#
# Method:
#   * One linear regression value ~ time per (PLR, variable) over the monthly panels.
#   * Time axis is in MONTHS (0, 1, 2, ...); the raw monthly slope is then multiplied
#     by 12 to express it PER YEAR (annualised monthly trend, not a year-over-year change).
#   * NaN months are skipped (e.g. upscale_share is NaN where a PLR had no gastronomy
#     that month). The regression uses only the valid points.
#   * A slope is only computed when a PLR has at least MIN_VALID_MONTHS (=12) valid
#     points; otherwise the slope is NaN (too few points for a reliable trend).
#
# Output per PLR (one row): for each of the 19 variables
#   * <var>_slope       — annualised trend (per year), float, NaN if < MIN_VALID_MONTHS
#   * <var>_r2          — R^2 of the linear fit (trend quality): high = clean linear
#                         trend, low = non-linear / noisy. NaN if < MIN_VALID_MONTHS.
#   * <var>_n_months_in_slope     — number of months that actually entered the regression
# plus planungsraum_id.  -> a 542-row feature matrix for K-Means / classification.

# %% Imports & configuration
from pathlib import Path
import numpy as np
import pandas as pd

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()

BASE_DIR = REPO_ROOT / "data/commercial/IHK_Berlin_Gewerbedaten"
ANALYSIS_DIR = BASE_DIR / "analysis_ready_data"

MIN_VALID_MONTHS = 12          # minimum valid monthly points required to fit a slope
MONTHS_PER_YEAR = 12           # scale monthly slope -> per-year slope

# The four monthly infrastructure panels and the share/quotient (+ median_age) variables
# whose trend we want. Counts and raw flows are intentionally excluded.
PANEL_VARS = {
    "demographics.csv": [
        "median_age",
        "share_max_1y", "share_max_2y", "share_max_5y",
        "share_min_20y", "share_min_30y", "share_min_40y",
        "share_solo", "share_1_9", "share_10plus",
    ],
    "gastronomy.csv": [
        "gastro_share", "upscale_share",
        "gastro_quotient_bezirk", "gastro_quotient_berlin",
        "upscale_quotient_bezirk", "upscale_quotient_berlin",
    ],
    "tourism.csv": [
        "tourism_share", "tourism_quotient_bezirk", "tourism_quotient_berlin",
    ],
}


# %% Helpers
def load_long() -> pd.DataFrame:
    """Load the four monthly panels, keep only the wanted variables, return one
    long table keyed on (planungsraum_id, month) with all 19 variables side by side."""
    merged = None
    key = ["planungsraum_id", "month"]
    for fname, variables in PANEL_VARS.items():
        path = ANALYSIS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"missing monthly panel: {path}")
        cols = key + variables
        df = pd.read_csv(path, usecols=cols, dtype={"planungsraum_id": str})
        merged = df if merged is None else merged.merge(df, on=key, how="outer")
    # ordinal month index 0..35 for the time axis
    months = sorted(merged["month"].unique())
    month_idx = {m: i for i, m in enumerate(months)}
    merged["t"] = merged["month"].map(month_idx)
    return merged, months


def slope_and_n(t: np.ndarray, y: np.ndarray) -> tuple:
    """OLS slope of y ~ t over the points where y is not NaN.
    Returns (slope_per_month, r2, n_months_in_slope). slope and r2 are NaN if fewer than
    MIN_VALID_MONTHS valid points or if the time values have no spread.
    r2 is the coefficient of determination of the linear fit (trend quality):
    1.0 = perfectly linear, near 0 = no linear structure. r2 is set to NaN when y
    has no variation (a flat series), since R^2 is undefined for zero total variance."""
    mask = ~np.isnan(y)
    n = int(mask.sum())
    if n < MIN_VALID_MONTHS:
        return np.nan, np.nan, n
    tt, yy = t[mask], y[mask]
    t_mean = tt.mean()
    denom = ((tt - t_mean) ** 2).sum()
    if denom == 0:                       # no variation in time -> slope undefined
        return np.nan, np.nan, n
    slope = ((tt - t_mean) * (yy - yy.mean())).sum() / denom
    intercept = yy.mean() - slope * t_mean
    ss_tot = ((yy - yy.mean()) ** 2).sum()
    if ss_tot == 0:                      # flat series: slope is 0, R^2 undefined
        return slope, np.nan, n
    ss_res = ((yy - (slope * tt + intercept)) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    return slope, r2, n


# %% Build the feature matrix
def build_slopes() -> pd.DataFrame:
    long, months = load_long()
    all_vars = [v for vs in PANEL_VARS.values() for v in vs]

    plrs = sorted(long["planungsraum_id"].unique())
    records = []
    for plr, grp in long.groupby("planungsraum_id"):
        grp = grp.sort_values("t")
        t = grp["t"].to_numpy(dtype=float)
        rec = {"planungsraum_id": plr}
        for var in all_vars:
            y = grp[var].to_numpy(dtype=float)
            slope_m, r2, n_months = slope_and_n(t, y)
            rec[f"{var}_slope"] = slope_m * MONTHS_PER_YEAR if not np.isnan(slope_m) else np.nan
            rec[f"{var}_r2"] = r2
            rec[f"{var}_n_months_in_slope"] = n_months
        records.append(rec)

    feat = pd.DataFrame.from_records(records)
    # column order: id, then per variable (slope, r2, n_months_in_slope) blocks
    ordered = ["planungsraum_id"]
    for var in all_vars:
        ordered += [f"{var}_slope", f"{var}_r2", f"{var}_n_months_in_slope"]
    feat = feat[ordered]
    # n_months_in_slope columns are integers
    n_months_cols = [c for c in feat.columns if c.endswith("_n_months_in_slope")]
    feat[n_months_cols] = feat[n_months_cols].astype("int64")
    return feat, all_vars, months


# %% Meta (per-variable coverage: how many PLRs got a defined slope)
def build_meta(feat: pd.DataFrame, all_vars: list, n_months: int) -> pd.DataFrame:
    recs = []
    n_plr = len(feat)
    for var in all_vars:
        s = feat[f"{var}_slope"]
        r2 = feat[f"{var}_r2"]
        nv = feat[f"{var}_n_months_in_slope"]
        recs.append({
            "variable": var,
            "n_plr_with_slope": int(s.notna().sum()),       # PLRs with a defined slope
            "n_plr_nan_slope": int(s.isna().sum()),         # PLRs below the 12-month threshold
            "mean_n_months_in_slope": round(float(nv.mean()), 2),     # avg valid months across PLRs
            "min_n_months_in_slope": int(nv.min()),
            "max_n_months_in_slope": int(nv.max()),
            "slope_mean": round(float(s.mean()), 6),        # mean annual slope (defined PLRs)
            "slope_median": round(float(s.median()), 6),
            "r2_mean": round(float(r2.mean()), 4),          # mean trend quality (defined PLRs)
            "r2_median": round(float(r2.median()), 4),
        })
    meta = pd.DataFrame(recs)
    meta.attrs["n_months"] = n_months
    meta.attrs["n_plr"] = n_plr
    return meta


# %% Run
def run() -> tuple:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    feat, all_vars, months = build_slopes()
    n_months = len(months)

    out = ANALYSIS_DIR / "slopes_panel.csv"
    feat.to_csv(out, index=False)
    print(f"  written: {out}  ({len(feat)} rows, {feat.shape[1]} cols)")

    meta = build_meta(feat, all_vars, n_months)
    meta_out = ANALYSIS_DIR / "stage6_slopes_meta.csv"
    meta.to_csv(meta_out, index=False)
    print(f"  meta written: {meta_out}  ({len(meta)} rows)")
    return feat, meta, all_vars, months


# %% Validation
def validate(feat: pd.DataFrame, meta: pd.DataFrame, all_vars: list, months: list) -> None:
    print("PLRs:", len(feat), "| months in series:", len(months),
          "| variables:", len(all_vars))
    print("feature columns:", feat.shape[1], "(= 1 id + 19 slopes + 19 r2 + 19 n_months_in_slope =",
          1 + 3 * len(all_vars), ")")

    # n_months_in_slope never exceeds the number of months
    n_months_cols = [c for c in feat.columns if c.endswith("_n_months_in_slope")]
    print("all n_months_in_slope <= n_months:",
          bool((feat[n_months_cols] <= len(months)).all().all()))

    # R^2 within [0, 1] where defined
    r2_cols = [c for c in feat.columns if c.endswith("_r2")]
    r2_vals = feat[r2_cols].to_numpy(dtype=float)
    r2_defined = r2_vals[~np.isnan(r2_vals)]
    print("all defined R^2 in [0, 1]:",
          bool((r2_defined >= -1e-9).all() and (r2_defined <= 1 + 1e-9).all()))

    # a slope is defined exactly when n_months_in_slope >= MIN_VALID_MONTHS
    ok = True
    for var in all_vars:
        defined = feat[f"{var}_slope"].notna()
        enough = feat[f"{var}_n_months_in_slope"] >= MIN_VALID_MONTHS
        if not defined.eq(enough).all():
            ok = False
            break
    print(f"slope defined  <=>  n_months_in_slope >= {MIN_VALID_MONTHS}:", ok)

    # coverage for ALL variables: list every variable that has any NaN slope,
    # sorted by how many PLRs fall below the threshold. If a variable we did NOT
    # expect to have NaN shows up here, it surfaces a problem instead of hiding it.
    cov = meta[["variable", "n_plr_with_slope", "n_plr_nan_slope",
                "mean_n_months_in_slope", "r2_mean"]].copy()
    with_nan = cov[cov["n_plr_nan_slope"] > 0].sort_values(
        "n_plr_nan_slope", ascending=False)
    print(f"\ncoverage — variables with any NaN slope (of {len(feat)} PLRs):")
    if with_nan.empty:
        print("  (none — every variable has a defined slope for all PLRs)")
    else:
        print(with_nan.to_string(index=False))

    # full per-variable coverage is always in stage6_slopes_meta.csv; also report the
    # variables that are fully covered, so nothing is implicit.
    full = cov[cov["n_plr_nan_slope"] == 0]["variable"].tolist()
    print(f"\nfully covered variables ({len(full)} of {len(all_vars)}):",
          ", ".join(full) if full else "(none)")


if __name__ == "__main__":
    feat, meta, all_vars, months = run()
    validate(feat, meta, all_vars, months)
# %%