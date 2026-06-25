# %% [markdown]
# # Stage 4 — Aggregation  (enriched_data -> analysis_ready_data)
#
# Builds four PLR × month panels from the enriched business-level data:
#   * demographics.csv      — age + employee-size structure
#   * gastronomy.csv        — gastronomy / upscale shares + location quotients
#   * tourism.csv           — accommodation share + location quotients
#   * age_gastro_cross.csv  — young upscale businesses (age × gastro cross-cut)
#
# Principles (see data_dictionary_GI-Dimension):
#   * Each panel is keyed on (planungsraum_id, month) and is self-contained.
#   * Shared columns (n_total, n_upscale) are computed identically -> merge-safe.
#   * Denominators differ by dimension; zero/undefined denominator -> NaN (never 0, never imputed).
#   * Location quotients: _bezirk vs. the PLR's district, _berlin vs. all of Berlin (same month).
#   * Bezirk grouping uses the first 2 digits of planungsraum_id (official LOR district key).

# %% Imports & configuration
from pathlib import Path
import numpy as np
import pandas as pd

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()

BASE_DIR = REPO_ROOT / "data/commercial/IHK_Berlin_Gewerbedaten"
ENRICHED_DIR = BASE_DIR / "enriched_data"
ANALYSIS_DIR = BASE_DIR / "analysis_ready_data"

FILE_PATTERN = "*_IHK_Berlin_Gewerbedaten.csv"   # matches monthly files only (not scheme/meta CSVs)

# IDs as strings so codes keep their exact form (leading zeros, full length).
ID_DTYPES = {
    "opendata_id": str, "postcode": str, "planungsraum_id": str,
    "ihk_branch_id": str, "nace_id": str, "branch_top_level_id": str,
}

# employees_range bins (ordinal category strings, exactly as they appear in the data)
SIZE_SOLO = "0 Beschäftigte"
SIZE_1_9 = {"1 - 3 Beschäftigte", "4 - 6 Beschäftigte", "7 - 9 Beschäftigte"}
SIZE_10PLUS = {
    "10 - 19 Beschäftigte", "20 - 49 Beschäftigte", "50 - 99 Beschäftigte",
    "100 - 199 Beschäftigte", "200 - 499 Beschäftigte", "500 - 999 Beschäftigte",
    "1000 - 2499 Beschäftigte", "2500 - 4999 Beschäftigte", "5000 - 7499 Beschäftigte",
    "7500 - 9999 Beschäftigte", "10000 und mehr Beschäftigte",
}
SIZE_LARGE_200 = {
    "200 - 499 Beschäftigte", "500 - 999 Beschäftigte", "1000 - 2499 Beschäftigte",
    "2500 - 4999 Beschäftigte", "5000 - 7499 Beschäftigte", "7500 - 9999 Beschäftigte",
    "10000 und mehr Beschäftigte",
}
SIZE_UNKNOWN = "unbekannt"


# %% Helpers
def month_from_filename(name: str) -> str:
    """'2023_07_IHK_Berlin_Gewerbedaten.csv' -> '2023-07'."""
    return pd.to_datetime(name[:7], format="%Y_%m").strftime("%Y-%m")


def read_enriched(path: Path) -> pd.DataFrame:
    """Read one enriched month, only the columns the aggregation needs, with correct dtypes."""
    df = pd.read_csv(
        path, dtype=ID_DTYPES, low_memory=False,
        usecols=["planungsraum_id", "business_age", "employees_range", "gastro_class", "is_tourism"],
    )
    df["business_age"] = pd.to_numeric(df["business_age"], errors="coerce")
    # normalize is_tourism to a real bool (CSV may store it as "True"/"False" text)
    df["is_tourism"] = df["is_tourism"].map(lambda v: str(v).strip().lower() in ("true", "1"))
    return df


def safe_div(num, den):
    """Series / Series with zero or NaN denominator -> NaN (never 0, never imputed)."""
    den = pd.to_numeric(den, errors="coerce").astype(float)
    return num / den.where(den != 0)


def scalar_div(num_series, den):
    """Series / scalar with zero/NaN/None denominator -> all-NaN."""
    if den is None or pd.isna(den) or den == 0:
        return pd.Series(np.nan, index=num_series.index)
    return num_series / den


def aggregate_month(df: pd.DataFrame) -> pd.DataFrame:
    """Per-PLR raw counts for one month (one row per PLR). Shares/quotients derived later."""
    a, er, gc = df["business_age"], df["employees_range"], df["gastro_class"]

    flags = pd.DataFrame({
        "planungsraum_id": df["planungsraum_id"],
        # gastro / tourism
        "is_gastro": gc.isin(["aufwertung", "gesamtgastro"]),
        "is_upscale": gc.eq("aufwertung"),
        "is_tour": df["is_tourism"],
        # age
        "has_age": a.notna(),
        "age_le_1": a.le(1), "age_le_2": a.le(2), "age_le_5": a.le(5),
        "age_ge_20": a.ge(20), "age_ge_30": a.ge(30), "age_ge_40": a.ge(40),
        # size
        "is_solo": er.eq(SIZE_SOLO),
        "is_1_9": er.isin(SIZE_1_9),
        "is_10plus": er.isin(SIZE_10PLUS),
        "is_large": er.isin(SIZE_LARGE_200),
        "known_size": er.notna() & er.ne(SIZE_UNKNOWN),
        # cross-cut: young upscale
        "up_le_1": gc.eq("aufwertung") & a.le(1),
        "up_le_2": gc.eq("aufwertung") & a.le(2),
        "up_le_5": gc.eq("aufwertung") & a.le(5),
    })
    bool_cols = [c for c in flags.columns if c != "planungsraum_id"]
    flags[bool_cols] = flags[bool_cols].astype("int64")

    g = flags.groupby("planungsraum_id")
    agg = g.sum()
    agg["n_total"] = g.size()
    agg["median_age"] = df.groupby("planungsraum_id")["business_age"].median()  # NaN skipped
    agg = agg.reset_index()
    agg["bezirk_key"] = agg["planungsraum_id"].str[:2]  # official LOR district key
    return agg


# %% Panel builders (each takes the per-PLR agg for one month, returns that month's panel rows)
def panel_demographics(agg: pd.DataFrame, month: str) -> pd.DataFrame:
    return pd.DataFrame({
        "planungsraum_id": agg["planungsraum_id"],
        "month": month,
        "n_total": agg["n_total"],
        "n_with_age": agg["has_age"],
        "n_age_missing": agg["n_total"] - agg["has_age"],
        "median_age": agg["median_age"],
        "share_max_1y": safe_div(agg["age_le_1"], agg["has_age"]),
        "share_max_2y": safe_div(agg["age_le_2"], agg["has_age"]),
        "share_max_5y": safe_div(agg["age_le_5"], agg["has_age"]),
        "share_min_20y": safe_div(agg["age_ge_20"], agg["has_age"]),
        "share_min_30y": safe_div(agg["age_ge_30"], agg["has_age"]),
        "share_min_40y": safe_div(agg["age_ge_40"], agg["has_age"]),
        "n_known_size": agg["known_size"],
        "n_size_unknown": agg["n_total"] - agg["known_size"],
        "share_solo": safe_div(agg["is_solo"], agg["known_size"]),
        "share_1_9": safe_div(agg["is_1_9"], agg["known_size"]),
        "share_10plus": safe_div(agg["is_10plus"], agg["known_size"]),
        "n_large": agg["is_large"],
    })


def panel_gastronomy(agg: pd.DataFrame, month: str) -> pd.DataFrame:
    # PLR-level shares
    gastro_share = safe_div(agg["is_gastro"], agg["n_total"])
    upscale_share = safe_div(agg["is_upscale"], agg["is_gastro"])

    # Bezirk-level reference shares (summed counts per district)
    bez = agg.groupby("bezirk_key")[["n_total", "is_gastro", "is_upscale"]].sum()
    bez_gastro_share = (bez["is_gastro"] / bez["n_total"].where(bez["n_total"] != 0))
    bez_upscale_share = (bez["is_upscale"] / bez["is_gastro"].where(bez["is_gastro"] != 0))
    ref_bz_gastro = agg["bezirk_key"].map(bez_gastro_share)
    ref_bz_upscale = agg["bezirk_key"].map(bez_upscale_share)

    # Berlin-level reference shares (scalars)
    B_total, B_gastro, B_upscale = agg["n_total"].sum(), agg["is_gastro"].sum(), agg["is_upscale"].sum()
    berlin_gastro_share = (B_gastro / B_total) if B_total else np.nan
    berlin_upscale_share = (B_upscale / B_gastro) if B_gastro else np.nan

    return pd.DataFrame({
        "planungsraum_id": agg["planungsraum_id"],
        "month": month,
        "n_total": agg["n_total"],
        "n_gastro": agg["is_gastro"],
        "n_upscale": agg["is_upscale"],
        "gastro_share": gastro_share,
        "upscale_share": upscale_share,
        "gastro_quotient_bezirk": safe_div(gastro_share, ref_bz_gastro),
        "gastro_quotient_berlin": scalar_div(gastro_share, berlin_gastro_share),
        "upscale_quotient_bezirk": safe_div(upscale_share, ref_bz_upscale),
        "upscale_quotient_berlin": scalar_div(upscale_share, berlin_upscale_share),
    })


def panel_tourism(agg: pd.DataFrame, month: str) -> pd.DataFrame:
    tourism_share = safe_div(agg["is_tour"], agg["n_total"])

    bez = agg.groupby("bezirk_key")[["n_total", "is_tour"]].sum()
    bez_tour_share = (bez["is_tour"] / bez["n_total"].where(bez["n_total"] != 0))
    ref_bz_tour = agg["bezirk_key"].map(bez_tour_share)

    B_total, B_tour = agg["n_total"].sum(), agg["is_tour"].sum()
    berlin_tour_share = (B_tour / B_total) if B_total else np.nan

    return pd.DataFrame({
        "planungsraum_id": agg["planungsraum_id"],
        "month": month,
        "n_total": agg["n_total"],
        "n_tourism": agg["is_tour"],
        "tourism_share": tourism_share,
        "tourism_quotient_bezirk": safe_div(tourism_share, ref_bz_tour),
        "tourism_quotient_berlin": scalar_div(tourism_share, berlin_tour_share),
    })


def panel_age_gastro_cross(agg: pd.DataFrame, month: str) -> pd.DataFrame:
    return pd.DataFrame({
        "planungsraum_id": agg["planungsraum_id"],
        "month": month,
        "n_upscale": agg["is_upscale"],
        "n_upscale_max_1y": agg["up_le_1"],
        "n_upscale_max_2y": agg["up_le_2"],
        "n_upscale_max_5y": agg["up_le_5"],
        "share_upscale_max_1y": safe_div(agg["up_le_1"], agg["is_upscale"]),
        "share_upscale_max_2y": safe_div(agg["up_le_2"], agg["is_upscale"]),
        "share_upscale_max_5y": safe_div(agg["up_le_5"], agg["is_upscale"]),
    })


# %% Run
PANELS = {
    "demographics": panel_demographics,
    "gastronomy": panel_gastronomy,
    "tourism": panel_tourism,
    "age_gastro_cross": panel_age_gastro_cross,
}


def run() -> dict:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(ENRICHED_DIR.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files {FILE_PATTERN!r} in {ENRICHED_DIR}")

    parts = {name: [] for name in PANELS}
    meta_records = []
    for path in files:
        month = month_from_filename(path.name)
        agg = aggregate_month(read_enriched(path))
        for name, builder in PANELS.items():
            parts[name].append(builder(agg, month))
        meta_records.append({
            "month": month,
            "n_plr_present": len(agg),                         # PLRs with businesses this month
            "n_total": int(agg["n_total"].sum()),             # Berlin-wide counts
            "n_gastro": int(agg["is_gastro"].sum()),
            "n_upscale": int(agg["is_upscale"].sum()),
            "n_tourism": int(agg["is_tour"].sum()),
            "n_age_missing": int((agg["n_total"] - agg["has_age"]).sum()),
            "n_size_unknown": int((agg["n_total"] - agg["known_size"]).sum()),
        })
        print(f"{month}: {len(agg)} PLRs aggregated")

    # full balanced grid: every PLR (union across all months) × every month.
    # PLRs absent in a month had 0 businesses -> count columns become 0, shares/quotients stay NaN.
    all_plrs = sorted({p for part in parts["demographics"] for p in part["planungsraum_id"]})
    all_months = [month_from_filename(f.name) for f in files]
    grid = pd.MultiIndex.from_product([all_plrs, all_months], names=["planungsraum_id", "month"])

    # count columns to fill with 0 when a PLR-month is absent (everything else -> NaN)
    count_cols = {
        "n_total", "n_with_age", "n_age_missing", "n_known_size", "n_size_unknown", "n_large",
        "n_gastro", "n_upscale", "n_tourism",
        "n_upscale_max_1y", "n_upscale_max_2y", "n_upscale_max_5y",
    }

    panels = {}
    for name, builder in PANELS.items():
        panel = pd.concat(parts[name], ignore_index=True)
        panel = (panel.set_index(["planungsraum_id", "month"])
                      .reindex(grid)        # force every PLR into every month
                      .reset_index())
        # absent PLR-months: counts are truly 0; shares/quotients stay NaN (undefined)
        fill0 = [c for c in panel.columns if c in count_cols]
        panel[fill0] = panel[fill0].fillna(0).astype("int64")
        panel = panel.sort_values(["month", "planungsraum_id"]).reset_index(drop=True)
        out = ANALYSIS_DIR / f"{name}.csv"
        panel.to_csv(out, index=False)
        panels[name] = panel
        print(f"  written: {out}  ({len(panel)} rows, {panel.shape[1]} cols)")

    # monthly meta: grid occupancy + Berlin-wide headline counts + coverage
    n_plr_full = len(all_plrs)
    meta = pd.DataFrame(meta_records)
    meta.insert(2, "n_plr_missing", n_plr_full - meta["n_plr_present"])  # filled with 0-rows
    meta = meta[["month", "n_plr_present", "n_plr_missing",
                 "n_total", "n_gastro", "n_upscale", "n_tourism",
                 "n_age_missing", "n_size_unknown"]]
    meta_out = ANALYSIS_DIR / "stage4_aggregate_meta.csv"
    meta.to_csv(meta_out, index=False)
    print(f"  meta written: {meta_out}  ({len(meta)} rows)")
    return panels


# %% Validation (optional sanity checks)
def validate(panels: dict) -> None:
    demo = panels["demographics"]
    gas = panels["gastronomy"]
    tour = panels["tourism"]
    cross = panels["age_gastro_cross"]

    print("rows per panel:", {k: len(v) for k, v in panels.items()})
    print("PLRs:", demo["planungsraum_id"].nunique(), "| months:", demo["month"].nunique())

    # count identities
    print("n_with_age + n_age_missing == n_total:",
          (demo["n_with_age"] + demo["n_age_missing"] == demo["n_total"]).all())
    print("n_known_size + n_size_unknown == n_total:",
          (demo["n_known_size"] + demo["n_size_unknown"] == demo["n_total"]).all())

    # size shares sum to 1 where known_size > 0
    m = demo["n_known_size"] > 0
    s = demo.loc[m, ["share_solo", "share_1_9", "share_10plus"]].sum(axis=1)
    print("share_solo+share_1_9+share_10plus == 1 (known_size>0):",
          bool(np.allclose(s, 1.0, atol=1e-9)))

    # cross-cut monotonicity
    print("n_upscale_max_1y <= _2y <= _5y <= n_upscale:",
          bool((cross["n_upscale_max_1y"] <= cross["n_upscale_max_2y"]).all()
               and (cross["n_upscale_max_2y"] <= cross["n_upscale_max_5y"]).all()
               and (cross["n_upscale_max_5y"] <= cross["n_upscale"]).all()))

    # shared columns identical across panels (merge-safety)
    key = ["planungsraum_id", "month"]
    nt = (demo.set_index(key)["n_total"]
          .eq(gas.set_index(key)["n_total"]).all()
          and demo.set_index(key)["n_total"].eq(tour.set_index(key)["n_total"]).all())
    nu = gas.set_index(key)["n_upscale"].eq(cross.set_index(key)["n_upscale"]).all()
    print("n_total identical across demographics/gastronomy/tourism:", bool(nt))
    print("n_upscale identical across gastronomy/age_gastro_cross:", bool(nu))


if __name__ == "__main__":
    panels = run()
    validate(panels)
# %%