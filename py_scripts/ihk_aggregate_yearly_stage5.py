# %% [markdown]
# # Stage 5 — Yearly aggregation  (enriched_data + churn_panel -> analysis_ready_data)
#
# Aggregates the monthly metrics to CALENDAR YEARS (2023, 2024, 2025, 2026 — partial years kept).
#
# Aggregation rules (different by quantity type):
#   * Stock counts (n_total, n_gastro, ...)  -> AVERAGE stock = sum over months / months in year
#   * Stock shares (gastro_share, ...)        -> from YEARLY SUMS = Σ numerator / Σ denominator
#   * Stock quotients (_bezirk / _berlin)     -> recomputed from the yearly shares
#   * median_age                              -> recomputed from enriched raw over all month-rows of the year
#   * Churn flows (entries, exits, movers_*)  -> YEARLY SUM (first month's NaN skipped)
#   * Churn rates                             -> from yearly sums, denominator = yearly average stock
#
# Output: five yearly panels + one merged panel, in analysis_ready_data/.

# %% Imports & configuration
from pathlib import Path
import numpy as np
import pandas as pd

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()

BASE_DIR = REPO_ROOT / "data/IHK_Berlin_Gewerbedaten"
ENRICHED_DIR = BASE_DIR / "enriched_data"
ANALYSIS_DIR = BASE_DIR / "analysis_ready_data"
CHURN_PATH = ANALYSIS_DIR / "churn_panel.csv"

FILE_PATTERN = "*_IHK_Berlin_Gewerbedaten.csv"

ID_DTYPES = {
    "opendata_id": str, "postcode": str, "planungsraum_id": str,
    "ihk_branch_id": str, "nace_id": str, "branch_top_level_id": str,
}

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
    return pd.to_datetime(name[:7], format="%Y_%m").strftime("%Y-%m")


def read_enriched(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype=ID_DTYPES, low_memory=False,
        usecols=["planungsraum_id", "business_age", "employees_range", "gastro_class", "is_tourism"],
    )
    df["business_age"] = pd.to_numeric(df["business_age"], errors="coerce")
    df["is_tourism"] = df["is_tourism"].map(lambda v: str(v).strip().lower() in ("true", "1"))
    return df


def safe_div(num, den):
    """Series / Series; zero or NaN denominator -> NaN."""
    den = pd.to_numeric(den, errors="coerce").astype(float)
    num = pd.to_numeric(num, errors="coerce").astype(float)
    return num / den.where(den != 0)


def month_flag_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Per-PLR counts for one month (sums of boolean flags). Same flag logic as Stage 4."""
    a, er, gc = df["business_age"], df["employees_range"], df["gastro_class"]
    flags = pd.DataFrame({
        "planungsraum_id": df["planungsraum_id"],
        "is_gastro": gc.isin(["aufwertung", "gesamtgastro"]),
        "is_upscale": gc.eq("aufwertung"),
        "is_tour": df["is_tourism"],
        "has_age": a.notna(),
        "age_le_1": a.le(1), "age_le_2": a.le(2), "age_le_5": a.le(5),
        "age_ge_20": a.ge(20), "age_ge_30": a.ge(30), "age_ge_40": a.ge(40),
        "is_solo": er.eq(SIZE_SOLO),
        "is_1_9": er.isin(SIZE_1_9),
        "is_10plus": er.isin(SIZE_10PLUS),
        "is_large": er.isin(SIZE_LARGE_200),
        "known_size": er.notna() & er.ne(SIZE_UNKNOWN),
        "up_le_1": gc.eq("aufwertung") & a.le(1),
        "up_le_2": gc.eq("aufwertung") & a.le(2),
        "up_le_5": gc.eq("aufwertung") & a.le(5),
    })
    bool_cols = [c for c in flags.columns if c != "planungsraum_id"]
    flags[bool_cols] = flags[bool_cols].astype("int64")
    out = flags.groupby("planungsraum_id").sum()
    out["n_total"] = flags.groupby("planungsraum_id").size()
    return out.reset_index()


# %% Step 1 — collect yearly SUMS of all counts (one enriched pass)
def collect_yearly_sums():
    files = sorted(ENRICHED_DIR.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files {FILE_PATTERN!r} in {ENRICHED_DIR}")

    monthly = []
    months_by_year = {}
    for f in files:
        month = month_from_filename(f.name)
        year = month[:4]
        months_by_year.setdefault(year, set()).add(month)
        agg = month_flag_sums(read_enriched(f))
        agg["year"] = year
        monthly.append(agg)
        print(f"{month}: summed")

    big = pd.concat(monthly, ignore_index=True)
    count_cols = [c for c in big.columns if c not in ("planungsraum_id", "year")]
    ysum = big.groupby(["planungsraum_id", "year"], as_index=False)[count_cols].sum()
    ysum["bezirk_key"] = ysum["planungsraum_id"].str[:2]
    n_months = {y: len(ms) for y, ms in months_by_year.items()}
    ysum["n_months"] = ysum["year"].map(n_months)
    return ysum, n_months, months_by_year


# %% Step 2 — yearly median age (recomputed from raw, per year, memory-bounded)
def yearly_median_age():
    files = sorted(ENRICHED_DIR.glob(FILE_PATTERN))
    by_year = {}
    for f in files:
        year = month_from_filename(f.name)[:4]
        d = pd.read_csv(f, usecols=["planungsraum_id", "business_age"], dtype={"planungsraum_id": str})
        d["business_age"] = pd.to_numeric(d["business_age"], errors="coerce")
        by_year.setdefault(year, []).append(d)
    out = []
    for year, lst in by_year.items():
        dd = pd.concat(lst, ignore_index=True)
        med = dd.groupby("planungsraum_id")["business_age"].median().rename("median_age").reset_index()
        med["year"] = year
        out.append(med)
    return pd.concat(out, ignore_index=True)


# %% Step 3 — build the four infrastructure yearly panels from ysum
def build_infrastructure(ysum: pd.DataFrame, medians: pd.DataFrame) -> dict:
    nm = ysum["n_months"]

    # --- demographics ---
    demo = pd.DataFrame({
        "planungsraum_id": ysum["planungsraum_id"], "year": ysum["year"],
        "n_total": ysum["n_total"] / nm,
        "n_with_age": ysum["has_age"] / nm,
        "n_age_missing": (ysum["n_total"] - ysum["has_age"]) / nm,
        "share_max_1y": safe_div(ysum["age_le_1"], ysum["has_age"]),
        "share_max_2y": safe_div(ysum["age_le_2"], ysum["has_age"]),
        "share_max_5y": safe_div(ysum["age_le_5"], ysum["has_age"]),
        "share_min_20y": safe_div(ysum["age_ge_20"], ysum["has_age"]),
        "share_min_30y": safe_div(ysum["age_ge_30"], ysum["has_age"]),
        "share_min_40y": safe_div(ysum["age_ge_40"], ysum["has_age"]),
        "n_known_size": ysum["known_size"] / nm,
        "n_size_unknown": (ysum["n_total"] - ysum["known_size"]) / nm,
        "share_solo": safe_div(ysum["is_solo"], ysum["known_size"]),
        "share_1_9": safe_div(ysum["is_1_9"], ysum["known_size"]),
        "share_10plus": safe_div(ysum["is_10plus"], ysum["known_size"]),
        "n_large": ysum["is_large"] / nm,
    })
    demo = demo.merge(medians, on=["planungsraum_id", "year"], how="left")
    # place median_age right after n_age_missing
    cols = list(demo.columns); cols.insert(cols.index("share_max_1y"), cols.pop(cols.index("median_age")))
    demo = demo[cols]

    # --- gastronomy (with quotients from yearly shares) ---
    g_share = safe_div(ysum["is_gastro"], ysum["n_total"])
    u_share = safe_div(ysum["is_upscale"], ysum["is_gastro"])

    bz = ysum.groupby(["bezirk_key", "year"], as_index=False)[["n_total", "is_gastro", "is_upscale", "is_tour"]].sum()
    bz["bz_g"] = safe_div(bz["is_gastro"], bz["n_total"])
    bz["bz_u"] = safe_div(bz["is_upscale"], bz["is_gastro"])
    bz["bz_t"] = safe_div(bz["is_tour"], bz["n_total"])
    be = ysum.groupby("year", as_index=False)[["n_total", "is_gastro", "is_upscale", "is_tour"]].sum()
    be["be_g"] = safe_div(be["is_gastro"], be["n_total"])
    be["be_u"] = safe_div(be["is_upscale"], be["is_gastro"])
    be["be_t"] = safe_div(be["is_tour"], be["n_total"])

    ref = ysum[["planungsraum_id", "year", "bezirk_key"]].copy()
    ref["g_share"], ref["u_share"] = g_share.values, u_share.values
    ref["t_share"] = safe_div(ysum["is_tour"], ysum["n_total"]).values
    ref = ref.merge(bz[["bezirk_key", "year", "bz_g", "bz_u", "bz_t"]], on=["bezirk_key", "year"], how="left")
    ref = ref.merge(be[["year", "be_g", "be_u", "be_t"]], on="year", how="left")

    gas = pd.DataFrame({
        "planungsraum_id": ysum["planungsraum_id"], "year": ysum["year"],
        "n_total": ysum["n_total"] / nm,
        "n_gastro": ysum["is_gastro"] / nm,
        "n_upscale": ysum["is_upscale"] / nm,
        "gastro_share": g_share,
        "upscale_share": u_share,
        "gastro_quotient_bezirk": safe_div(ref["g_share"], ref["bz_g"]),
        "gastro_quotient_berlin": safe_div(ref["g_share"], ref["be_g"]),
        "upscale_quotient_bezirk": safe_div(ref["u_share"], ref["bz_u"]),
        "upscale_quotient_berlin": safe_div(ref["u_share"], ref["be_u"]),
    })

    # --- tourism ---
    tour = pd.DataFrame({
        "planungsraum_id": ysum["planungsraum_id"], "year": ysum["year"],
        "n_total": ysum["n_total"] / nm,
        "n_tourism": ysum["is_tour"] / nm,
        "tourism_share": safe_div(ysum["is_tour"], ysum["n_total"]),
        "tourism_quotient_bezirk": safe_div(ref["t_share"], ref["bz_t"]),
        "tourism_quotient_berlin": safe_div(ref["t_share"], ref["be_t"]),
    })

    # --- age_gastro_cross ---
    cross = pd.DataFrame({
        "planungsraum_id": ysum["planungsraum_id"], "year": ysum["year"],
        "n_upscale": ysum["is_upscale"] / nm,
        "n_upscale_max_1y": ysum["up_le_1"] / nm,
        "n_upscale_max_2y": ysum["up_le_2"] / nm,
        "n_upscale_max_5y": ysum["up_le_5"] / nm,
        "share_upscale_max_1y": safe_div(ysum["up_le_1"], ysum["is_upscale"]),
        "share_upscale_max_2y": safe_div(ysum["up_le_2"], ysum["is_upscale"]),
        "share_upscale_max_5y": safe_div(ysum["up_le_5"], ysum["is_upscale"]),
    })

    return {"demographics": demo, "gastronomy": gas, "tourism": tour, "age_gastro_cross": cross}


# %% Step 4 — churn yearly (flows summed, rates from sums, denom = yearly avg stock)
def build_churn() -> pd.DataFrame:
    ch = pd.read_csv(CHURN_PATH, dtype={"planungsraum_id": str})
    ch["year"] = ch["month"].str[:4]
    g = ch.groupby(["planungsraum_id", "year"], as_index=False).agg(
        entries=("entries", "sum"),        # NaN (first month) skipped by sum
        exits=("exits", "sum"),
        movers_in=("movers_in", "sum"),
        movers_out=("movers_out", "sum"),
        n_total=("n_curr", "mean"),         # yearly average stock
    )
    denom = g["n_total"].where(g["n_total"] > 0)
    g["entry_rate"] = g["entries"] / denom
    g["exit_rate"] = g["exits"] / denom
    g["churn_rate"] = (g["entries"] + g["exits"]) / denom
    g["churn_rate_exclusive"] = ((g["entries"] - g["movers_in"]) + (g["exits"] - g["movers_out"])) / denom
    return g[["planungsraum_id", "year", "n_total", "entries", "exits", "movers_in", "movers_out",
              "entry_rate", "exit_rate", "churn_rate", "churn_rate_exclusive"]]


# %% Yearly meta (coverage + Berlin-wide average-stock headline figures)
def build_meta(ysum: pd.DataFrame, n_months: dict, months_by_year: dict) -> pd.DataFrame:
    be = ysum.groupby("year")[["n_total", "is_gastro", "is_upscale", "is_tour"]].sum()
    n_plr = ysum.groupby("year")["planungsraum_id"].nunique()

    recs = []
    for year in sorted(months_by_year):
        nm = n_months[year]
        ms = sorted(months_by_year[year])
        recs.append({
            "year": year,
            "n_months": nm,
            "months_covered": f"{ms[0]} … {ms[-1]}",
            "is_partial_year": nm < 12,
            "n_plr": int(n_plr.get(year, 0)),
            # Berlin-wide average stock (sum over PLR-months / months in year)
            "n_total": be.loc[year, "n_total"] / nm,
            "n_gastro": be.loc[year, "is_gastro"] / nm,
            "n_upscale": be.loc[year, "is_upscale"] / nm,
            "n_tourism": be.loc[year, "is_tour"] / nm,
        })
    return pd.DataFrame(recs)


# %% Run
AVG_STOCK_COLS = {  # yearly count columns that are averages -> absent PLR-year filled with 0
    "n_total", "n_with_age", "n_age_missing", "n_known_size", "n_size_unknown", "n_large",
    "n_gastro", "n_upscale", "n_tourism", "n_upscale_max_1y", "n_upscale_max_2y", "n_upscale_max_5y",
}
CHURN_FLOW_COLS = {"entries", "exits", "movers_in", "movers_out"}


def _balance(panel: pd.DataFrame, all_plrs, all_years) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([all_plrs, all_years], names=["planungsraum_id", "year"])
    panel = panel.set_index(["planungsraum_id", "year"]).reindex(grid).reset_index()
    fill0 = [c for c in panel.columns if c in AVG_STOCK_COLS or c in CHURN_FLOW_COLS]
    panel[fill0] = panel[fill0].fillna(0)
    return panel.sort_values(["year", "planungsraum_id"]).reset_index(drop=True)


def run() -> dict:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ysum, n_months, months_by_year = collect_yearly_sums()
    medians = yearly_median_age()
    panels = build_infrastructure(ysum, medians)
    panels["churn"] = build_churn()

    # balance to full PLR × year grid (counts/flows -> 0, shares/quotients/median -> NaN)
    all_plrs = sorted(set().union(*[set(p["planungsraum_id"]) for p in panels.values()]))
    all_years = sorted(set().union(*[set(p["year"]) for p in panels.values()]))
    panels = {name: _balance(p, all_plrs, all_years) for name, p in panels.items()}

    # write the five yearly panels
    for name, panel in panels.items():
        out = ANALYSIS_DIR / f"{name}_yearly.csv"
        panel.to_csv(out, index=False)
        print(f"  written: {out}  ({len(panel)} rows, {panel.shape[1]} cols)")

    # merged panel: join all on (planungsraum_id, year), dropping duplicate shared columns
    key = ["planungsraum_id", "year"]
    order = ["demographics", "gastronomy", "tourism", "age_gastro_cross", "churn"]
    merged = panels[order[0]]
    seen = set(merged.columns)
    for name in order[1:]:
        p = panels[name]
        new_cols = key + [c for c in p.columns if c not in seen and c not in key]
        merged = merged.merge(p[new_cols], on=key, how="outer")
        seen.update(new_cols)
    merged = merged.sort_values(["year", "planungsraum_id"]).reset_index(drop=True)
    merged_out = ANALYSIS_DIR / "yearly_panel_merged.csv"
    merged.to_csv(merged_out, index=False)
    print(f"  written: {merged_out}  ({len(merged)} rows, {merged.shape[1]} cols)")

    meta = build_meta(ysum, n_months, months_by_year)
    meta_out = ANALYSIS_DIR / "stage5_yearly_meta.csv"
    meta.to_csv(meta_out, index=False)
    print(f"  meta written: {meta_out}  ({len(meta)} rows)")

    return panels, merged


# %% Validation
def validate(panels: dict, merged: pd.DataFrame) -> None:
    print("rows per yearly panel:", {k: len(v) for k, v in panels.items()})
    print("merged rows:", len(merged), "| PLRs:", merged["planungsraum_id"].nunique(),
          "| years:", sorted(merged["year"].unique()))

    demo, gas, tour, cross, churn = (panels[k] for k in
                                     ["demographics", "gastronomy", "tourism", "age_gastro_cross", "churn"])
    key = ["planungsraum_id", "year"]

    # n_total identical across infrastructure panels (and ~equal to churn avg stock)
    nt_ok = (demo.set_index(key)["n_total"].round(6).eq(gas.set_index(key)["n_total"].round(6)).all()
             and demo.set_index(key)["n_total"].round(6).eq(tour.set_index(key)["n_total"].round(6)).all())
    print("n_total identical across demographics/gastronomy/tourism:", bool(nt_ok))
    # churn avg stock vs infrastructure avg stock (should match closely)
    j = demo.set_index(key)["n_total"].sub(churn.set_index(key)["n_total"]).abs()
    print("max |n_total(demo) - n_total(churn)|:", float(np.nanmax(j.values)))

    # size shares sum to 1 where defined
    m = demo["share_solo"].notna()
    s = demo.loc[m, ["share_solo", "share_1_9", "share_10plus"]].sum(axis=1)
    print("share_solo+share_1_9+share_10plus == 1 (defined):", bool(np.allclose(s, 1.0, atol=1e-9)))

    # cross-cut monotonicity (counts are averages but ordering must hold)
    print("n_upscale_max_1y <= _2y <= _5y <= n_upscale:",
          bool((cross["n_upscale_max_1y"] <= cross["n_upscale_max_2y"] + 1e-9).all()
               and (cross["n_upscale_max_2y"] <= cross["n_upscale_max_5y"] + 1e-9).all()
               and (cross["n_upscale_max_5y"] <= cross["n_upscale"] + 1e-9).all()))

    # churn rate sanity
    cm = churn["churn_rate"].notna()
    print("churn_rate_exclusive <= churn_rate (defined):",
          bool((churn.loc[cm, "churn_rate_exclusive"] <= churn.loc[cm, "churn_rate"] + 1e-9).all()))


if __name__ == "__main__":
    panels, merged = run()
    validate(panels, merged)
# %%