# %% [markdown]
# # Stage 7 — Feature assembly  (yearly_panel_merged + slopes_panel -> feature_matrix)
#
# Combines, per PLR, two complementary views of every variable into one feature matrix
# for the downstream models (K-Means clustering, classification):
#   * LEVEL  — the 2026 yearly value (the current state), from yearly_panel_merged.csv
#   * SLOPE  — the annualised monthly trend, from slopes_panel.csv (Stage 6)
#   * R2     — the linear-trend quality of that slope (Stage 6)
#
# 22 variables x (level + slope + r2) = 66 feature columns.
# PLUS gastronomy size context (kept raw, NOT transformed here):
#   * n_gastro, n_upscale          — average-stock denominators (2026 level, raw)
#   * has_gastro_structure         — binary, 1 if n_gastro (2026) >= 5 (a PLR with at
#                                    least a minimal gastronomic base, where upscale
#                                    shares are meaningful)
# = 60 feature columns + plr_id = 61 columns total, 542 rows.
#
# Output: data/final_datasets/feature_matrix_commercial.csv  (the model-ready table).
#
# Cross-dimension naming (for merging with the real-estate dimension later):
#   * key column is plr_id (8-digit string), same as the housing dimension
#   * every feature column is prefixed com_ (commercial), e.g. com_gastro_share_slope
# Within a variable: levels get _level_2026, trends keep _slope, quality keeps _r2.
#
# NOTE on NaN: both inputs carry meaningful NaNs (a level is NaN where the 2026
# denominator was 0; a slope is NaN where the PLR had < 12 valid months). These are
# preserved, NOT imputed. n_gastro/n_upscale are kept raw (log/scaling happens later,
# in the model preprocessing). XGBoost handles NaN natively; K-Means needs them
# resolved (drop or impute) BEFORE clustering — that decision belongs to the
# modelling step, not here.

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
FINAL_DIR = REPO_ROOT / "data/final_datasets"   # model-ready output lives here

YEARLY_PATH = ANALYSIS_DIR / "yearly_panel_merged.csv"
SLOPES_PATH = ANALYSIS_DIR / "slopes_panel.csv"
LEVEL_YEAR = "2026"
GASTRO_STRUCTURE_MIN = 5      # n_gastro >= 5 -> has_gastro_structure = 1

# Output conventions for cross-dimension merging (matches the real-estate dimension):
#   * key column renamed planungsraum_id -> plr_id (same 8-digit string)
#   * every feature column prefixed com_ (commercial), so a later merge with the
#     re_-prefixed housing features keeps each variable's origin unambiguous.
KEY_OUT = "plr_id"
PREFIX = "com_"

# The variables that have both a 2026 level and a slope (must match Stage 6).
# The share_upscale_max_* group was dropped (denominator n_upscale too small in ~36%
# of PLRs to make the young-share meaningful).
VARS = [
    # demographics
    "median_age",
    "share_max_1y", "share_max_2y", "share_max_5y",
    "share_min_20y", "share_min_30y", "share_min_40y",
    "share_solo", "share_1_9", "share_10plus",
    # gastronomy
    "gastro_share", "upscale_share",
    "gastro_quotient_bezirk", "gastro_quotient_berlin",
    "upscale_quotient_bezirk", "upscale_quotient_berlin",
    # tourism
    "tourism_share", "tourism_quotient_bezirk", "tourism_quotient_berlin",
    # churn rates (incl. moves between PLRs — relevant for displacement/influx)
    "entry_rate", "exit_rate", "churn_rate",
]

# raw gastronomy size denominators to carry over as features (2026 level, kept raw)
SIZE_VARS = ["n_gastro", "n_upscale"]


# %% Build
def build_feature_matrix() -> pd.DataFrame:
    # --- 2026 levels ---
    yearly = pd.read_csv(YEARLY_PATH, dtype={"planungsraum_id": str, "year": str})
    level = yearly[yearly["year"] == LEVEL_YEAR].copy()
    if level.empty:
        raise ValueError(f"no rows for year {LEVEL_YEAR} in {YEARLY_PATH}")
    missing = [v for v in VARS if v not in level.columns]
    if missing:
        raise KeyError(f"variables missing in yearly panel: {missing}")
    miss_size = [v for v in SIZE_VARS if v not in level.columns]
    if miss_size:
        raise KeyError(f"size variables missing in yearly panel: {miss_size}")
    # keep the modelling vars (renamed to _level_2026) plus the raw size denominators
    keep = level[["planungsraum_id"] + VARS + SIZE_VARS].copy()
    keep = keep.rename(columns={v: f"{v}_level_2026" for v in VARS})

    # --- slopes + r2 ---
    slopes = pd.read_csv(SLOPES_PATH, dtype={"planungsraum_id": str})
    slope_cols = [f"{v}_slope" for v in VARS]
    r2_cols = [f"{v}_r2" for v in VARS]
    miss_s = [c for c in slope_cols + r2_cols if c not in slopes.columns]
    if miss_s:
        raise KeyError(f"columns missing in slopes panel: {miss_s}")
    slopes = slopes[["planungsraum_id"] + slope_cols + r2_cols]

    # --- merge on PLR ---
    feat = keep.merge(slopes, on="planungsraum_id", how="outer")

    # --- structure flag from 2026 n_gastro (>= 5 -> has gastronomic base) ---
    # NaN-safe: a PLR with NaN/0 n_gastro is 0 (no structure). Stored as nullable Int.
    feat["has_gastro_structure"] = (
        (feat["n_gastro"].fillna(0) >= GASTRO_STRUCTURE_MIN).astype("int64")
    )

    # tidy column order: id, then per variable (level, slope, r2), then size context
    ordered = ["planungsraum_id"]
    for v in VARS:
        ordered += [f"{v}_level_2026", f"{v}_slope", f"{v}_r2"]
    ordered += ["n_gastro", "n_upscale", "has_gastro_structure"]
    feat = feat[ordered].sort_values("planungsraum_id").reset_index(drop=True)

    # cross-dimension naming: key -> plr_id, every feature column prefixed com_
    feat = feat.rename(columns={"planungsraum_id": KEY_OUT})
    feat = feat.rename(columns={c: f"{PREFIX}{c}" for c in feat.columns if c != KEY_OUT})
    return feat


# %% Run
def run() -> pd.DataFrame:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    feat = build_feature_matrix()
    out = FINAL_DIR / "feature_matrix_commercial.csv"
    feat.to_csv(out, index=False)
    print(f"  written: {out}  ({len(feat)} rows, {feat.shape[1]} cols)")
    return feat


# %% Validation
def validate(feat: pd.DataFrame) -> None:
    n_feat_cols = feat.shape[1] - 1
    expected = 3 * len(VARS) + len(SIZE_VARS) + 1   # 3 per var + 2 counts + 1 flag
    print("PLRs:", len(feat), "| feature columns:", n_feat_cols,
          f"(= {len(VARS)}x3 + n_gastro + n_upscale + has_gastro_structure = {expected})")
    print("unique PLRs:", feat[KEY_OUT].nunique())
    print("no duplicate columns:", feat.columns.is_unique)
    print(f"key column '{KEY_OUT}' present:", KEY_OUT in feat.columns)
    print(f"all feature cols prefixed '{PREFIX}':",
          all(c.startswith(PREFIX) for c in feat.columns if c != KEY_OUT))

    # NaN overview per block (informative, not a failure)
    level_cols = [f"{PREFIX}{v}_level_2026" for v in VARS]
    slope_cols = [f"{PREFIX}{v}_slope" for v in VARS]
    r2_cols = [f"{PREFIX}{v}_r2" for v in VARS]
    print("\nNaN counts (of", len(feat), "PLRs):")
    print("  levels  (2026):  total", int(feat[level_cols].isna().sum().sum()),
          "| max per column", int(feat[level_cols].isna().sum().max()))
    print("  slopes:          total", int(feat[slope_cols].isna().sum().sum()),
          "| max per column", int(feat[slope_cols].isna().sum().max()))
    print("  r2:              total", int(feat[r2_cols].isna().sum().sum()),
          "| max per column", int(feat[r2_cols].isna().sum().max()))
    print("  n_gastro NaN:", int(feat[f"{PREFIX}n_gastro"].isna().sum()),
          "| n_upscale NaN:", int(feat[f"{PREFIX}n_upscale"].isna().sum()))

    # structure flag distribution + consistency with n_gastro
    struct_col = f"{PREFIX}has_gastro_structure"
    ng_col = f"{PREFIX}n_gastro"
    n_struct = int(feat[struct_col].sum())
    print(f"\n{struct_col}: {n_struct} of {len(feat)} PLRs = 1 "
          f"({n_struct/len(feat):.1%})")
    check = ((feat[ng_col].fillna(0) >= GASTRO_STRUCTURE_MIN).astype(int)
             == feat[struct_col]).all()
    print(f"{struct_col} == (n_gastro >= 5):", bool(check))

    # show the variables with the most level-NaN and slope-NaN
    lvl_na = feat[level_cols].isna().sum().sort_values(ascending=False)
    slp_na = feat[slope_cols].isna().sum().sort_values(ascending=False)
    print("\ntop level-NaN variables:")
    print(lvl_na[lvl_na > 0].head(6).to_string() or "  (none)")
    print("\ntop slope-NaN variables:")
    print(slp_na[slp_na > 0].head(6).to_string() or "  (none)")


if __name__ == "__main__":
    feat = run()
    validate(feat)
# %%