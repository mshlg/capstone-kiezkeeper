#%% [markdown]
# # Stage 1 — Cleaning & Profiling  (raw_data -> intermediate_data)
#
# Produces the **churn-ready (Tier 1)** files plus one consolidated **meta file**.
#
# What it does:
#   * Standardizes IDs (coded as floats and int) -> clean strings (4501044.0 -> "4501044"; NA stays NA).
#   * NORMALIZES planungsraum_id to one canonical 8-digit form across all months
#     (the raw data mixes "s_lor_plr_2021.01100312", "01100312" and 1100312.0).
#   * REBUILDS Bezirk deterministically from the planungsraum_id prefix (first 2 digits
#     = official LOR district key). Bezirk is unreliable in the raw data: empty in three
#     months (2024-08/09/10), abbreviated there, and sometimes wrong for the same PLR, resulting in duplicates.
#   * DEDUPLICATES to one row per business per month (drop_duplicates on opendata_id).
#     After the Bezirk rebuild the duplicate rows are identical, so this is lossless.
#   * DROP: a row is removed only if it is missing `opendata_id` OR
#     `planungsraum_id` (identity + neighbourhood). These are required by EVERY
#     downstream metric, so dropping here cannot create phantom churn.
#   * RECORD: missingness for ALL 20 columns, plus rows_duplicate_id, bezirk_changed,
#     bezirk_unmapped, planungsraum_id_bad_width.
#
# Runs as a script (`python stage1_clean.py`) or cell-by-cell in VS Code.
# %% Imports & configuration
from pathlib import Path
import pandas as pd
 
 
# --- PATHS ------------------------------------------------------------------
# The script lives in <repo>/py_scripts/ and the data lives in <repo>/data/.
# Anchor to the repo root so paths work no matter where you launch from, and
# stay portable (nothing hardcoded -> safe to commit).
#
# __file__ is the script's own path and exists when running the file directly.
# .resolve().parent.parent walks: py_scripts/ -> repo root.
# In the VS Code interactive window __file__ is often undefined, so fall back to
# the current working directory (which there is the repo root by default).
try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # interactive cells: no __file__
    REPO_ROOT = Path.cwd()
 
# BASE_DIR is the folder that contains raw_data / intermediate_data / analysis_ready_data
BASE_DIR = REPO_ROOT / "data/IHK_Berlin_Gewerbedaten"
RAW_DIR = BASE_DIR / "raw_data"
INTERMEDIATE_DIR = BASE_DIR / "intermediate_data"
META_PATH = INTERMEDIATE_DIR / "stage1_cleaning_meta.csv"
# ---------------------------------------------------------------------------
FILE_PATTERN = "*_IHK_Berlin_Gewerbedaten.csv"
 
# All identifier / code columns -> clean strings. IDs are labels, not quantities:
# uniform string dtype keeps every join key the same type across stages, so
# column-wise merges and churn-matching can't silently miss. The float-coded IDs
# also shed their "4501044.0" artifact here.
#
# NOTE on postcode: Berlin postcodes all start with 1 (no leading zeros), so the
# Int64->string path below is correct here. If this data ever includes other regions
# whose postcodes start with 0 (e.g. 01067), postcode must instead be read as a string
# AT READ TIME (pd.read_csv(..., dtype={"postcode": str})) — once read as int, a leading
# zero is already lost and no later conversion can recover it.
ID_COLS = [
    "opendata_id",
    "postcode",
    "ihk_branch_id",
    "nace_id",
    "branch_top_level_id",
]
 
# planungsraum_id is handled SEPARATELY (see normalize_planungsraum_id): across the
# 36 files it appears in three forms — "s_lor_plr_2021.01100312" (prefixed string),
# "01100312" (zero-padded string), and 1100312.0 (float that already dropped its
# leading zero). It must NOT go through the Int64 path, which would null the string
# forms and keep the leading zero stripped. The canonical key is the 8-digit code.
PLANUNGSRAUM_CODE_WIDTH = 8
 
# A row is dropped ONLY if it is missing one of these (identity + neighbourhood).
GLOBAL_REQUIRED = ["opendata_id", "planungsraum_id"]
 
# Fallback only: official LOR district key = first 2 digits of the 8-digit PLR code.
# Used in fix_bezirk ONLY for a PLR whose Bezirk is ambiguous in EVERY row of a file
# (no clean reference). Verify against your data before trusting it (see note below).
BEZIRK_BY_PREFIX = {
    "01": "Mitte",
    "02": "Friedrichshain-Kreuzberg",
    "03": "Pankow",
    "04": "Charlottenburg-Wilmersdorf",
    "05": "Spandau",
    "06": "Steglitz-Zehlendorf",
    "07": "Tempelhof-Schöneberg",
    "08": "Neukölln",
    "09": "Treptow-Köpenick",
    "10": "Marzahn-Hellersdorf",
    "11": "Lichtenberg",
    "12": "Reinickendorf",
}
 
# If False, cleaned files that already exist are NOT rewritten.
# (The meta file is regenerated for every file regardless, so it stays complete.)
OVERWRITE = True
 
 
# %% Helpers
def normalize_planungsraum_id(s: pd.Series) -> tuple[pd.Series, dict]:
    """Collapse the three raw forms of planungsraum_id to one canonical 8-digit string.
 
        s_lor_plr_2021.01100312  -> "01100312"   (strip prefix, keep code)
        01100312                 -> "01100312"   (already canonical)
        1100312.0                -> "01100312"   (strip .0, left-pad lost leading zero)
 
    Returns (normalized_series, diagnostics). A `planungsraum_id_bad_width` count
    flags any code that isn't 7 or 8 digits after extraction — i.e. not a valid LOR
    key — so a surprise shows up in the meta instead of silently becoming a bad key.
    """
    def extract(val):
        if pd.isna(val):
            return pd.NA
        text = str(val)
        if "." in text and not text.replace(".", "").isdigit():
            code = text.split(".")[-1]   # prefixed form: take part after the last dot
        else:
            code = text.split(".")[0]    # numeric form: drop the ".0"
        return code
 
    codes = s.map(extract)
    present = codes.dropna()
    # flag anything that won't pad cleanly to an 8-digit code (expected widths: 7 or 8)
    bad_width = int((~present.str.len().isin([PLANUNGSRAUM_CODE_WIDTH - 1, PLANUNGSRAUM_CODE_WIDTH])).sum())
    # left-pad to fixed width, restoring the leading zero the float form dropped
    normalized = codes.where(codes.isna(), codes.str.zfill(PLANUNGSRAUM_CODE_WIDTH)).astype("string")
    return normalized, {"planungsraum_id_bad_width": bad_width}
 
 
def fix_bezirk(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rebuild Bezirk deterministically from the planungsraum_id prefix.
 
    Bezirk is unreliable in the raw data: in three months (2024-08/09/10) the column is
    almost entirely empty, in those months the few present values use short names
    ("Friedrichshain" instead of "Friedrichshain-Kreuzberg"), and in other months the
    same PLR can carry several different (wrong) Bezirk values. planungsraum_id, by
    contrast, is complete and clean, and its first two digits ARE the official LOR
    district key (verified against all well-populated months). So we set Bezirk for
    EVERY row from BEZIRK_BY_PREFIX — one rule, always correct, no guessing.
 
    This also fixes the duplicate-id rows (same PLR, mismatched Bezirk) so they become
    identical and the later dedup is lossless. Returns (df, diagnostics).
    """
    before = df["Bezirk"]
    prefix = df["planungsraum_id"].str[:2]
    rebuilt = prefix.map(BEZIRK_BY_PREFIX)
 
    # diagnostics: how many Bezirk values changed, and any prefix with no mapping
    n_changed = int((before.fillna("") != rebuilt.fillna("")).sum())
    n_unmapped = int(rebuilt.isna().sum())  # rows whose PLR prefix is not in BEZIRK_BY_PREFIX
 
    df["Bezirk"] = rebuilt
    return df, {"bezirk_changed": n_changed, "bezirk_unmapped": n_unmapped}
 
 
def standardize_ids(df: pd.DataFrame, id_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Coerce float- or int-coded IDs to clean nullable-integer-backed strings.
    4501044.0 -> "4501044"; missing values stay <NA>. Returns (df, diagnostics).
    A `<col>_non_integer` count is recorded so a non-ID column hiding here would
    show up; if a column cannot be converted it is left untouched and flagged
    rather than crashing the run.
    """
    diags: dict = {} #set up dict > becomes meta columns
    for col in id_cols: #loop through id_cols
        if col not in df.columns: # column missing from this file?
            continue # skip it rather than crash (handles schema drift)
        s = df[col]
        s_num = pd.to_numeric(s, errors="coerce") # force numeric; non-numbers -> NaN (avoids string-% crash on text columns)
        non_na = s_num.dropna() # look only at present numeric values (can't do % 1 on NaN)
        non_integer = int((non_na % 1 != 0).sum()) if len(non_na) else 0 # count values with a fractional part (suspicious ID)
        diags[f"{col}_non_integer"] = non_integer  # record that count (should be 0 for a real ID)
        try:
            df[col] = s_num.astype("Int64").astype("string")  # 4501044.0 -> 4501044 (NA kept) -> "4501044"
        except (TypeError, ValueError): # conversion failed (e.g. genuine fractional values)
            diags[f"{col}_standardize_failed"] = 1 # flag it, leave column as-is, keep the run alive
    return df, diags # hand back the modified frame + the diagnostics
 
 
def month_from_filename(name: str) -> str:
    """'2023_07_IHK_Berlin_Gewerbedaten.csv' -> '2023-07' (consistent join key)."""
    return pd.to_datetime(name[:7], format="%Y_%m").strftime("%Y-%m")
 
 
def clean_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Clean a single monthly file and return (cleaned_df, meta_record)."""
 
    df = pd.read_csv(path)
    n_before = len(df)
 
    # Record missing values for ALL columns BEFORE any deletion.
    missing_per_col = df.isna().sum()
 
    # Normalize planungsraum_id to the canonical 8-digit string FIRST, so the
    # drop reason and core drop below see the cleaned key (raw forms aren't NaN,
    # just inconsistently formatted, so counting before this would mislead).
    df["planungsraum_id"], plr_diags = normalize_planungsraum_id(df["planungsraum_id"])
 
    # Why rows will be dropped, broken out per core field.
    drop_reason = {
        f"dropped_missing_{col}": int(df[col].isna().sum()) for col in GLOBAL_REQUIRED
    }
 
    # Standardize float and int-coded IDs -> clean strings.
    df, id_diags = standardize_ids(df, ID_COLS)
 
    # Apply the narrow core drop.
    keep_mask = df[GLOBAL_REQUIRED].notna().all(axis=1)
    df_clean = df[keep_mask].copy()
 
    # Repair Bezirk from planungsraum_id (only Bezirk is corrupted; other geo cols are fine).
    df_clean, bezirk_diags = fix_bezirk(df_clean)
 
    # One row per business per month: after the Bezirk repair the duplicate-id rows are
    # identical, so dropping them is lossless. This is a frozen property of Tier 1.
    n_before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates("opendata_id", keep="first")
    n_duplicate_id = n_before_dedup - len(df_clean)
 
    n_after = len(df_clean)
 
    meta = {
        "source_file": path.name,
        "month": month_from_filename(path.name),
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_before - n_after,
        "rows_duplicate_id": n_duplicate_id,
    }
    meta.update(drop_reason)
    meta.update({f"missing_{c}": int(missing_per_col[c]) for c in df.columns})
    meta.update(id_diags)
    meta.update(plr_diags) # planungsraum_id_bad_width diagnostic
    meta.update(bezirk_diags) # bezirk_changed, bezirk_ambiguous_plr
    return df_clean, meta
 
 
# %% Run
def run() -> pd.DataFrame:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True) #Creates intermediate_data if not there. also creates any missing parent folders
 
    files = sorted(RAW_DIR.glob(FILE_PATTERN)) #finds everything matching the pattern and sorts according to date
    if not files: #raise error if no files are found
        raise FileNotFoundError(f"No files matching {FILE_PATTERN!r} in {RAW_DIR}")
 
    meta_records = [] #collects meta dict per file as loop runs
 
    for path in files:
        out_path = INTERMEDIATE_DIR / path.name #sets output path
        df_clean, meta = clean_file(path)  # meta is always computed
 
        if out_path.exists() and not OVERWRITE: #If the output already exists and OVERWRITE is False, skip writing
            meta["written"] = False
        else:
            df_clean.to_csv(out_path, index=False)
            meta["written"] = True
 
        meta_records.append(meta) #add meta to list
        print(
            f"{path.name}: {meta['rows_before']:>7} -> {meta['rows_after']:>7} " #progress print
            f"({meta['rows_dropped']} dropped)"
        )
 
    meta_df = pd.DataFrame(meta_records)
 
    # Tidy column order: summary first, then per-column missingness / diagnostics.
    front = (
        ["source_file", "month", "rows_before", "rows_after", "rows_dropped", "written"]
        + [f"dropped_missing_{c}" for c in GLOBAL_REQUIRED]
    )
 
    cols = front + [c for c in meta_df.columns if c not in front]
    meta_df = meta_df[cols]
 
    meta_df.to_csv(META_PATH, index=False)
    print(f"\nMeta written: {META_PATH}  ({len(meta_df)} rows)")
    return meta_df
 
 
if __name__ == "__main__":
    meta_df = run()

# %%
