# %% [markdown]
# # Stage 3 — Enrichment  (intermediate_data -> enriched_data)
#
# Adds branch classification labels to every business, WITHOUT removing any row
# (enrichment, not filtering — filtering happens later, per metric, at aggregation).
#
# What it does:
#   * Picks the correct WZ scheme per month: WZ2008 up to 2025-05, WZ2025 from 2025-06.
#   * Matches ihk_branch_id EXACTLY (full code length) against that scheme.
#   * Adds columns:
#       - gastro_class : "aufwertung" | "gesamtgastro" | <NA>   (the two counted buckets;
#                        neutral / gemischt / Sammelcode / non-gastro -> <NA> = not counted)
#       - is_tourism   : True for Beherbergung codes flagged "relevant" (Touristifizierung)
#       - wz_scheme    : which scheme was applied (traceability across the switch)
#   * RECORD: gastro_unmapped = codes starting 55/56 that are NOT in the scheme.
#
# IDs are read as strings so planungsraum_id / ihk_branch_id keep their exact form.

# %% Imports & configuration
from pathlib import Path
import pandas as pd

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()

BASE_DIR = REPO_ROOT / "data/IHK_Berlin_Gewerbedaten"
INTERMEDIATE_DIR = BASE_DIR / "intermediate_data"
ENRICHED_DIR = BASE_DIR / "enriched_data"            # NEW output folder (created if missing)

# Classification scheme files live in their own input folder (kept separate from
# the enriched_data output so regenerating the output can't touch them).
CLASS_DIR = BASE_DIR / "classification_docs"
WZ2008_PATH = CLASS_DIR / "WZ2008_categories_until_may2025.csv"
WZ2025_PATH = CLASS_DIR / "WZ2025_categories_from_jun2025.csv"

FILE_PATTERN = "*_IHK_Berlin_Gewerbedaten.csv"

# First month that uses WZ2025 (everything before uses WZ2008).
SCHEME_SWITCH = "2025-06"

# Read IDs as strings so codes keep their exact form (leading zeros, full length).
ID_DTYPES = {
    "opendata_id": str, "postcode": str, "planungsraum_id": str,
    "ihk_branch_id": str, "nace_id": str, "branch_top_level_id": str,
}


# %% Helpers
def month_from_filename(name: str) -> str:
    """'2023_07_IHK_Berlin_Gewerbedaten.csv' -> '2023-07'."""
    return pd.to_datetime(name[:7], format="%Y_%m").strftime("%Y-%m")


def load_scheme(path: Path) -> pd.DataFrame:
    """Load a WZ classification file into a lookup keyed by exact ihk_branch_id.

    Bewertung -> gastro_class: only "Aufwertung" and "Gesamtgastro" are kept as the
    two counted buckets; neutral / gemischt / Sammelcode / empty become <NA> (not
    counted). Touristifizierung == "relevant" -> is_tourism.
    """
    df = pd.read_csv(path, dtype={"ihk_branch_id": str})
    bew = df["Bewertung"].astype("string").str.strip()           # "Gesamtgastro " -> "Gesamtgastro"
    tour = df["Touristifizierung"].astype("string").str.strip()

    gastro_class = pd.Series(pd.NA, index=df.index, dtype="string")
    gastro_class[bew == "Aufwertung"] = "aufwertung"
    gastro_class[bew == "Gesamtgastro"] = "gesamtgastro"

    out = pd.DataFrame({
        "ihk_branch_id": df["ihk_branch_id"],
        "gastro_class": gastro_class,
        "is_tourism": (tour == "relevant").fillna(False),
    }).dropna(subset=["ihk_branch_id"]).drop_duplicates("ihk_branch_id")
    return out.set_index("ihk_branch_id")


def enrich_file(path: Path, scheme: pd.DataFrame, scheme_name: str):
    """Add gastro_class, is_tourism, wz_scheme to one month's file. Returns (df, diagnostics)."""
    df = pd.read_csv(path, dtype=ID_DTYPES, low_memory=False)

    # exact match on full ihk_branch_id against the scheme lookup
    df = df.merge(scheme, left_on="ihk_branch_id", right_index=True, how="left")
    df["is_tourism"] = df["is_tourism"].fillna(False)
    df["wz_scheme"] = scheme_name

    # diagnostic: gastro/beherbergung codes (55/56) present in data but NOT in the scheme
    is_5556 = df["ihk_branch_id"].str.startswith(("55", "56"), na=False)
    matched = df["ihk_branch_id"].isin(scheme.index)
    n_unmapped = int((is_5556 & ~matched).sum())

    diags = {
        "source_file": path.name,
        "month": month_from_filename(path.name),
        "wz_scheme": scheme_name,
        "rows": len(df),
        "n_aufwertung": int((df["gastro_class"] == "aufwertung").sum()),
        "n_gesamtgastro": int((df["gastro_class"] == "gesamtgastro").sum()),
        "n_tourism": int(df["is_tourism"].sum()),
        "gastro_unmapped": n_unmapped,
    }
    return df, diags


# %% Run
def run() -> pd.DataFrame:
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

    wz2008 = load_scheme(WZ2008_PATH)
    wz2025 = load_scheme(WZ2025_PATH)

    files = sorted(INTERMEDIATE_DIR.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files {FILE_PATTERN!r} in {INTERMEDIATE_DIR}")

    meta_records = []
    for path in files:
        month = month_from_filename(path.name)
        if month < SCHEME_SWITCH:
            scheme, name = wz2008, "WZ2008"
        else:
            scheme, name = wz2025, "WZ2025"

        df, diags = enrich_file(path, scheme, name)
        df.to_csv(ENRICHED_DIR / path.name, index=False)
        meta_records.append(diags)
        print(f"{path.name}: {name} | aufw={diags['n_aufwertung']} "
              f"gesamt={diags['n_gesamtgastro']} tour={diags['n_tourism']} "
              f"unmapped={diags['gastro_unmapped']}")

    meta = pd.DataFrame(meta_records)
    meta.to_csv(ENRICHED_DIR / "stage3_enrich_meta.csv", index=False)
    print(f"\nMeta written: {ENRICHED_DIR / 'stage3_enrich_meta.csv'}  ({len(meta)} rows)")
    return meta


if __name__ == "__main__":
    meta = run()
# %%