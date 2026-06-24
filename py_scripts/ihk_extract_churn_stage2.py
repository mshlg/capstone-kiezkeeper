# %% [markdown]
# # Stage 2 — Churn Extraction  (intermediate_data -> PLR × Month-Panel)
#
# Reads the cleaned data from stage 1 and builds a PLR x Month Panel (one row per PLR and month)
#
# Calculations for each PLR and month:
#   * Business stock:   n_prev (t-1), n_curr (t)
#   * Churn: entries, exits, movers_in, movers_out  (raw)
#   * Churn Rates (denominator = average (n_prev + n_curr) / 2):
#       - churn_rate            = (entries + exits) / denominator (including moves from one to another PLR)
#       - churn_rate_exclusive  = (real new businesses + closings) / denominator
#       - entry_rate, exit_rate
#
# First month (2023-07) does not have t-1 -> Churn & Rates = NaN, only n_curr will be filled.
# IMPORTANT: opendata_id and planungsraum_id are read as dtype = str,
# so that planungsraum_id retains its leading zero ("01100312").

# %% Imports & configuration
from pathlib import Path
import pandas as pd

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # interactive cells: no __file__
    REPO_ROOT = Path.cwd()

BASE_DIR = REPO_ROOT / "data/IHK_Berlin_Gewerbedaten"
INTERMEDIATE_DIR = BASE_DIR / "intermediate_data"
ANALYSIS_DIR = BASE_DIR / "analysis_ready_data"
PANEL_PATH = ANALYSIS_DIR / "churn_panel.csv"

FILE_PATTERN = "*_IHK_Berlin_Gewerbedaten.csv"

# IDs MUST be read as strings -> "01100312" is preserved (don't lose the leading zero).
ID_DTYPES = {"opendata_id": str, "planungsraum_id": str}

MOVE_COLS = ["entries", "exits", "movers_in", "movers_out"]


# %% Helpers
def month_from_filename(name: str) -> str:
    """'2023_07_IHK_Berlin_Gewerbedaten.csv' -> '2023-07'."""
    return pd.to_datetime(name[:7], format="%Y_%m").strftime("%Y-%m")


def load_month(path: Path):
    """Read one cleaned file as a unique id -> planungsraum_id map for the month.
    IDs stay strings. Duplicate ids within a month are reduced to one (first)."""
    df = pd.read_csv(path, usecols=["opendata_id", "planungsraum_id"], dtype=ID_DTYPES)
    df = df.dropna(subset=["opendata_id", "planungsraum_id"])
    n_dup = int(df["opendata_id"].duplicated().sum())
    df = df.drop_duplicates("opendata_id", keep="first")
    return df, n_dup


def churn_between(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
    """Movement building blocks per PLR between two consecutive months.
    Expects unique opendata_id + planungsraum_id per frame. Indexed by PLR."""
    # outer merge on opendata_id -> one row per business with its PLR last month (plr_prev)
    # and this month (plr_curr); NaN where the business is absent in that month
    m = prev_df.rename(columns={"planungsraum_id": "plr_prev"}).merge(
        curr_df.rename(columns={"planungsraum_id": "plr_curr"}),
        on="opendata_id", how="outer",
    )
    pp = m["plr_prev"]  # PLR last month (NaN if business didn't exist then)
    pc = m["plr_curr"]  # PLR this month (NaN if business gone now)
    present_prev = pp.notna()  # was the business here last month?
    present_curr = pc.notna()  # is it here this month?
    both = present_prev & present_curr  # existed in both months
    # moved = present in both months but in a different PLR (fillna avoids NA in comparisons)
    moved = both & (pp.fillna("") != pc.fillna(""))

    is_entry = (present_curr & (~present_prev | moved)).fillna(False)      # entry into pc (new OR moved in)
    is_exit = (present_prev & (~present_curr | moved)).fillna(False)       # exit from pp (closed OR moved out)
    is_mover_in = (present_curr & moved).fillna(False)                     # entry into pc, came from another PLR
    is_mover_out = (present_prev & moved).fillna(False)                    # exit from pp, went to another PLR

    out = pd.DataFrame({
        "entries": pc[is_entry].value_counts(),        # count entries per destination PLR (plr_curr)
        "exits": pp[is_exit].value_counts(),           # count exits per origin PLR (plr_prev)
        "movers_in": pc[is_mover_in].value_counts(),   # of the entries, how many were moves in
        "movers_out": pp[is_mover_out].value_counts(), # of the exits, how many were moves out
    })
    return out  # per-PLR counts; missing PLR in a category -> NaN (filled to 0 in run())


# %% Run
def run() -> pd.DataFrame:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)  # output folder, create if missing

    files = sorted(INTERMEDIATE_DIR.glob(FILE_PATTERN))  # cleaned files, chronological
    if not files:  # wrong path -> fail early with a clear message
        raise FileNotFoundError(f"No files {FILE_PATTERN!r} in {INTERMEDIATE_DIR}")

    months = [month_from_filename(f.name) for f in files]  # -> ["2023-07", ...]

    # load all months once + collect the universe of all PLRs
    maps, dup_total = {}, 0  # maps: month -> (id, PLR) table ; dup_total: dedup safety counter
    all_plrs = set()
    for f, mth in zip(files, months):
        df, n_dup = load_month(f)
        maps[mth] = df  # keep each month in memory (needed for stock AND pair comparison)
        dup_total += n_dup
        all_plrs.update(df["planungsraum_id"].unique())  # union of all PLRs across months

    # stock n_curr per PLR × month (authoritative, from the files)
    n_curr_parts = []
    for mth in months:
        s = maps[mth]["planungsraum_id"].value_counts().rename("n_curr")  # businesses per PLR
        part = s.reset_index().rename(columns={"index": "planungsraum_id"})
        part["month"] = mth
        n_curr_parts.append(part)
    panel = pd.concat(n_curr_parts, ignore_index=True)  # long: one row per (PLR, month) that had businesses

    # balanced grid: every PLR in every month
    grid = pd.MultiIndex.from_product(
        [sorted(all_plrs), months], names=["planungsraum_id", "month"]
    )  # every PLR × month combination, including the empty ones
    panel = (panel.set_index(["planungsraum_id", "month"])
                  .reindex(grid)       # force panel onto the full grid; missing combos -> NaN
                  .reset_index())
    panel["n_curr"] = panel["n_curr"].fillna(0).astype("Int64")  # missing from value_counts = 0 businesses

    # n_prev = previous month's n_curr (within PLR, shifted chronologically)
    month_order = {m: i for i, m in enumerate(months)}  # month -> position 0..35 (chronological sort key)
    panel = panel.sort_values(
        ["planungsraum_id", "month"], key=lambda c: c.map(month_order) if c.name == "month" else c
    )
    panel["n_prev"] = panel.groupby("planungsraum_id")["n_curr"].shift(1)  # first month per PLR -> <NA>

    # movements per consecutive month pair
    move_parts = []
    for i in range(1, len(months)):  # start at 1: the first month has no previous month
        prev_df, curr_df = maps[months[i - 1]], maps[months[i]]
        mv = churn_between(prev_df, curr_df).reset_index().rename(columns={"index": "planungsraum_id"})
        mv["month"] = months[i]  # attribute movements to the CURRENT month
        move_parts.append(mv)
    moves = pd.concat(move_parts, ignore_index=True)

    panel = panel.merge(moves, on=["planungsraum_id", "month"], how="left")  # left: keep the full grid

    # movement columns: keep NaN in the first month, otherwise missing = 0
    not_first = panel["month"] != months[0]  # mask: everything except the first month
    for c in MOVE_COLS:
        panel.loc[not_first, c] = panel.loc[not_first, c].fillna(0)  # no movement = 0 (but NaN in first month)
        panel[c] = panel[c].astype("Int64")

    # rates (denominator = average; n_prev NaN in first month -> rates NaN; denominator 0 -> NaN)
    denom = (panel["n_prev"] + panel["n_curr"]) / 2
    denom = denom.where(denom > 0)  # 0 or NaN denominator -> NaN (no division by zero / no first month)
    true_new = panel["entries"] - panel["movers_in"]       # real new businesses (entries minus movers)
    true_closed = panel["exits"] - panel["movers_out"]     # real closings (exits minus movers)
    panel["entry_rate"] = panel["entries"] / denom
    panel["exit_rate"] = panel["exits"] / denom
    panel["churn_rate"] = (panel["entries"] + panel["exits"]) / denom               # incl. moves
    panel["churn_rate_exclusive"] = (true_new + true_closed) / denom                # excl. moves

    cols = (["planungsraum_id", "month", "n_prev", "n_curr"]  # readable column order
            + MOVE_COLS
            + ["entry_rate", "exit_rate", "churn_rate", "churn_rate_exclusive"])
    panel = panel[cols].sort_values(
        ["month", "planungsraum_id"], key=lambda c: c.map(month_order) if c.name == "month" else c
    ).reset_index(drop=True)  # final sort: month (chronological), then PLR

    panel.to_csv(PANEL_PATH, index=False)
    print(f"Panel written: {PANEL_PATH}")
    print(f"  Rows: {len(panel)}  ({len(all_plrs)} PLRs × {len(months)} months)")
    if dup_total:  # should not appear anymore: Stage 1 already deduplicated
        print(f"  Note: reduced {dup_total} duplicate opendata_id(s) across months.")
    return panel


# %% Validation (optional sanity checks — run after run(), warns instead of raising)
def validate(panel: pd.DataFrame, n_months: int) -> None:
    """Print sanity checks on the finished panel. NaN in the first month is expected,
    so checks ignore those rows rather than failing on them."""
    n_plr = panel["planungsraum_id"].nunique()
    expected = n_plr * n_months
    print(f"rows: {len(panel)}  (expected {n_plr} PLRs × {n_months} months = {expected})")

    # exclusive churn can never exceed total churn (compare only rows where rates exist)
    mask = panel["churn_rate"].notna()
    ok_excl = (panel.loc[mask, "churn_rate_exclusive"] <= panel.loc[mask, "churn_rate"] + 1e-9).all()
    print(f"churn_rate_exclusive <= churn_rate: {ok_excl}")

    # every move out of one PLR is a move into another -> global sums must match
    mo, mi = int(panel["movers_out"].sum()), int(panel["movers_in"].sum())
    print(f"movers_out total = {mo} ; movers_in total = {mi} ; equal: {mo == mi}")

    # first month has no t-1 -> all movement/rate values must be NaN
    first = sorted(panel["month"].unique())[0]
    first_nan = panel.loc[panel["month"] == first, "churn_rate"].isna().all()
    print(f"first month ({first}) churn all NaN: {first_nan}")

    # this one is a real impossibility, not an edge case -> hard assert
    assert mo == mi, "movers_out != movers_in (every move must be one out + one in)"


if __name__ == "__main__":
    panel = run()
    validate(panel, n_months=36)