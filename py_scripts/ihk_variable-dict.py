# %% [markdown]
# # Feature dictionary builder
#
# Documents the 60 feature columns of the final model-ready table
# (data/final_datasets/feature_matrix_commercial.csv) in two formats:
#   * feature_dictionary_commercial.xlsx   (formatted, colour-coded by dimension)
#   * feature_dictionary_commercial.md     (for the thesis / README)
#
# Each of the 19 modelling variables contributes three columns:
#   <var>_level_2026  — the 2026 yearly value (current state / level)
#   <var>_slope       — annualised monthly trend over the 36 months (per year, x12)
#   <var>_r2          — R^2 of that linear trend (trend quality; NaN if flat series)
# plus three gastronomy-size context columns (n_gastro, n_upscale, has_gastro_structure).
#
# The "N missing" / "% missing" columns are DATASET-SPECIFIC (computed from the current
# feature_matrix.csv), not definitional — they describe this particular run.

# %% Imports & configuration
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

try:
    REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    REPO_ROOT = Path.cwd()

FINAL_DIR = REPO_ROOT / "data/final_datasets"
MATRIX_PATH = FINAL_DIR / "feature_matrix_commercial.csv"
XLSX_OUT = FINAL_DIR / "feature_dictionary_commercial.xlsx"
MD_OUT = FINAL_DIR / "feature_dictionary_commercial.md"

LEVEL_YEAR = "2026"
PREFIX = "com_"          # feature columns in the matrix carry this prefix
KEY_OUT = "plr_id"       # key column name in the matrix

# --- the 19 modelling variables: (name, dimension, description) ---
# dimension is used for grouping + colour coding
VARS = [
    # name, dimension, human description of WHAT the base variable measures
    ("median_age",             "Demografie", "Median-Alter der Betriebe im PLR (Jahre)"),
    ("share_max_1y",           "Demografie", "Anteil Betriebe ≤ 1 Jahr alt (junge Betriebe)"),
    ("share_max_2y",           "Demografie", "Anteil Betriebe ≤ 2 Jahre alt"),
    ("share_max_5y",           "Demografie", "Anteil Betriebe ≤ 5 Jahre alt"),
    ("share_min_20y",          "Demografie", "Anteil Betriebe ≥ 20 Jahre alt (alteingesessen)"),
    ("share_min_30y",          "Demografie", "Anteil Betriebe ≥ 30 Jahre alt"),
    ("share_min_40y",          "Demografie", "Anteil Betriebe ≥ 40 Jahre alt"),
    ("share_solo",             "Demografie", "Anteil Soloselbstständiger (0 Beschäftigte) an Betrieben bekannter Größe"),
    ("share_1_9",              "Demografie", "Anteil Kleinbetriebe (1–9 Beschäftigte) an Betrieben bekannter Größe"),
    ("share_10plus",           "Demografie", "Anteil größerer Betriebe (≥ 10 Beschäftigte) an Betrieben bekannter Größe"),
    ("gastro_share",           "Gastronomie", "Anteil Gastronomie am gesamten Gewerbe des PLR"),
    ("upscale_share",          "Gastronomie", "Anteil Aufwertungsgastronomie an der Gastronomie des PLR"),
    ("gastro_quotient_bezirk", "Gastronomie", "gastro_share des PLR relativ zum Bezirksdurchschnitt (>1 = überdurchschnittlich)"),
    ("gastro_quotient_berlin", "Gastronomie", "gastro_share des PLR relativ zum Berlin-Durchschnitt"),
    ("upscale_quotient_bezirk","Gastronomie", "upscale_share des PLR relativ zum Bezirksdurchschnitt"),
    ("upscale_quotient_berlin","Gastronomie", "upscale_share des PLR relativ zum Berlin-Durchschnitt"),
    ("tourism_share",          "Tourismus", "Anteil Beherbergung am gesamten Gewerbe des PLR"),
    ("tourism_quotient_bezirk","Tourismus", "tourism_share des PLR relativ zum Bezirksdurchschnitt"),
    ("tourism_quotient_berlin","Tourismus", "tourism_share des PLR relativ zum Berlin-Durchschnitt"),
]

# how each of the three derived columns per variable is computed
TYPE_INFO = {
    "level_2026": (
        "Niveau (2026)",
        "Jahreswert 2026: aus den Monatssummen Jan–Jun 2026 gebildet "
        "(Anteile = Σ Zähler / Σ Nenner; median_age aus allen Betriebs-Beobachtungen "
        "des Jahres; Quotienten aus den Jahres-Anteilen von PLR/Bezirk/Berlin).",
    ),
    "slope": (
        "Trend (pro Jahr)",
        "Steigung einer linearen Regression über die 36 Monatswerte (Jul 2023–Jun 2026), "
        "monatliche Steigung × 12 → pro Jahr. NaN-Monate übersprungen; nur berechnet, wenn "
        "≥ 12 gültige Monate vorliegen, sonst NaN.",
    ),
    "r2": (
        "Trendgüte (R²)",
        "Bestimmtheitsmaß der linearen Trend-Regression (0–1). Hoch = geradliniger Trend, "
        "niedrig = verrauscht/nicht-linear. NaN, wenn die Reihe flach ist (keine Varianz) "
        "oder < 12 gültige Monate.",
    ),
}

# the three size/structure context columns: (name, dimension, type label, description, computation)
SIZE_COLS = [
    ("n_gastro", "Struktur", "Zählung (2026)",
     "Durchschnittliche Zahl der Gastronomiebetriebe im PLR (2026, roh).",
     "Average stock: Σ Monatsbestände Jan–Jun 2026 / Anzahl Monate. Roh belassen "
     "(keine Transformation; Log/Skalierung erst im Modell-Preprocessing)."),
    ("n_upscale", "Struktur", "Zählung (2026)",
     "Durchschnittliche Zahl der Aufwertungsgastronomie-Betriebe im PLR (2026, roh).",
     "Average stock: Σ Monatsbestände Jan–Jun 2026 / Anzahl Monate. Roh belassen."),
    ("has_gastro_structure", "Struktur", "Flag (binär)",
     "1, wenn der PLR 2026 eine gastronomische Mindeststruktur hat (n_gastro ≥ 5), sonst 0.",
     "Binär aus n_gastro (2026): 1 wenn ≥ 5, sonst 0. Markiert PLRs, in denen "
     "upscale-Anteile auf ausreichender Fallzahl beruhen."),
]

DIM_COLOR = {
    "Demografie":  "DDEBF7",   # light blue
    "Gastronomie": "FCE4D6",   # light orange
    "Tourismus":   "E2EFDA",   # light green
    "Struktur":    "FFF2CC",   # light yellow
}


# %% Build the dictionary rows
def build_rows(matrix: pd.DataFrame) -> list[dict]:
    n_plr = len(matrix)
    rows = []

    def na_stats(col):
        if col not in matrix.columns:
            return ("—", "—")
        n_missing = int(matrix[col].isna().sum())
        pct = f"{n_missing / n_plr * 100:.1f}%"
        return (n_missing, pct)

    # 19 variables × 3 derived columns (columns in the matrix carry the com_ prefix)
    for name, dim, desc in VARS:
        for suffix in ("level_2026", "slope", "r2"):
            col = f"{PREFIX}{name}_{suffix}"
            type_label, comp = TYPE_INFO[suffix]
            n_missing, pct = na_stats(col)
            rows.append({
                "Spalte": col,
                "Basis-Variable": name,
                "Dimension": dim,
                "Typ": type_label,
                "Beschreibung": desc,
                "Berechnung": comp,
                "N fehlend (2026-Lauf)": n_missing,
                "% fehlend": pct,
            })

    # 3 size/structure columns (also prefixed in the matrix)
    for name, dim, type_label, desc, comp in SIZE_COLS:
        col = f"{PREFIX}{name}"
        n_missing, pct = na_stats(col)
        rows.append({
            "Spalte": col,
            "Basis-Variable": name,
            "Dimension": dim,
            "Typ": type_label,
            "Beschreibung": desc,
            "Berechnung": comp,
            "N fehlend (2026-Lauf)": n_missing,
            "% fehlend": pct,
        })
    return rows


# %% Markdown writer
def write_markdown(rows: list[dict], n_plr: int) -> None:
    cols = ["Spalte", "Dimension", "Typ", "Beschreibung", "Berechnung",
            "N fehlend (2026-Lauf)", "% fehlend"]
    lines = []
    lines.append("# Feature-Dictionary — finale Modell-Tabelle")
    lines.append("")
    lines.append(f"Datei: `data/final_datasets/feature_matrix_commercial.csv`  ·  "
                 f"{n_plr} PLRs × {len(rows)} Feature-Spalten (+ planungsraum_id)")
    lines.append("")
    lines.append("Jede der 19 Modell-Variablen liefert drei Spalten: **Niveau (2026)**, "
                 "**Trend (Slope, pro Jahr)** und **Trendgüte (R²)**. Dazu kommen drei "
                 "Struktur-Spalten zur Gastronomie-Größe.")
    lines.append("")
    lines.append("> Die Spalten *N fehlend* / *% fehlend* sind datensatz-spezifisch "
                 "(dieser 2026-Lauf), keine Definitionsmerkmale.")
    lines.append("")
    # group by dimension for readability
    for dim in ["Demografie", "Gastronomie", "Tourismus", "Struktur"]:
        dim_rows = [r for r in rows if r["Dimension"] == dim]
        if not dim_rows:
            continue
        lines.append(f"## {dim}")
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in dim_rows:
            vals = [str(r[c]).replace("|", "\\|") for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  written: {MD_OUT}")


# %% Excel writer
def write_excel(rows: list[dict], n_plr: int) -> None:
    cols = ["Spalte", "Basis-Variable", "Dimension", "Typ", "Beschreibung",
            "Berechnung", "N fehlend (2026-Lauf)", "% fehlend"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Feature-Dictionary"

    title_font = Font(name="Arial", bold=True, size=13)
    head_font = Font(name="Arial", bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", start_color="4472C4")
    base_font = Font(name="Arial", size=10)
    wrap = Alignment(vertical="top", wrap_text=True)
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # title
    ws["A1"] = "Feature-Dictionary — finale Modell-Tabelle (feature_matrix.csv)"
    ws["A1"].font = title_font
    ws["A2"] = (f"{n_plr} PLRs × {len(rows)} Feature-Spalten (+ planungsraum_id). "
                "N/% fehlend = dieser 2026-Lauf (datensatz-spezifisch).")
    ws["A2"].font = Font(name="Arial", size=9, italic=True, color="808080")

    header_row = 4
    for j, c in enumerate(cols, start=1):
        cell = ws.cell(row=header_row, column=j, value=c)
        cell.font = head_font
        cell.fill = head_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = border

    for i, r in enumerate(rows, start=header_row + 1):
        for j, c in enumerate(cols, start=1):
            cell = ws.cell(row=i, column=j, value=r[c])
            cell.font = base_font
            cell.alignment = wrap
            cell.border = border
            cell.fill = PatternFill("solid", start_color=DIM_COLOR.get(r["Dimension"], "FFFFFF"))

    widths = {"A": 28, "B": 22, "C": 13, "D": 16, "E": 46, "F": 60, "G": 18, "H": 11}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A5"
    ws.row_dimensions[header_row].height = 28

    wb.save(XLSX_OUT)
    print(f"  written: {XLSX_OUT}")


# %% Run
def run() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"feature_matrix not found: {MATRIX_PATH} (run Stage 7 first)")
    matrix = pd.read_csv(MATRIX_PATH, dtype={"planungsraum_id": str})
    n_plr = len(matrix)
    rows = build_rows(matrix)
    print(f"feature_matrix: {n_plr} PLRs, {matrix.shape[1]} columns | documenting {len(rows)} features")
    write_excel(rows, n_plr)
    write_markdown(rows, n_plr)


if __name__ == "__main__":
    run()
# %%