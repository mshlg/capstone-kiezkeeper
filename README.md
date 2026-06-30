# Capstone - KiezKeeper

KiezKeeper analyzes commercial gentrification across Berlin at the level of its
542 Planungsräume (PLR), the city's fine-grained neighborhood units. Rather than
reducing gentrification to rising rents, the project approaches it through three
complementary dimensions — real-estate, social, and commercial — each built from
its own data sources and indicators.  **INSERT MODELING + RAG DESCRIPTION HERE**

## Requirements:

- pyenv with Python: 3.11.3

### Setup

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
## Data

All data used in this project is openly accessible (links provided below). The final dataset used for modeling can be found in the folder **final_datasets** `dataset_0.csv`. A description of all variables and their meaning can be found under **INSERT FILE HERE** The data extraction process for each dimension is provided below, along with a short overview of the merge and cleaning process of the final dataset. 

### Extraction
#### Real-estate dimension
#### Social dimension
The social dimension is based on several datasets from the [Amt für Statistik Berlin-Brandenburg](https://www.statistik-berlin-brandenburg.de/) — including population data (*Einwohnerbestand*), median income data (*Medianeinkommen*), population fluctuation data (*Einwohnerbewegung*), and household data (*Privathaushalte*) — as well as data from the [Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen](https://www.berlin.de/sen/stadt/stadtdaten/stadtwissen/monitoring-soziale-stadtentwicklung/) (*Monitoring Soziale Stadtentwicklung*, MSS context and index files).

All data was collected in June 2026. Some datasets for previous years were obtained by sending personal requests via email to the two agencies mentioned above.

The unchanged raw datasets can be found in [`data/social/raw`](./data/social/raw), including a [file explaining the variables](./data/social/raw/explanation_variables_raw_datasets_socialD.md) contained in the raw datasets.

##### Cleaning, merging, and final dataset pipeline

1. All raw datasets were cleaned in the [`social_data_cleaning.ipynb` notebook](./notebooks/data_prep/social_data_cleaning.ipynb). PLR IDs were corrected, date columns were converted to pandas datetime format, numerical values were formatted consistently, and missing-value symbols were removed. Afterwards, all datasets were merged into one dataset, and duplicate and empty rows were dropped.
   The resulting dataset was saved as [`social_panel_clean.csv`](./data/social/raw/social_panel_clean.csv) in the `data/social/raw` directory.

2. In the [`social_data_final_features.ipynb` notebook](./notebooks/data_prep/social_data_final_features.ipynb), only the features relevant for modeling were extracted from the `social_panel_clean.csv` dataset. The resulting dataset was saved as [`final_social_features.csv`](./data/social/final_variables_for_EDA/final_social_features.csv).

3. In the [`social_dimension_EDA.ipynb` notebook](./notebooks/EDA/social_dimension_EDA.ipynb), the final features of the social dimension were analyzed. Due to correlations above 0.8, two features were dropped from the final dataset: average household size and unemployment share. This feature removal was implemented as the final step in the [`social_data_final_features.ipynb` notebook](./notebooks/data_prep/social_data_final_features.ipynb).


#### Commercial dimension
The commercial dimension is built from 36 monthly IHK Berlin business-register (https://cloud.ihk.berlin/d/62fad06b540745098ab7/; accessed on June 15th, 2026)
files through a seven-stage pipeline (`py_scripts/ihk_*`), producing one row per Planungsraum (PLR).

Data extraction and aggregation pipeline
1. **Stage 1 — Cleaning (`ihk_clean_stage1`):** Standardizes business and PLR
identifiers to a canonical 8-digit string form and rebuilds the district key
deterministically from the PLR prefix. A row is dropped only if it lacks a
business ID or a PLR ID; all other missingness is recorded, not removed
2. **Churn extraction (`ihk_extract_churn_stage2`):** Builds a PLR × month
panel of business stock and flows (entries, exits, moves between PLR) and derives
entry, exit and turnover rates using the average stock as denominator. The first
month has no predecessor, so its churn values are left as NaN.
3. **Stage 3 — Enrichment (`ihk_enrich_stage3`):** Adds branch-classification labels
(gastronomy type, tourism flag) to every business, switching from the WZ2008 to
the WZ2025 scheme at June 2025. The classification documents can be found under *data/IHK_Berlin_Gewerbedaten/classification_docs*. No rows are removed — filtering happens later, per
metric, at aggregation. 
4. **Stage 4 — Monthly aggregation (`ihk_aggregate_monthly_stage4`):** Collapses the
business-level data into PLR × month panels, including shares and location quotients. Where a
denominator is zero or undefined, the value is set to NaN — never to zero and
never imputed.
5. **Stage 5 — Yearly aggregation - Levels (`ihk_aggregate_yearly_stage5`).** Aggregates the monthly
metrics to calendar years using type-specific rules: stock counts as average
stock, shares from summed numerators and denominators, and churn rates from yearly
sums over the yearly average stock. This yields the *level* value of each
variable (the 2026 yearly state used in the feature matrix).
6. **Stage 6 — Yearly aggregation - Slopes (`ihk_yearly_slopes_stage6`).** For each PLR and variable, fits a
linear regression over the 36 monthly observations and reports the **slope**, with
the monthly coefficient multiplied by 12 to express an annualized trend. NaN months
are skipped and a slope is only computed when at least 12 valid months exist;
the fit quality (R²) is stored alongside for interpretation.
7. **Stage 7 — Feature matrix (`ihk_final-merge_stage7`).** Joins the 2026 levels and
the slopes into one table keyed on `plr_id`, prefixes every feature with `com_` for
cross-dimension merging, and adds the gastronomy-size context columns. Remaining
missing values (mainly upscale shares in PLR with very little gastronomy) are
preserved for the modelling step to handle.

### Dimension merge
The three dimension-level feature matrices (commercial,
real-estate, social) are joined on the shared `plr_id` key into a single
PLR-level dataset, with each variable carrying a dimension prefix (`com_`, `re_`,
`soc_`) so its origin stays unambiguous.The final feature matrix `feature_matrix_all_dimensions.csv` can be found in the final_datasets folder.

### Cleaning 

## Modeling 

### Module 1: 

### Module 2:

