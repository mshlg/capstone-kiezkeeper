# Feature Dictionary — Combined Modeling Matrix (all dimensions)

File: `dataset_0.csv` · 527 PLRs × 47 columns

The matrix combines three dimensions: **Real estate** · **Social** · **Commercial** 

Only the commercial variables are documented below; the real-estate and social rows are left blank as a template to be completed by their respective owners.

Commercial design note: each underlying business indicator contributes a **Level (2026)** and a **Trend (slope, per year)**; two churn rates (entry/exit) and one structural count complete the set. The R², quotient and binary-flag columns from the standalone commercial EDA were dropped for this reduced modeling matrix.

---

## Identifiers (N = 3)

| Variable | Type | Definition | Derivation | Source | File format | Spatial level |
|---|---|---|---|---|---|---|
| plr_id | ID | ID of PLR | n.a. | see dimensions | CSV | PLR |
| plr_name | ID | Name of PLR | n.a. | see dimensions | CSV | PLR |
| bez | ID | Name of Bezirk | n.a. | see dimensions | CSV | PLR |

## Real estate (N = 9)

| Variable | Type | Definition | Derivation | Source | File format | Spatial level |
|---|---|---|---|---|---|---|
| re_miete_niveau |  |  |  |  |  |  |
| re_miete_trend |  |  |  |  |  |  |
| re_leerstandsquote |  |  |  |  |  |  |
| re_brw_niveau |  |  |  |  |  |  |
| re_brw_trend |  |  |  |  |  |  |
| re_cov_wohnen_2025 |  |  |  |  |  |  |
| re_dichte_all |  |  |  |  |  |  |
| re_altbau_share |  |  |  |  |  |  |
| re_neubau_share |  |  |  |  |  |  |

## Social (N = 14)

| Variable | Type | Definition | Derivation | Source | File format | Spatial level |
|---|---|---|---|---|---|---|
| soc_young_to_middle_adult_share_2025 |  |  |  |  |  |  |
| soc_young_to_middle_adult_share_change_2021_2025 |  |  |  |  |  |  |
| soc_single_person_hh_share_2024 |  |  |  |  |  |  |
| soc_single_person_hh_share_change_2021_2024 |  |  |  |  |  |  |
| soc_households_without_minor_children_share_2024 |  |  |  |  |  |  |
| soc_households_without_minor_children_share_change_2021_2024 |  |  |  |  |  |  |
| soc_internal_migration_volume_rate_2025 |  |  |  |  |  |  |
| soc_internal_migration_volume_rate_change_2021_2025 |  |  |  |  |  |  |
| soc_internal_net_migration_rate_2025 |  |  |  |  |  |  |
| soc_internal_net_migration_rate_change_2021_2025 |  |  |  |  |  |  |
| soc_transfer_benefit_share_2024 |  |  |  |  |  |  |
| soc_transfer_benefit_share_change_2020_2024 |  |  |  |  |  |  |
| soc_single_parent_household_share_2024 |  |  |  |  |  |  |
| soc_single_parent_household_share_change_2021_2024 |  |  |  |  |  |  |
## Commercial (N = 21)

| Variable | Type | Definition | Derivation | Source | File format | Spatial level |
|---|---|---|---|---|---|---|
| com_median_age_level_2026 | Level (2026) | Median age of businesses in the PLR (years). | 2026 annual value built from the monthly figures Jan–Jun 2026 (shares = Σ numerator / Σ denominator; median age from all business observations of the year). | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_solo_level_2026 | Level (2026) | Share of solo self-employed (0 employees) among businesses of known size. | 2026 annual value built from the monthly figures Jan–Jun 2026 (share = Σ numerator / Σ denominator). | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_10plus_level_2026 | Level (2026) | Share of larger businesses (≥ 10 employees) among businesses of known size. | 2026 annual value built from the monthly figures Jan–Jun 2026 (share = Σ numerator / Σ denominator). | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_gastro_share_level_2026 | Level (2026) | Share of food service (gastronomy) in the PLR's total commerce. | 2026 annual value built from the monthly figures Jan–Jun 2026 (share = Σ numerator / Σ denominator). | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_upscale_share_level_2026 | Level (2026) | Share of upscale (upgrading) gastronomy within the PLR's gastronomy. | 2026 annual value built from the monthly figures Jan–Jun 2026 (share = Σ numerator / Σ denominator). | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_tourism_share_level_2026 | Level (2026) | Share of accommodation/lodging in the PLR's total commerce. | 2026 annual value built from the monthly figures Jan–Jun 2026 (share = Σ numerator / Σ denominator). | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_entry_rate_level_2026 | Level (2026) | Business entry rate: new registrations during the year relative to the average business stock in the PLR. | 2026 annual value from the monthly registry (Jan–Jun 2026): entries accumulated over the year / average business stock. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_exit_rate_level_2026 | Level (2026) | Business exit rate: deregistrations during the year relative to the average business stock in the PLR. | 2026 annual value from the monthly registry (Jan–Jun 2026): exits accumulated over the year / average business stock. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_median_age_slope | Trend (per year, Jul 2023–Jun 2026) | Median age of businesses in the PLR (years). | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_max_2y_slope | Trend (per year, Jul 2023–Jun 2026) | Share of businesses ≤ 2 years old. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_min_20y_slope | Trend (per year, Jul 2023–Jun 2026) | Share of businesses ≥ 20 years old (long-established). | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_min_30y_slope | Trend (per year, Jul 2023–Jun 2026) | Share of businesses ≥ 30 years old. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_min_40y_slope | Trend (per year, Jul 2023–Jun 2026) | Share of businesses ≥ 40 years old. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_solo_slope | Trend (per year, Jul 2023–Jun 2026) | Share of solo self-employed (0 employees) among businesses of known size. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_share_10plus_slope | Trend (per year, Jul 2023–Jun 2026) | Share of larger businesses (≥ 10 employees) among businesses of known size. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_gastro_share_slope | Trend (per year, Jul 2023–Jun 2026) | Share of food service (gastronomy) in the PLR's total commerce. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_upscale_share_slope | Trend (per year, Jul 2023–Jun 2026) | Share of upscale (upgrading) gastronomy within the PLR's gastronomy. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_tourism_share_slope | Trend (per year, Jul 2023–Jun 2026) | Share of accommodation/lodging in the PLR's total commerce. | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_entry_rate_slope | Trend (per year, Jul 2023–Jun 2026) | Business entry rate (new registrations relative to stock). | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_exit_rate_slope | Trend (per year, Jul 2023–Jun 2026) | Business exit rate (deregistrations relative to stock). | Slope of a linear regression over the 36 monthly values (Jul 2023–Jun 2026), monthly slope × 12 → per year. NaN months skipped; computed only with ≥ 12 valid months, otherwise NaN. | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
| com_n_gastro | Count (2026) | Average number of gastronomy businesses in the PLR (2026, raw). | Average stock: Σ monthly stocks Jan–Jun 2026 / number of months. Left raw (no transformation; log/scaling only in model preprocessing). | IHK Berlin Gewerbedaten (monthly) | CSV | PLR |
