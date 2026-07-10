"""
Display-only labels for KiezKeeper features.

The keys are the ACTUAL dataset_0 column names and must never be renamed —
they are the shared keys across all notebooks (feature_cols, LOG_COLS, the
re_/soc_/com_ prefix parsing, the leakage guard). pretty() only produces
human-readable strings for plots and printed tables; never feed its output
back into code that expects a real column name.

Usage :
    from names import pretty
    coef_table["name"] = coef_table["feature"].apply(pretty)
"""

PRETTY_NAMES = {
    # --- real estate (re_) ---
    "re_miete_niveau":    "Rent level",
    "re_miete_trend":     "Rent trend (Δ)",
    "re_leerstandsquote": "Vacancy rate",
    "re_brw_niveau":      "Land value (level)",
    "re_brw_trend":       "Land value (Δ)",
    "re_dichte_all":      "Airbnb density",
    "re_altbau_share":    "Pre-1919 building share",
    "re_neubau_share":    "New-build share",

    # --- social (soc_) ---
    "soc_young_to_middle_adult_share_2025":                        "Young-to-mid adult share",
    "soc_young_to_middle_adult_share_change_2021_2025":            "Young-to-mid adult share (Δ)",
    "soc_single_person_hh_share_change_2021_2024":                 "Single-person HH share (Δ)",
    "soc_average_household_size_2024":                             "Avg household size",
    "soc_average_household_size_change_2021_2024":                 "Avg household size (Δ)",
    "soc_households_without_minor_children_share_change_2021_2024":"HH without minors share (Δ)",
    "soc_internal_migration_volume_rate_2025":                     "Internal migration volume",
    "soc_internal_migration_volume_rate_change_2021_2025":         "Internal migration volume (Δ)",
    "soc_internal_net_migration_rate_2025":                        "Net internal migration",
    "soc_internal_net_migration_rate_change_2021_2025":            "Net internal migration (Δ)",
    "soc_unemployment_share_change_2020_2024":                     "Unemployment share (Δ)",
    "soc_transfer_benefit_share_2024":                             "Transfer-benefit share",
    "soc_transfer_benefit_share_change_2020_2024":                 "Transfer-benefit share (Δ)",
    "soc_single_parent_household_share_2024":                      "Single-parent HH share",
    "soc_single_parent_household_share_change_2021_2024":          "Single-parent HH share (Δ)",

    # --- commercial (com_) 
    "com_median_age_level_2026":   "Business median age",
    "com_share_solo_level_2026":   "Solo-business share",
    "com_share_10plus_level_2026": "Large-business (10+) share",
    "com_gastro_share_level_2026": "Gastro share",
    "com_upscale_share_level_2026":"Upscale-retail share",
    "com_tourism_share_level_2026":"Tourism share",
    "com_entry_rate_level_2026":   "Business entry rate",
    "com_exit_rate_level_2026":    "Business exit rate",
    "com_median_age_slope":        "Business median age (Δ)",
    "com_share_max_2y_slope":      "Very-new business share (Δ)",
    "com_share_min_20y_slope":     "20y+ business share (Δ)",
    "com_share_min_30y_slope":     "30y+ business share (Δ)",
    "com_share_min_40y_slope":     "40y+ business share (Δ)",
    "com_share_solo_slope":        "Solo-business share (Δ)",
    "com_share_10plus_slope":      "Large-business (10+) share (Δ)",
    "com_gastro_share_slope":      "Gastro share (Δ)",
    "com_entry_rate_slope":        "Business entry rate (Δ)",
    "com_exit_rate_slope":         "Business exit rate (Δ)",
    "com_n_gastro":                "Number of eateries",
}


def pretty(f):
    """Readable label for display only. Falls back to stripping the
    re_/soc_/com_ prefix and replacing underscores if the feature is
    not in PRETTY_NAMES."""
    if f in PRETTY_NAMES:
        return PRETTY_NAMES[f]
    for p in ("soc_", "re_", "com_"):
        if f.startswith(p):
            f = f[len(p):]
    return f.replace("_", " ")