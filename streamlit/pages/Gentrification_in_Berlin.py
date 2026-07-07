##################################################
#### PREREQUISITES###############################


# import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import geopandas as gpd
import json
import numpy as np
import re
import io

# set page to wide format
st.set_page_config(layout="wide")

# set title
st.logo('kiezkeeper_vector_logo.svg', size="large")
st.markdown("# KiezKeeper :small[Data-Driven Detection of Gentrification in Berlin]")
st.markdown("***")

# set style for containers
css = """
.st-key-white_container_upper,
.st-key-white_container_left,
.st-key-white_container_right,
.st-key-white_container_profile{
    background: rgba(255, 255, 255);
}
.st-key-short_profile_textbox,
.st-key-left_profile_textbox,
.st-key-middle_profile_textbox,
.st-key-right_profile_textbox{
    background: rgba(245, 244, 244);
    padding: 16px;
}
"""
st.html(f"<style>{css}</style>")

##########################################################
########## BERLIN MAP ###################################

# background style for map
grey_map_style = {
    "version": 8,
    "sources": {},
    "layers": [
        {
            "id": "background",
            "type": "background",
            "paint": {"background-color": "#f5f4f4"}
        }
    ]
}

# Calculate the zoom level needed so a bounding box always fits
# inside a given pixel width/height (prevents the map from being cropped)
def calculate_zoom(min_lon, max_lon, min_lat, max_lat, width_px, height_px, padding_factor=0.9):
    WORLD_DIM = 512  # tile size at zoom level 0 (Mapbox default)

    def lat_to_merc_y(lat):
        rad = np.radians(lat)
        return np.log(np.tan(np.pi / 4 + rad / 2))

    lon_diff = max_lon - min_lon
    lat_diff = lat_to_merc_y(max_lat) - lat_to_merc_y(min_lat)

    zoom_lon = np.log2(width_px * 360 / (lon_diff * WORLD_DIM))
    zoom_lat = np.log2(height_px * 2 * np.pi / (lat_diff * WORLD_DIM))

    zoom = min(zoom_lon, zoom_lat)
    zoom = zoom + np.log2(padding_factor)  # add a bit of padding at the edges
    return zoom


# Calculate the map height that matches Berlin's bounding-box aspect ratio
# at the target width, so top/bottom and left/right margins are both
# minimal at once (instead of guessing a fixed height)
def calculate_matching_height(min_lon, max_lon, min_lat, max_lat, width_px):
    def lat_to_merc_y(lat):
        rad = np.radians(lat)
        return np.log(np.tan(np.pi / 4 + rad / 2))

    lon_diff = max_lon - min_lon
    lat_diff = lat_to_merc_y(max_lat) - lat_to_merc_y(min_lat)

    height_px = width_px * (360 * lat_diff) / (2 * np.pi * lon_diff)
    return int(round(height_px))


# create Berlin map
# import final output dataset
df_final = pd.read_csv("../data/final_datasets/df_clusters_milieuschutz.csv")
df_final["plr_id"] = df_final["plr_id"].astype(str).str.zfill(8)

# slim copy with only the columns needed for THIS map
# FIX: "ms_binary" no longer exists in the current CSV -- the binary
# milieu-protection flag is now split into 5 threshold columns
# (ms_over50/60/70/80/90). We keep all of them here because the
# sidebar control lets the user switch between thresholds.
MS_THRESHOLD_COLUMNS = ["ms_over50", "ms_over60", "ms_over70", "ms_over80", "ms_over90"]
df_map = df_final[["plr_id", "cluster_4k", "ms_portion"] + MS_THRESHOLD_COLUMNS].copy()

# Cache the function so the geodata is not loaded again on every rerun
@st.cache_data
def load_plr_geometries():

    # Load the PLR geometries from the URL
    plr_geo = gpd.read_file("plr_geometries.gpkg")

    # Make sure the PLR ID is a string with 8 digits
    plr_geo["plr_id"] = plr_geo["plr_id"].astype(str).str.zfill(8)

    # Keep only the columns we need
    plr_geo = plr_geo[["plr_id", "plr_name", "geometry"]]

    # Make the geometries simpler, so the map loads faster
    plr_geo["geometry"] = plr_geo["geometry"].simplify(
        tolerance=3,
        preserve_topology=True
    )

    # Convert geometries to web map coordinates
    plr_geo = plr_geo.to_crs(epsg=4326)

    # Return the prepared geodata
    return plr_geo


# Load the prepared PLR geometries
plr_geo = load_plr_geometries()

# Join the geodata with the cluster / milieuschutz data
gdf = plr_geo.merge(
    df_map,
    on="plr_id",
    how="left"
)

#+++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++ CLUSTER MAP ++++++++++++++++++++++++++++

# Create readable cluster labels
cluster_labels = {
    1: "City core",
    3: "City ring",
    2: "Disadvantaged outskirts",
    0: "Affluent outskirts",
}
gdf["cluster_status"] = gdf["cluster_4k"].map(cluster_labels)
gdf.loc[gdf["cluster_4k"].isna(), "cluster_status"] = "No data available"

# Numeric code per cluster, needed for the single shared trace
# (this is what makes hover work the same for ALL areas, not just clustered ones)
gdf["cluster_code"] = gdf["cluster_4k"]
gdf.loc[gdf["cluster_4k"].isna(), "cluster_code"] = -1

# Reset the index and create a unique ID for Plotly
gdf = gdf.reset_index(drop=True)
geojson = json.loads(gdf.to_json())

# Convert the GeoDataFrame to GeoJSON for Plotly
geojson = json.loads(gdf.to_json())

# Calculate center from geometries
min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
center_lon = (min_lon + max_lon) / 2
center_lat = (min_lat + max_lat) / 2

# Assumed total browser window width for a typical laptop/desktop screen
ASSUMED_WINDOW_WIDTH_PX = 1400

# Must match the ratio you pass into st.columns([...]) below !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MAP_COLUMN_RATIO = 2 / 3

TARGET_WIDTH_PX = ASSUMED_WINDOW_WIDTH_PX * MAP_COLUMN_RATIO

# Map height derived from Berlin's aspect ratio at that width, so
# top/bottom and left/right margins are both minimal at once
MAP_HEIGHT_PX = calculate_matching_height(
    min_lon, max_lon, min_lat, max_lat,
    width_px=TARGET_WIDTH_PX
)

zoom_level = calculate_zoom(
    min_lon, max_lon, min_lat, max_lat,
    width_px=TARGET_WIDTH_PX,
    height_px=MAP_HEIGHT_PX,
    padding_factor=0.97
)

# Discrete colorscale for the 5 categories: no data + 4 clusters
# (same stepped-colorscale technique as the milieuschutz map, generalized
# to 5 evenly spaced bands instead of 3)
colorscale = [
    [0.0, "#FFFFFF"], [0.2, "#FFFFFF"],   # No data available
    [0.2, "#B8B8B8"], [0.4, "#B8B8B8"],   # Affluent outskirts (cluster 0)
    [0.4, "#8B0000"], [0.6, "#8B0000"],   # City core (cluster 1)
    [0.6, "#737373"], [0.8, "#737373"],   # Disadvantaged outskirts (cluster 2)
    [0.8, "#EE4B2B"], [1.0, "#EE4B2B"],   # City ring (cluster 3)
]

# initialize session state for map click with a default (here: first PLR in the data)
if "selected_plr_id" not in st.session_state:
    st.session_state.selected_plr_id = gdf["plr_id"].iloc[0]

# called every rerun so that gdf["ms_status"] (used later in the Short Profile section) is
# always up to date, regardless of whether the map figure itself came
# from cache or not.
def compute_ms_status(ms_column, threshold_pct):
    status = pd.Series(
        f"Proportion of milieu protected area less than {threshold_pct}% of PLR",
        index=gdf.index,
    )
    status[gdf[ms_column] == 1] = f"Proportion of milieu protected area more than {threshold_pct}% of PLR"
    status[gdf[ms_column].isna()] = "No data available"
    return status


# Builds the cluster map figure for a given milieuschutz threshold column
# (e.g. "ms_over50", "ms_over80", ...). Cached so that flipping back and
# forth between thresholds on the sidebar doesn't rebuild the whole
# Plotly figure every single time
#
# IMPORTANT: this function must not mutate the global "gdf" (e.g. via
# "gdf['ms_status'] = ..."). On a cache HIT the function body doesn't
# run at all, so any such side effect would silently go stale. That's
# why ms_status is computed as a local Series here and merged into a
# throwaway "customdata" frame instead of written back onto gdf.
@st.cache_data(show_spinner=False)
def build_cluster_map(ms_column, threshold_pct):
    ms_status = compute_ms_status(ms_column, threshold_pct)

    # Static per-row border styling for milieuschutz areas (thicker + black border)
    ms_line_widths = np.where(gdf[ms_column] == 1, 2.0, 0.4)
    ms_line_colors = np.where(gdf[ms_column] == 1, "#000000", "#ffffff")

    customdata = gdf[["plr_name", "plr_id", "cluster_status"]].copy()
    customdata["ms_status"] = ms_status

    # Create map
    # Single Choroplethmapbox trace instead of one trace per category,
    # so hover works consistently for every area
    fig_map = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=gdf["plr_id"],
            z=gdf["cluster_code"],
            zmin=-1,
            zmax=3,
            featureidkey="properties.plr_id",
            colorscale=colorscale,
            showscale=False,
            marker_opacity=0.85,
            marker_line_width=ms_line_widths,
            marker_line_color=ms_line_colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "PLR-ID: %{customdata[1]}<br>"
                "Cluster: %{customdata[2]}<br>"
                "Status: %{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )

    # Style borders and layout
    fig_map.update_layout(
        mapbox_style=grey_map_style,
        mapbox=dict(
            center={"lat": center_lat, "lon": center_lon},
            zoom=zoom_level,
            bearing=0,
            pitch=0,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=MAP_HEIGHT_PX,   # fixed height, matched to TARGET_WIDTH_PX
        autosize=True,
    )
    return fig_map

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++ SIMILARITY MAP +++++++++++++++++++++++++++++++++++



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++ SHOW MAPS +++++++++++++++++++++++++++++++++++++++++
with st.container(key="white_container_upper", border=True):
    st.markdown("##### Welcome to KiezKeeper.")
    st.markdown("KiezKeeper was developed to detect gentrificaiton in Berlin. On the sidebar, you have the option to choose between the cluster outcome and the similarity score. BLABLABLA")
    map_status = st.sidebar.radio("Please choose a map.", options=["Gentrification Profiles", "Similarity Scores"], horizontal=True)
    if map_status == "Gentrification Profiles":
        ms_proportion = st.sidebar.radio(
            label="% of total area of PLR designated for milieu protection:", 
            options=["more than 50%", "more than 60%", "more than 70%", "more than 80%", "more than 90%"]
        )
    else:
        # default so nothing breaks if "Similarity Scores" is selected.
        ms_proportion = "more than 50%"

# translate the radio label ("more than 70%") into the matching
# dataframe column ("ms_over70"), and pull out the bare number for the
# hover text, then (re)build the map figure for that threshold.
ms_threshold_columns = {
    "more than 50%": "ms_over50",
    "more than 60%": "ms_over60",
    "more than 70%": "ms_over70",
    "more than 80%": "ms_over80",
    "more than 90%": "ms_over90",
}
ms_column = ms_threshold_columns[ms_proportion]
threshold_pct = re.search(r"\d+", ms_proportion).group()

# Always keep gdf["ms_status"] current
gdf["ms_status"] = compute_ms_status(ms_column, threshold_pct)

# Cached: only actually rebuilds the figure the first time a given
# threshold is selected in this session; subsequent switches back to an
# already-seen threshold are served straight from cache.
fig_map = build_cluster_map(ms_column, threshold_pct)

# Show map on half the page
left_col, right_col = st.columns([3, 2], gap="small")

with left_col:
    with st.container(key="white_container_left", border=True):
        
        # initialize both as None, so the click-handling code below
        # can safely check "whichever one actually got clicked"
        map_event = None
        map_second_event = None

        # showing cluster map
        if map_status == "Gentrification Profiles":
            st.markdown("#### Planning areas (PLR) of Berlin -- Gentrification Profiles", text_alignment="center")
            st.markdown(body="*- Please choose a map on the sidebar -*", text_alignment="center")

            # plot map
            map_event = st.plotly_chart(
                fig_map,
                width="stretch",
                config={"responsive": True},
                on_select="rerun",
                selection_mode="points",
                key="cluster_map"
            )

            # legend -- two explicit rows: cluster colors on top,
            # "No data" + "Milieu protection" always on their own row below
            st.markdown(
                """
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.9rem; margin-top:8px;">
                <span><span style="display:inline-block;width:15px;height:15px;background:#8B0000;border-radius:2px;"></span> City core</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#EE4B2B;border-radius:2px;"></span> City belt</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#737373;border-radius:2px;"></span> Disadv. outskirts</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#B8B8B8;border-radius:2px;"></span> Affluent outskirts</span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.9rem; margin-top:8px;">
                <span><span style="display:inline-block;width:15px;height:15px;background:#ffffff;border:1px solid #999;border-radius:2px;"></span> No data</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:none;border:2px solid black;border-radius:2px;"></span> Milieu protection (&gt;""" + threshold_pct + """%)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        
        # showing similarity map
        else:
            st.markdown("#### Planning areas (PLR) of Berlin -- Similarity Scores", text_alignment="center")
            st.markdown(body="*- Please choose a map on the sidebar -*", text_alignment="center")

            # map_second_event = st.plotly_chart(
            #     fig_similarity,   # deine zweite, noch zu bauende Figure
            #     width="stretch",
            #     config={"responsive": True},
            #     on_select="rerun",
            #     selection_mode="points",
            #     key="similarity_map"
            # )

# update selection based on WHICHEVER map was actually clicked
active_event = map_event if map_event is not None else map_second_event

if active_event and active_event["selection"]["points"]:
    clicked_plr_id = active_event["selection"]["points"][0]["location"]
    st.session_state.selected_plr_id = clicked_plr_id


################################################################
##################### SHORT PROFILE #############################

# load another dataframe for resident count and other variables
plot_df = pd.read_csv("data/plot_df.csv", dtype={"plr_id": str})

# initialize state
if "show_profile" not in st.session_state:
    st.session_state.show_profile = False

# container height for right container
HEADER_FOOTER_OVERHEAD_PX = 188  # rough space for title, divider, legend etc.
RIGHT_CONTAINER_HEIGHT_PX = MAP_HEIGHT_PX + HEADER_FOOTER_OVERHEAD_PX

# toggle it on click
with right_col:
    with st.container(key="white_container_right", border=True, height=RIGHT_CONTAINER_HEIGHT_PX):

        # look up the row mathing the currently selected PLR
        selected_row = gdf.loc[gdf["plr_id"] == st.session_state.selected_plr_id].iloc[0]

        # some PLRs (no-data areas) don't exist in df_final at all --
        # guard against the empty lookup instead of crashing on .iloc[0]
        matching_final_rows = df_final.loc[df_final["plr_id"] == st.session_state.selected_plr_id]

        # same for plot df
        matching_plot_df_rows = plot_df.loc[plot_df["plr_id"] == st.session_state.selected_plr_id]

        PLR = selected_row["plr_name"]
        plr_id = selected_row["plr_id"]

        if not matching_final_rows.empty:
            selected_row_from_final = matching_final_rows.iloc[0]
            bez = selected_row_from_final["bez"]
        else:
            bez = "No data available"

        if not matching_plot_df_rows.empty:
            selected_row_from_plot_df = matching_plot_df_rows.iloc[0]
            res_count = selected_row_from_plot_df["res_count"]
            res_count = int(res_count)
        else:
            res_count = "No data available"

        cluster_status = selected_row["cluster_status"]
        cluster_profile = "This is an explanation of a cluster profile"
        ms_status = selected_row["ms_status"]

        if not matching_final_rows.empty:
            
            if selected_row_from_final["ms_over50"] == 0:
                similarity_score = round(selected_row_from_final["oof_prob"], 2)
            else:
                similarity_score = "Proportion of milieu protected area already more than 50% of PLR"
            
            proportion_milieu = f'{round(selected_row_from_final["ms_portion"] * 100, 1)}%'
            watchlist = "Yes" if bool(selected_row_from_final["on_watchlist"]) else "No"
        else:
            similarity_score = "No data available"
            proportion_milieu = "No data available"
            watchlist = "No data available"
        rank_of_watchlist = "This is a placeholder"

        profile_rows = {
            "Name of PLR": PLR,
            "Identification number": plr_id,
            "District": bez,
            "Resident count": res_count,
            "Milieu protection status": ms_status,
            "Gentrification profile": cluster_status,
            "Profile details": cluster_profile,
            "Similarity score": similarity_score,
            "% of milieu protection": proportion_milieu,
            "Watchlist status": watchlist,
            "Watchlist rank": rank_of_watchlist
        }

        rows_html = "".join(
            f'<div style="font-weight:600;">{label}</div><div>{value}</div>'
            for label, value in profile_rows.items()
        )
        st.markdown("#### Short Profile of PLR", text_alignment="center")
        st.markdown(body="*- Please click on a planning area in the map -*", text_alignment="center")
        
        # gray box wih text
        with st.container(key="short_profile_textbox", border=False):
            st.markdown("")
            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:auto 1fr; column-gap:12px; row-gap:12px;">
                    {rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        
           
        if st.button(label="↓ Show more", type="primary"):
            st.session_state.show_profile = not st.session_state.show_profile

###############################################################################
#################### PLOTS PLR ##############################################

#create table with most important variables: 
id_vars = ['plr_id', 'plr_name', 'bez']

vars_keep = [
    "re_miete_niveau", #rent level, 2025
    "re_miete_trend", #rent trend, 2021-2025
    "re_altbau_share", #pre-war building share, 2022
    "re_dichte_all", #airbnb density, 2025
    "re_brw_niveau", #land value, 2022
    "soc_single_parent_household_share_2024", #single parent hh share, 2024
    "soc_transfer_benefit_share_2024", #transfer benefit share, 2024
    "soc_young_to_middle_adult_share_2025", #younger adult (18-45) share, 2025
    "soc_average_household_size_2024", #household size, 2024
    "com_exit_rate_level_2026", #business exit rate, 2026
    "com_upscale_share_level_2026", #share of upscale gastro, 2026
    "com_share_solo_level_2026", #share solo businesses, 2026
    "com_gastro_share_level_2026", #gastro establishments, 2026
    "com_n_gastro" #gastro count, 2026
]

# -----------------------------------------------------------
# Single source of truth: technical name -> display label.
# Renaming happens ONCE here -- every downstream cell reuses
# this dict instead of redefining its own labels.
# -----------------------------------------------------------
labels = {
    "re_miete_niveau": "Rent level (€/m²)",
    "re_miete_trend": "Rent trend",
    "re_altbau_share": "Share of buildings built before 1919",
    "re_dichte_all": "AirBnB density (per 1000 Apts.)",
    "re_brw_niveau": "Land value (€/m²)",
    "soc_single_parent_household_share_2024": "Share of single-parent households",
    "soc_transfer_benefit_share_2024": "Share of benefit recipients",
    "soc_young_to_middle_adult_share_2025": "Share of younger adults",
    "soc_average_household_size_2024": "Average household size",
    "com_exit_rate_level_2026": "Business exit rate",
    "com_upscale_share_level_2026": "Share of upscale gastronomy",
    "com_share_solo_level_2026": "Share of solo businesses",
    "com_gastro_share_level_2026": "Share of gastronomy businesses",
    "com_n_gastro": "Number of gastro establishments"
}
# -----------------------------------------------------------
# Single source of truth: technical name -> reference year
# -----------------------------------------------------------
years_by_tech = {
    "re_miete_niveau": "2025",
    "re_miete_trend": "2021-2025",
    "re_altbau_share": "2022",
    "re_dichte_all": "2025",
    "re_brw_niveau": "2022",
    "soc_single_parent_household_share_2024": "2024",
    "soc_transfer_benefit_share_2024": "2024",
    "soc_young_to_middle_adult_share_2025": "2025",
    "soc_average_household_size_2024": "2024",
    "com_exit_rate_level_2026": "2026",
    "com_upscale_share_level_2026": "2026",
    "com_share_solo_level_2026": "2026",
    "com_gastro_share_level_2026": "2026",
    "com_n_gastro": "2026",
}

# Derived once, reused everywhere downstream -- never redefine labels/years again
years_by_label = {labels[tech]: year for tech, year in years_by_tech.items()}

# plr table
plr_table = df_final[id_vars + vars_keep].rename(columns = labels)

# district table
bez_table = (
    df_final.groupby("bez", as_index=False)[vars_keep]
      .mean().rename(columns = labels)
)

# berlin table
vars_plot = list(labels.values())  # derived from the single labels dict, not a separate list

berlin_table = pd.DataFrame({
    "Variable": vars_plot,
    "Median": df_final[vars_keep].median().values,
    "SD": df_final[vars_keep].std().values,
    "Q1": df_final[vars_keep].quantile(0.25).values,
    "Q3": df_final[vars_keep].quantile(0.75).values,
})
berlin_table = berlin_table.round(2)

# -----------------------------------------------------------
# Sanitize column names: spaces/slashes -> "_", collapse "__"
# -----------------------------------------------------------
plr_table.columns = (
    plr_table.columns
    .str.replace(" ", "_", regex=False)
    .str.replace("/", "_", regex=False)
    .str.replace(r"_+", "_", regex=True)
)

# -----------------------------------------------------------
# Variables to summarize (everything except id/name/bez)
# -----------------------------------------------------------
vars_plot = [c for c in plr_table.columns if c not in ["plr_id", "plr_name", "bez"]]

# -----------------------------------------------------------
# Bezirk aggregates: one row per bez, columns suffixed _bez_<stat>
# add_suffix keeps naming clean -- no MultiIndex, no lambda renaming
# -----------------------------------------------------------
bez_mean   = plr_table.groupby("bez")[vars_plot].mean().add_suffix("_bez_mean")
bez_median = plr_table.groupby("bez")[vars_plot].median().add_suffix("_bez_median")
bez_sd     = plr_table.groupby("bez")[vars_plot].std().add_suffix("_bez_sd")
bez_q1     = plr_table.groupby("bez")[vars_plot].quantile(0.25).add_suffix("_bez_q1")
bez_q3     = plr_table.groupby("bez")[vars_plot].quantile(0.75).add_suffix("_bez_q3")

bez_agg = pd.concat([bez_mean, bez_median, bez_sd, bez_q1, bez_q3], axis=1).reset_index()

plot_table = plr_table.merge(bez_agg, on="bez", how="left")

# -----------------------------------------------------------
# Berlin aggregates: a scalar per variable, broadcast to every row.
# Built as a dict and concatenated once to avoid fragmenting the
# DataFrame with repeated single-column inserts.
# -----------------------------------------------------------
berlin_cols = {}
for var in vars_plot:
    berlin_cols[f"{var}_berlin_mean"]   = plot_table[var].mean()
    berlin_cols[f"{var}_berlin_median"] = plot_table[var].median()
    berlin_cols[f"{var}_berlin_sd"]     = plot_table[var].std()
    berlin_cols[f"{var}_berlin_q1"]     = plot_table[var].quantile(0.25)
    berlin_cols[f"{var}_berlin_q3"]     = plot_table[var].quantile(0.75)

berlin_df = pd.DataFrame([berlin_cols] * len(plot_table), index=plot_table.index)
plot_table = pd.concat([plot_table, berlin_df], axis=1).copy()


# -----------------------------------------------------------
# Dimension colors / labels
# -----------------------------------------------------------
dimension_colors = {
    "re": "#A06B34",   # Real estate -- brown
    "soc": "#6FA8C7",  # Social -- blue
    "com": "#1D5B4E",  # Commercial -- teal
}

dimension_labels = {
    "re": "Real estate",
    "soc": "Social",
    "com": "Commercial",
}

selected_vars_raw = {
    "re":  ["Rent level (€/m²)", "AirBnB density (per 1000 Apts.)"],
    "soc": ["Share of single-parent households", "Share of benefit recipients"],
    "com": ["Business exit rate", "Share of upscale gastronomy"],
}

# -----------------------------------------------------------
# Font sizes
# -----------------------------------------------------------
dimension_fontsize = 14
var_title_fontsize = 11
suptitle_fontsize = 20
legend_fontsize = 14

dims = ["re", "soc", "com"]

# this function takes the already-looked-up "row" as an argument and is only called once we
# actually have a valid row to plot (see the show_profile section further down).
def build_profile_figure(row):
    fig_profile, axes = plt.subplots(2, 3, figsize=(4.2 * 3, 4.5 * 2))

    for col, dim in enumerate(dims):
        for row_idx in range(2):
            ax = axes[row_idx][col]
            var_raw = selected_vars_raw[dim][row_idx]
            var = re.sub(r"_+", "_", var_raw.replace(" ", "_").replace("/", "_"))

            plr_val = row[var]

            bez_center = row[f"{var}_bez_median"]
            berlin_center = row[f"{var}_berlin_median"]

            ax.bar([0], [plr_val], width=0.5, color=dimension_colors[dim], zorder=2)

            bez_line, = ax.plot([-0.5, 0.5], [bez_center, bez_center], color="gray",
                                 linestyle="--", linewidth=2, zorder=3)
            berlin_line, = ax.plot([-0.5, 0.5], [berlin_center, berlin_center], color="black",
                                    linestyle="--", linewidth=2, zorder=3)

            ax.set_xticks([0])
            ax.set_xticklabels([""])
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylabel(var_raw)

            if row_idx == 0:
                ax.text(0.5, 1.28, f"Dimension: {dimension_labels[dim]}",
                        transform=ax.transAxes, ha="center",
                        fontsize=dimension_fontsize, fontweight="bold",
                        color=dimension_colors[dim])
                ax.text(0.5, 1.05, f"{var_raw} ({years_by_label[var_raw]})",
                        transform=ax.transAxes, ha="center",
                        fontsize=var_title_fontsize, fontweight="normal",
                        color="black")
            else:
                ax.set_title(f"{var_raw} ({years_by_label[var_raw]})",
                             fontsize=var_title_fontsize, fontweight="normal", color="black")

    fig_profile.legend(
        handles=[bez_line, berlin_line],
        labels=["District Median", "Berlin Median"],
        loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=2, frameon=False,
        fontsize=legend_fontsize,
    )

    fig_profile.tight_layout(rect=[0, 0.05, 1, 1])
    return fig_profile


#### PROFILE CONTAINER
# render content based on state
if st.session_state.show_profile:
    with st.container(key="white_container_profile", border=True):
        st.markdown(f"#### Profile of Planning Area: {PLR}")
        st.markdown(f"**PLR ID**: {plr_id}")

        # get the row for the currently selected PLR (uses id_vars/vars_keep
        # table you already built above: plot_table)
        matching_plot_rows = plot_table.loc[plot_table["plr_id"] == st.session_state.selected_plr_id]

        if matching_plot_rows.empty:
            st.info("No detailed data available for this planning area.")
        else:
            row = matching_plot_rows.iloc[0]
            bez_name = row["bez"]
            plr_name_for_plot = row.get("plr_name", st.session_state.selected_plr_id)

            bottom_left_col, bottom_right_col = st.columns([1, 2], gap="small")

            with bottom_left_col:
                with st.container(key="left_profile_textbox", border=False, horizontal_alignment="center"):
                    st.markdown("I am a text box that can be used for descriptions :P")

            with bottom_right_col:
                with st.container(key="right_profile_textbox", border=False, horizontal_alignment="center"):
                    st.markdown("##### Commercial Dimension", text_alignment="center")
                    # FIX: build the figure now that "row" actually exists,
                    # and use st.pyplot() -- st.plot() doesn't exist in Streamlit
                    fig_profile = build_profile_figure(row)
                    st.pyplot(fig_profile)