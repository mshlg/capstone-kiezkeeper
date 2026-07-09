##################################################
#### PREREQUISITES ###############################

# load libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd
import json
import numpy as np
import re

# page config
st.set_page_config(layout="wide")

# title
st.logo('kiezkeeper_vector_logo.svg', size="large")
st.markdown("# KiezKeeper :small[Data-Driven Detection of Gentrification in Berlin]")
st.markdown("***")

# container styling
css = """
.st-key-white_container_upper,
.st-key-white_container_left,
.st-key-white_container_right,
.st-key-white_container_profile{
    background: rgba(255, 255, 255);
}
.st-key-upper_intro_textbox,
.st-key-short_profile_textbox{
    background: rgba(245, 244, 244);
    padding: 16px;
}
div[data-testid="stHorizontalBlock"]:has(.st-key-left_profile_textbox) {
    align-items: stretch;
}
div[data-testid="stHorizontalBlock"]:has(.st-key-left_profile_textbox) [data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"]:has(.st-key-left_profile_textbox) [data-testid="stElementContainer"],
div[data-testid="stHorizontalBlock"]:has(.st-key-left_profile_textbox) [data-testid="stLayoutWrapper"] {
    height: 100%;
}
.st-key-left_profile_textbox,
.st-key-right_profile_textbox{
    background: rgba(245, 244, 244);
    padding: 16px;
    height: 100% !important;
    box-sizing: border-box;
}
"""
st.html(f"<style>{css}</style>")

###########################################################
######### INTRODUCTION CONTAINER #########################

# intro container
with st.container(key="white_container_upper", border=True):
    st.markdown("#### This is KiezKeeper.")
    with st.container(key="upper_intro_textbox", border=False):
        st.markdown("##### Gentrification Profiles")
        st.markdown("KiezKeeper groups all of Berlin's neighbourhoods into four profiles: City core, City belt, Disadvantaged outskirts and Affluent outskirts. "
                    "These profiles are based on similarities across three dimensions of neighbourhood change: real estate, social structure and commercial structure. "
                    "The profiles are identified from the data itself rather than defined in advance, "
                    "and can be read as different stages along the path of neighbourhood change.", text_alignment="justify")
        st.markdown("##### Watchlist")
        st.markdown("Within neighbourhoods that do not currently have milieu protection, KiezKeeper compares each area with those already designated by Berlin "
                    "and measures how closely their profiles match. The areas that most closely resemble the existing milieu protection pattern form the watchlist: "
                    "a Berlin-wide, objective and consistent starting point for identifying where protection might be needed next.", text_alignment="justify")
        st.markdown("**On the sidebar, you can choose which resulting map you would like to explore.**")


##########################################################
########## BERLIN MAP ###################################

# map background style
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

# fit zoom to bounding box
def calculate_zoom(min_lon, max_lon, min_lat, max_lat, width_px, height_px, padding_factor=0.9):
    WORLD_DIM = 512  # tile size at zoom level 0

    def lat_to_merc_y(lat):
        rad = np.radians(lat)
        return np.log(np.tan(np.pi / 4 + rad / 2))

    lon_diff = max_lon - min_lon
    lat_diff = lat_to_merc_y(max_lat) - lat_to_merc_y(min_lat)

    zoom_lon = np.log2(width_px * 360 / (lon_diff * WORLD_DIM))
    zoom_lat = np.log2(height_px * 2 * np.pi / (lat_diff * WORLD_DIM))

    zoom = min(zoom_lon, zoom_lat)
    zoom = zoom + np.log2(padding_factor)  # padding
    return zoom


# fit height to bounding box aspect ratio
def calculate_matching_height(min_lon, max_lon, min_lat, max_lat, width_px):
    def lat_to_merc_y(lat):
        rad = np.radians(lat)
        return np.log(np.tan(np.pi / 4 + rad / 2))

    lon_diff = max_lon - min_lon
    lat_diff = lat_to_merc_y(max_lat) - lat_to_merc_y(min_lat)

    height_px = width_px * (360 * lat_diff) / (2 * np.pi * lon_diff)
    return int(round(height_px))


# load final dataset
df_final = pd.read_csv("../data/final_datasets/df_clusters_milieuschutz.csv")
df_final["plr_id"] = df_final["plr_id"].astype(str).str.zfill(8)

# ms threshold columns
MS_THRESHOLD_COLUMNS = ["ms_over50", "ms_over60", "ms_over70", "ms_over80", "ms_over90"]

# slim data for map
df_map = df_final[["plr_id", "bez", "cluster_4k", "ms_portion", "oof_prob", "on_watchlist"] + MS_THRESHOLD_COLUMNS].copy()

# load PLR geometries (cached)
@st.cache_data
def load_plr_geometries():
    plr_geo = gpd.read_file("plr_geometries.gpkg")
    plr_geo["plr_id"] = plr_geo["plr_id"].astype(str).str.zfill(8)
    plr_geo = plr_geo[["plr_id", "plr_name", "geometry"]]
    plr_geo["geometry"] = plr_geo["geometry"].simplify(
        tolerance=3,
        preserve_topology=True
    )
    plr_geo = plr_geo.to_crs(epsg=4326)
    return plr_geo


# load watchlist (cached)
@st.cache_data
def load_watchlist():
    watchlist_df = pd.read_csv("../models/M2_watchlist.csv", dtype={"plr_id": str})
    return watchlist_df


# load data
plr_geo = load_plr_geometries()
watchlist_df = load_watchlist()

# merge geometry with data
gdf = plr_geo.merge(
    df_map,
    on="plr_id",
    how="left"
)

#+++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++ CLUSTER MAP ++++++++++++++++++++++++++++

# cluster labels
cluster_labels = {
    1: "City core",
    3: "City ring",
    2: "Disadvantaged outskirts",
    0: "Affluent outskirts",
}
gdf["cluster_status"] = gdf["cluster_4k"].map(cluster_labels)
gdf.loc[gdf["cluster_4k"].isna(), "cluster_status"] = "No data available"

# cluster code for shared trace
gdf["cluster_code"] = gdf["cluster_4k"]
gdf.loc[gdf["cluster_4k"].isna(), "cluster_code"] = -1

# build geojson
gdf = gdf.reset_index(drop=True)
geojson = json.loads(gdf.to_json())

# map center
min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
center_lon = (min_lon + max_lon) / 2
center_lat = (min_lat + max_lat) / 2

# assumed window width
ASSUMED_WINDOW_WIDTH_PX = 1400

# must match st.columns([...]) ratio below !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MAP_COLUMN_RATIO = 3 / 5

TARGET_WIDTH_PX = ASSUMED_WINDOW_WIDTH_PX * MAP_COLUMN_RATIO

# map sizing
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

# cluster colorscale
colorscale = [
    [0.0, "#FFFFFF"], [0.2, "#FFFFFF"],   # No data available
    [0.2, "#B8B8B8"], [0.4, "#B8B8B8"],   # Affluent outskirts (cluster 0)
    [0.4, "#8B0000"], [0.6, "#8B0000"],   # City core (cluster 1)
    [0.6, "#737373"], [0.8, "#737373"],   # Disadvantaged outskirts (cluster 2)
    [0.8, "#EE4B2B"], [1.0, "#EE4B2B"],   # City ring (cluster 3)
]

# init selected PLR
if "selected_plr_id" not in st.session_state:
    st.session_state.selected_plr_id = gdf["plr_id"].iloc[0]

# ms status text, uncached so it's always current
def compute_ms_status(ms_column, threshold_pct):
    status = pd.Series(
        f"Proportion of milieu protected area less than {threshold_pct}% of PLR",
        index=gdf.index,
    )
    status[gdf[ms_column] == 1] = f"Proportion of milieu protected area more than {threshold_pct}% of PLR"
    status[gdf[ms_column].isna()] = "No data available"
    return status


# build cluster map (cached per threshold)
@st.cache_data(show_spinner=False)
def build_cluster_map(ms_column, threshold_pct):
    ms_status = compute_ms_status(ms_column, threshold_pct)

    # border: black = protected, grey = no data, white = regular
    ms_line_widths = np.select(
        [gdf[ms_column] == 1, gdf["cluster_code"] == -1],
        [2.0, 0.4],
        default=0.4,
    )
    ms_line_colors = np.select(
        [gdf[ms_column] == 1, gdf["cluster_code"] == -1],
        ["#000000", "#999999"],
        default="#ffffff",
    )

    customdata = gdf[["plr_name", "plr_id", "cluster_status"]].copy()
    customdata["ms_status"] = ms_status

    # cluster choropleth
    fig_map = go.Figure(
        go.Choroplethmap(
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

    # layout
    fig_map.update_layout(
        map_style=grey_map_style,
        map=dict(
            center={"lat": center_lat, "lon": center_lon},
            zoom=zoom_level,
            bearing=0,
            pitch=0,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=MAP_HEIGHT_PX,
        autosize=True,
    )
    return fig_map

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++ SIMILARITY MAP +++++++++++++++++++++++++++++++++++

# watchlist gradient colorscale, hard steps via duplicate positions
SIMILARITY_COLORSCALE = [
    [0.000, "#FFFFFF"],
    [0.333, "#FFFFFF"],   # no data band ends
    [0.334, "#8a8a8a"],
    [0.666, "#8a8a8a"],   # base/protected band ends
    [0.667, "#FFFFFF"],   # watchlist gradient starts (resemblance 0)
    [0.833, "#EE4B2B"],   # resemblance 0.5
    [1.000, "#8B0000"],   # resemblance 1
]


# build similarity map, single trace like the cluster map
@st.cache_data(show_spinner=False)
def build_similarity_map(colorscale=SIMILARITY_COLORSCALE):
    local = gdf[["plr_id", "plr_name", "ms_over50", "on_watchlist", "oof_prob"]].merge(
        watchlist_df[["plr_id", "rank"]], on="plr_id", how="left"
    )
    is_wl = local["on_watchlist"] == True

    # combined z: -1 no data, 0 base/protected, [1, 2] watchlist by resemblance
    z = pd.Series(0.0, index=local.index)
    z[local["ms_over50"].isna()] = -1.0
    wl_min = local.loc[is_wl, "oof_prob"].min()
    wl_max = local.loc[is_wl, "oof_prob"].max()
    wl_range = wl_max - wl_min if wl_max > wl_min else 1.0
    z[is_wl] = 1.0 + (local.loc[is_wl, "oof_prob"] - wl_min) / wl_range

    # border: black = protected, grey = no data, white = regular
    line_widths = np.select(
        [local["ms_over50"] == 1, local["ms_over50"].isna()],
        [0.3, 0.4],
        default=0.3,
    )
    line_colors = np.select(
        [local["ms_over50"] == 1, local["ms_over50"].isna()],
        ["#000000", "#999999"],
        default="#ffffff",
    )

    # hover status text
    status = pd.Series("Not on watchlist", index=local.index)
    status[local["ms_over50"] == 1] = "Already milieu-protected"
    status[local["ms_over50"].isna()] = "No data available"
    status[is_wl] = (
        "On watchlist -- rank " + local.loc[is_wl, "rank"].astype("Int64").astype(str)
        + ", resemblance " + local.loc[is_wl, "oof_prob"].round(2).astype(str)
    )

    customdata = local[["plr_name", "plr_id"]].copy()
    customdata["status"] = status

    fig_similarity = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=local["plr_id"],
            z=z,
            zmin=-1,
            zmax=2,
            featureidkey="properties.plr_id",
            colorscale=colorscale,
            showscale=False,
            marker_opacity=0.85,
            marker_line_width=line_widths,
            marker_line_color=line_colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "PLR-ID: %{customdata[1]}<br>"
                "%{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    # layout
    fig_similarity.update_layout(
        map_style=grey_map_style,
        map=dict(
            center={"lat": center_lat, "lon": center_lon},
            zoom=zoom_level,
            bearing=0,
            pitch=0,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=MAP_HEIGHT_PX,
        autosize=True,
        showlegend=False,
    )
    return fig_similarity


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++ SHOW MAPS +++++++++++++++++++++++++++++++++++++++++

# map choice
map_status = st.sidebar.radio("Please choose a map.", options=["Gentrification Profiles", "Watchlist"], horizontal=True)
if map_status == "Gentrification Profiles":
    ms_proportion = st.sidebar.radio(
        label="% of total area of PLR designated for milieu protection:",
        options=["more than 50%", "more than 60%", "more than 70%", "more than 80%", "more than 90%"]
    )
else:
    ms_proportion = "more than 50%"  # default, unused for similarity map

# translate threshold selection
ms_threshold_columns = {
    "more than 50%": "ms_over50",
    "more than 60%": "ms_over60",
    "more than 70%": "ms_over70",
    "more than 80%": "ms_over80",
    "more than 90%": "ms_over90",
}
ms_column = ms_threshold_columns[ms_proportion]
threshold_pct = re.search(r"\d+", ms_proportion).group()

# keep ms status current
gdf["ms_status"] = compute_ms_status(ms_column, threshold_pct)

# build maps
fig_map = build_cluster_map(ms_column, threshold_pct)
fig_similarity = build_similarity_map(SIMILARITY_COLORSCALE)

# show maps
left_col, right_col = st.columns([3, 2], gap="small")

with left_col:
    with st.container(key="white_container_left", border=True):

        map_event = None
        map_second_event = None

        # cluster map
        if map_status == "Gentrification Profiles":
            st.markdown("#### Planning areas (PLR) of Berlin -- Gentrification Profiles", text_alignment="center")
            st.markdown(body="*- Please choose a map on the sidebar -*", text_alignment="center")

            map_event = st.plotly_chart(
                fig_map,
                width="stretch",
                config={"responsive": True},
                on_select="rerun",
                selection_mode="points",
                key="cluster_map"
            )

            # legend
            st.markdown(
                """
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.9rem; margin-top:8px;">
                <span><span style="display:inline-block;width:15px;height:15px;background:#8B0000;border-radius:2px;"></span> City core</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#EE4B2B;border-radius:2px;"></span> City belt</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#737373;border-radius:2px;"></span> Disadvantaged outskirts</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#B8B8B8;border-radius:2px;"></span> Affluent outskirts</span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.9rem; margin-top:8px;">
                <span><span style="display:inline-block;width:15px;height:15px;background:#ffffff;border:1px solid #999;border-radius:2px;"></span> No data</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:none;border:2px solid black;border-radius:2px;"></span> Milieu protection (&gt;""" + threshold_pct + """%)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # similarity map
        else:
            st.markdown("#### Planning areas (PLR) of Berlin -- Watchlist", text_alignment="center")
            st.markdown(body="*- Please choose a map on the sidebar -*", text_alignment="center")

            map_second_event = st.plotly_chart(
                fig_similarity,
                width="stretch",
                config={"responsive": True},
                on_select="rerun",
                selection_mode="points",
                key="similarity_map"
            )

            # legend
            st.markdown(
                """
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.9rem; margin-top:8px;">
                <span><span style="display:inline-block;width:15px;height:15px;background:linear-gradient(90deg,#FFFFFF,#EE4B2B,#8B0000);border-radius:2px;"></span> On watchlist (shaded by similarity score)</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#cccccc;border-radius:2px;"></span> Not on watchlist</span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.9rem; margin-top:8px;">
                <span><span style="display:inline-block;width:15px;height:15px;background:#ffffff;border:1px solid #999;border-radius:2px;"></span> No data</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#ffffff;border:2px solid black;border-radius:2px;"></span> Milieu protection (&gt; 50 %) </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# handle map click
active_event = map_event if map_event is not None else map_second_event

if active_event and active_event["selection"]["points"]:
    clicked_point = active_event["selection"]["points"][0]
    if "location" in clicked_point:
        st.session_state.selected_plr_id = clicked_point["location"]


################################################################
##################### SHORT PROFILE #############################

# load short profile data
plot_df = pd.read_csv("data/plot_df.csv", dtype={"plr_id": str})
# watchlist_df already loaded above (load_watchlist())

# init state
if "show_profile" not in st.session_state:
    st.session_state.show_profile = False

# right container height
HEADER_FOOTER_OVERHEAD_PX = 188
RIGHT_CONTAINER_HEIGHT_PX = MAP_HEIGHT_PX + HEADER_FOOTER_OVERHEAD_PX

# short profile container
with right_col:
    with st.container(key="white_container_right", border=True, height=RIGHT_CONTAINER_HEIGHT_PX):

        # look up selected PLR
        selected_row = gdf.loc[gdf["plr_id"] == st.session_state.selected_plr_id].iloc[0]
        matching_final_rows = df_final.loc[df_final["plr_id"] == st.session_state.selected_plr_id]
        matching_plot_df_rows = plot_df.loc[plot_df["plr_id"] == st.session_state.selected_plr_id]
        matching_watchlist_df_rows = watchlist_df.loc[watchlist_df["plr_id"] == st.session_state.selected_plr_id]

        PLR = selected_row["plr_name"]
        plr_id = selected_row["plr_id"]

        if not matching_final_rows.empty:
            selected_row_from_final = matching_final_rows.iloc[0]
            bez = selected_row_from_final["bez"]
        else:
            bez = "No data available"

        if not matching_plot_df_rows.empty:
            selected_row_from_plot_df = matching_plot_df_rows.iloc[0]
            res_count = int(selected_row_from_plot_df["res_count"])
        else:
            res_count = "No data available"

        cluster_status = selected_row["cluster_status"]
        ms_status = selected_row["ms_status"]

        # short cluster descriptions
        cluster_profile_1 = (
            "City core: The high-value inner city where gentrification is already advanced"
            "\n - Highest rents, land values and Airbnb density; steepest rent increase"
            "\n - Most gastronomy, fewest solo businesses, lowest exit rate; low benefit dependency"
            " \n - 13% of PLRs majority-protected by Milieuschutz")

        cluster_profile_3 = (
            "City belt: The actively transforming inner city — gentrification in progress"
            "\n - Highest share of buildings built before 1919, high land value and Airbnb density, strong rent increase"
            "\n - Youngest residents, smallest households; active churn (elevated exit rate, strong gastronomy"
            "\n - 63% of PLRs majority-protected — by far the most protected cluster")

        cluster_profile_2 = (
            "Disadvantaged outskirts: Socially strained periphery, little upgrading pressure"
            "\n - Lowest rents and land values, negligible Airbnb; weakest rent growth"
            "\n - Highest benefit dependency and single-parent share; most solo businesses, highest exit rate"
            "\n - 5% of PLRs majority-protected — almost no coverage")

        cluster_profile_0 = (
            "Affluent outskirts: Settled, prosperous, family-oriented periphery — not gentrifying"
            "\n - Land values and rents below city average; lowest young-adult share"
            "\n - Lowest benefit dependency, largest households, oldest businesses, most newer buildings"
            "\n - 0% of PLRs protected — displacement not a policy concern")

        if not matching_final_rows.empty:
            if cluster_status == "City core":
                cluster_profile = cluster_profile_1
            elif cluster_status == "City belt":
                cluster_profile = cluster_profile_3
            elif cluster_status == "Disadvantaged outskirts":
                cluster_profile = cluster_profile_2
            else:
                cluster_profile = cluster_profile_0
        else:
            cluster_profile = "No data available"

        if not matching_final_rows.empty:
            if selected_row_from_final["ms_over50"] == 0:
                similarity_score = round(selected_row_from_final["oof_prob"], 2)
            else:
                similarity_score = "Proportion of milieu protected area already more than 50% of PLR"

            proportion_milieu = f'{round(selected_row_from_final["ms_portion"] * 100, 1)}%'
            watchlist = "On watchlist" if bool(selected_row_from_final["on_watchlist"]) else "Not on watchlist"
        else:
            similarity_score = "No data available"
            proportion_milieu = "No data available"
            watchlist = "No data available"

        if not matching_final_rows.empty:
            if bool(selected_row_from_final["on_watchlist"]):
                selected_row_from_watchlist_df = matching_watchlist_df_rows.iloc[0]
                watchlist_rank = selected_row_from_watchlist_df['rank']
            else:
                watchlist_rank = "Not on watchlist"
        else:
            watchlist_rank = "No data available"

        profile_rows = {
            "Name of PLR": PLR,
            "Identification number": plr_id,
            "District": bez,
            "Resident count": res_count,
            "Milieu protection status": ms_status,
            "% of milieu protection": proportion_milieu,
            "Gentrification profile": cluster_status,
            "Similarity score": similarity_score,
            "Watchlist status": watchlist,
            "Watchlist rank": watchlist_rank
        }

        rows_html = "".join(
            f'<div style="font-weight:600;">{label}</div><div>{value}</div>'
            for label, value in profile_rows.items()
        )
        st.markdown("#### Short Profile of PLR", text_alignment="center")
        st.markdown(body="*- Please click on a planning area in the map -*", text_alignment="center")

        # profile text box
        with st.container(key="short_profile_textbox", border=False, height=MAP_HEIGHT_PX):
            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:auto 1fr; column-gap:11px; row-gap:7px;">
                    {rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(f"**Profile description** \n\n {cluster_profile}")

        if st.button(label="↓ Show more", type="primary"):
            st.session_state.show_profile = not st.session_state.show_profile

###############################################################################
#################### PLOTS PLR ##############################################

# display label -> reference year
years_by_label = {
    "Rent level (€/m²)": "2025",
    "Rent trend": "2021-2025",
    "Share of buildings built before 1919": "2022",
    "AirBnB density (per 1000 Apts.)": "2025",
    "Land value (€/m²)": "2022",
    "Share of single-parent households": "2024",
    "Share of benefit recipients": "2024",
    "Share of younger adults": "2025",
    "Average household size": "2024",
    "Business exit rate": "2026",
    "Share of upscale gastronomy": "2026",
    "Share of solo businesses": "2026",
    "Share of gastronomy businesses": "2026",
    "Number of gastro establishments": "2026",
}

# dimension colors/labels
dimension_colors = {
    "re": "#A06B34",
    "soc": "#6FA8C7",
    "com": "#1D5B4E",
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

# font sizes
dimension_fontsize = 16
var_title_fontsize = 16
legend_fontsize = 14
y_label_fontsize = 14

dims = ["re", "soc", "com"]

# build profile figure
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
            ax.set_ylabel(var_raw, fontsize=y_label_fontsize)

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


# profile container
if st.session_state.show_profile:
    with st.container(key="white_container_profile", border=True, height="content"):
        st.markdown(f"#### Key Indicators by Dimension for PLR: {PLR}", text_alignment="center")

        matching_plot_rows = plot_df.loc[plot_df["plr_id"] == st.session_state.selected_plr_id]

        if matching_plot_rows.empty:
            st.info("No detailed data available for this planning area.")
        else:
            row = matching_plot_rows.iloc[0]
            bez_name = row["bez"]
            plr_name_for_plot = row.get("plr_name", st.session_state.selected_plr_id)

            with st.container(key="right_profile_textbox", border=False, horizontal_alignment="center"):
                fig_profile = build_profile_figure(row)
                st.pyplot(fig_profile, width=1200)
                st.markdown("##### Description", text_alignment="center")
                st.markdown("Each planning area (PLR) is characterised across the three dimensions — real estate, social, and commercial — "
                    "with every indicator shown against two reference lines: the median of its district (grey) and of Berlin as a whole (black). "
                    "This dual benchmark places each PLR both in its local and in its city-wide context. "
                    "The variables are chosen to capture the mechanisms through which gentrification becomes visible at the neighbourhood level. " \
                    "In the real estate dimension, rent level (€/m²) tracks the price pressure that drives displacement, while Airbnb density (listings per 1,000 apartments) " \
                    "measures the withdrawal of housing from the regular market through short-term letting — an early and spatially concentrated signal of touristic upgrading. " \
                    "The social dimension captures displacement pressure on vulnerable residents: the share of benefit recipients and of single-parent households " \
                    "identify two groups that are very exposed to displacement. The commercial dimension reflects the transformation of the local economy: " \
                    "the business exit rate captures the turnover and closure of established businesses, and the share of upscale gastronomy indicates the " \
                    "commercial upgrading that typically accompanies — and reinforces — residential gentrification.", text_alignment="justify")