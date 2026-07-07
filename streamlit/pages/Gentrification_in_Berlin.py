##################################################
#### PREREQUISITES###############################

# load libraries
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

# flatten geometry to lon/lat lines (for boundary traces)
def geometry_to_lonlat_lines(geoseries):
    lons, lats = [], []
    for geom in geoseries:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            xs, ys = part.xy
            lons.extend(xs)
            lats.extend(ys)
            lons.append(None)
            lats.append(None)
    return lons, lats


# berlin outline (cached, shared by both maps)
@st.cache_data(show_spinner=False)
def get_berlin_outline():
    boundary = gdf.geometry.unary_union.boundary
    return geometry_to_lonlat_lines([boundary])


# ms status text (uncached, always current)
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

    # border marks milieu protection
    ms_line_widths = np.where(gdf[ms_column] == 1, 2.0, 0.4)
    ms_line_colors = np.where(gdf[ms_column] == 1, "#000000", "#ffffff")

    customdata = gdf[["plr_name", "plr_id", "cluster_status"]].copy()
    customdata["ms_status"] = ms_status

    # cluster choropleth
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

    # berlin outline
    outline_lons, outline_lats = get_berlin_outline()
    fig_map.add_trace(
        go.Scattermapbox(
            lon=outline_lons,
            lat=outline_lats,
            mode="lines",
            line=dict(width=1.2, color="#3a3a3a"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # layout
    fig_map.update_layout(
        mapbox_style=grey_map_style,
        mapbox=dict(
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

# watchlist colorscale (matches cluster map reds)
WATCHLIST_COLORSCALE = [
    [0.0, "#FFFFFF"],
    [0.5, "#EE4B2B"],
    [1.0, "#8B0000"],
]


# build similarity map (cached, no sidebar dependency)
@st.cache_data(show_spinner=False)
def build_similarity_map():
    fig_similarity = go.Figure()

    # no data
    no_data = gdf.loc[gdf["ms_over50"].isna()]
    if not no_data.empty:
        fig_similarity.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=no_data["plr_id"],
                z=[0] * len(no_data),
                featureidkey="properties.plr_id",
                colorscale=[[0, "#FFFFFF"], [1, "#FFFFFF"]],
                showscale=False,
                marker_opacity=0.85,
                marker_line_width=0.4,
                marker_line_color="#999999",
                customdata=no_data[["plr_name", "plr_id"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "PLR-ID: %{customdata[1]}<br>"
                    "Status: No data available"
                    "<extra></extra>"
                ),
                name="no_data",
            )
        )

    # base (not on watchlist, not protected)
    base = gdf.loc[gdf["ms_over50"].notna()]
    if not base.empty:
        fig_similarity.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=base["plr_id"],
                z=[0] * len(base),
                featureidkey="properties.plr_id",
                colorscale=[[0, "#B8B8B8"], [1, "#B8B8B8"]],
                showscale=False,
                marker_opacity=0.85,
                marker_line_width=0.3,
                marker_line_color="#ffffff",
                customdata=base[["plr_name", "plr_id"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "PLR-ID: %{customdata[1]}<br>"
                    "Status: not on watchlist"
                    "<extra></extra>"
                ),
                name="base",
            )
        )

    # protected (same fill as base, marked only by border)
    protected = gdf.loc[gdf["ms_over50"] == 1]
    if not protected.empty:
        fig_similarity.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=protected["plr_id"],
                z=[0] * len(protected),
                featureidkey="properties.plr_id",
                colorscale=[[0, "#B8B8B8"], [1, "#B8B8B8"]],
                showscale=False,
                marker_opacity=0.85,
                marker_line_width=2.0,
                marker_line_color="#000000",
                customdata=protected[["plr_name", "plr_id"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "PLR-ID: %{customdata[1]}<br>"
                    "Status: already milieu-protected"
                    "<extra></extra>"
                ),
                name="protected",
            )
        )

    # watchlist, shaded by resemblance score
    watchlist_area = gdf.loc[gdf["on_watchlist"] == True].merge(
        watchlist_df[["plr_id", "rank"]], on="plr_id", how="left"
    )
    if not watchlist_area.empty:
        fig_similarity.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=watchlist_area["plr_id"],
                z=watchlist_area["oof_prob"],
                zmin=float(watchlist_area["oof_prob"].min()),
                zmax=float(watchlist_area["oof_prob"].max()),
                featureidkey="properties.plr_id",
                colorscale=WATCHLIST_COLORSCALE,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Resemblance<br>(oof_prob)", font=dict(size=11)),
                    thickness=15,
                    len=0.5,
                ),
                marker_opacity=0.9,
                marker_line_width=0.3,
                marker_line_color="#ffffff",
                customdata=watchlist_area[["plr_name", "plr_id", "oof_prob", "rank"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "PLR-ID: %{customdata[1]}<br>"
                    "Watchlist rank: %{customdata[3]}<br>"
                    "Resemblance score: %{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
                name="watchlist",
            )
        )

    # berlin outline
    outline_lons, outline_lats = get_berlin_outline()
    fig_similarity.add_trace(
        go.Scattermapbox(
            lon=outline_lons,
            lat=outline_lats,
            mode="lines",
            line=dict(width=1.2, color="#3a3a3a"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # layout
    fig_similarity.update_layout(
        mapbox_style=grey_map_style,
        mapbox=dict(
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

# sidebar controls
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
fig_similarity = build_similarity_map()

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

        # similarity map
        else:
            st.markdown("#### Planning areas (PLR) of Berlin -- Similarity Scores", text_alignment="center")
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
                <span><span style="display:inline-block;width:15px;height:15px;background:#B8B8B8;border-radius:2px;"></span> Not on watchlist</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:linear-gradient(90deg,#FFFFFF,#EE4B2B,#8B0000);border-radius:2px;"></span> On watchlist (shaded by resemblance score)</span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:0.9rem; margin-top:8px;">
                <span><span style="display:inline-block;width:15px;height:15px;background:#ffffff;border:1px solid #999;border-radius:2px;"></span> No data</span>
                <span><span style="display:inline-block;width:15px;height:15px;background:#B8B8B8;border:2px solid black;border-radius:2px;"></span> Already milieu-protected</span>
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
            "Gentrification profile": cluster_status,
            "Profile details": cluster_profile,
            "Similarity score": similarity_score,
            "% of milieu protection": proportion_milieu,
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

# profile table setup
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

# labels
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

# reference years
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

years_by_label = {labels[tech]: year for tech, year in years_by_tech.items()}

# plr/district/berlin tables
plr_table = df_final[id_vars + vars_keep].rename(columns = labels)

bez_table = (
    df_final.groupby("bez", as_index=False)[vars_keep]
      .mean().rename(columns = labels)
)

vars_plot = list(labels.values())

berlin_table = pd.DataFrame({
    "Variable": vars_plot,
    "Median": df_final[vars_keep].median().values,
    "SD": df_final[vars_keep].std().values,
    "Q1": df_final[vars_keep].quantile(0.25).values,
    "Q3": df_final[vars_keep].quantile(0.75).values,
})
berlin_table = berlin_table.round(2)

# sanitize column names
plr_table.columns = (
    plr_table.columns
    .str.replace(" ", "_", regex=False)
    .str.replace("/", "_", regex=False)
    .str.replace(r"_+", "_", regex=True)
)

vars_plot = [c for c in plr_table.columns if c not in ["plr_id", "plr_name", "bez"]]

# district aggregates
bez_mean   = plr_table.groupby("bez")[vars_plot].mean().add_suffix("_bez_mean")
bez_median = plr_table.groupby("bez")[vars_plot].median().add_suffix("_bez_median")
bez_sd     = plr_table.groupby("bez")[vars_plot].std().add_suffix("_bez_sd")
bez_q1     = plr_table.groupby("bez")[vars_plot].quantile(0.25).add_suffix("_bez_q1")
bez_q3     = plr_table.groupby("bez")[vars_plot].quantile(0.75).add_suffix("_bez_q3")

bez_agg = pd.concat([bez_mean, bez_median, bez_sd, bez_q1, bez_q3], axis=1).reset_index()

plot_table = plr_table.merge(bez_agg, on="bez", how="left")

# berlin aggregates
berlin_cols = {}
for var in vars_plot:
    berlin_cols[f"{var}_berlin_mean"]   = plot_table[var].mean()
    berlin_cols[f"{var}_berlin_median"] = plot_table[var].median()
    berlin_cols[f"{var}_berlin_sd"]     = plot_table[var].std()
    berlin_cols[f"{var}_berlin_q1"]     = plot_table[var].quantile(0.25)
    berlin_cols[f"{var}_berlin_q3"]     = plot_table[var].quantile(0.75)

berlin_df = pd.DataFrame([berlin_cols] * len(plot_table), index=plot_table.index)
plot_table = pd.concat([plot_table, berlin_df], axis=1).copy()

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
dimension_fontsize = 14
var_title_fontsize = 11
suptitle_fontsize = 20
legend_fontsize = 14

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


# profile container
if st.session_state.show_profile:
    with st.container(key="white_container_profile", border=True):
        st.markdown(f"#### Profile of Planning Area: {PLR}")
        st.markdown(f"**PLR ID**: {plr_id}")

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
                    fig_profile = build_profile_figure(row)
                    st.pyplot(fig_profile)