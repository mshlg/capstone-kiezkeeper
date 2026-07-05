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

# set page to wide format
st.set_page_config(layout="wide")

# set title
st.logo('kiezkeeper_vector_logo.svg', size="large")
st.markdown("# KiezKeeper :small[Data-Driven Detection of Gentrification in Berlin]")
st.markdown("***")

# set style for containers
css = """
.st-key-white_container_left,
.st-key-white_container_right,
.st-key-white_container_profile{
    background: rgba(255, 255, 255);
}
.st-key-profile_textbox{
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
df_map = df_final[["plr_id", "cluster_4k", "ms_binary", "ms_portion"]].copy()

# Cache the function so the geodata is not loaded again on every rerun
@st.cache_data
def load_plr_geometries():

    # URL for the Berlin PLR geometries
    url = (
        "https://gdi.berlin.de/services/wfs/lor_2021?service=WFS&version=2.0.0"
        "&request=GetFeature&typeNames=lor_2021:a_lor_plr_2021&outputFormat=application/json"
    )

    # Load the PLR geometries from the URL
    plr_geo = gpd.read_file(url)

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

# Create readable cluster labels
cluster_labels = {
    1: "City core",
    3: "City ring",
    2: "Disadvantaged outskirts",
    0: "Affluent outskirts",
}
gdf["cluster_status"] = gdf["cluster_4k"].map(cluster_labels)
gdf.loc[gdf["cluster_4k"].isna(), "cluster_status"] = "No data available"

# Readable milieuschutz label for the hover text
gdf["ms_status"] = "No milieu protection"
gdf.loc[gdf["ms_binary"] == 1, "ms_status"] = "Milieu protection area"

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

# Static per-row border styling for milieuschutz areas (thicker + black
# border). This does NOT depend on the click/session state, so it stays
# stable across reruns and doesn't interfere with the click handling
# (unlike a selection-based border, which caused instability earlier).
ms_line_widths = np.where(gdf["ms_binary"] == 1, 2.0, 0.4)
ms_line_colors = np.where(gdf["ms_binary"] == 1, "#000000", "#ffffff")

# Create map
# Single Choroplethmapbox trace instead of one trace per category,
# so hover works consistently for every area
fig = go.Figure(
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
        customdata=gdf[["plr_name", "plr_id", "cluster_status", "ms_status"]],
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
fig.update_layout(
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

# Show map on half the page
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    with st.container(key="white_container_left", border=True):
        st.subheader("Map of Berlin")

        map_event = st.plotly_chart(
            fig,
            width="stretch",
            config={"responsive": True},
            on_select="rerun",
            selection_mode="points",
            key="berlin_map"
        )

        # legend
        st.markdown(
            """
            <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:1rem; margin-top:8px;">
              <span><span style="display:inline-block;width:15px;height:15px;background:#8B0000;border-radius:2px;"></span> City core</span>
              <span><span style="display:inline-block;width:15px;height:15px;background:#EE4B2B;border-radius:2px;"></span> City ring</span>
              <span><span style="display:inline-block;width:15px;height:15px;background:#737373;border-radius:2px;"></span> Disadvantaged outskirts</span>
              <span><span style="display:inline-block;width:15px;height:15px;background:#B8B8B8;border-radius:2px;"></span> Affluent outskirts</span>
              <span><span style="display:inline-block;width:15px;height:15px;background:#ffffff;border:1px solid #999;border-radius:2px;"></span> No data available</span>
              <span><span style="display:inline-block;width:15px;height:15px;background:none;border:2px solid black;border-radius:2px;"></span> Milieu protection area</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# update selection if the user clicked on the map (NOT indented under left_col)
if map_event and map_event["selection"]["points"]:
    clicked_plr_id = map_event["selection"]["points"][0]["location"]
    st.session_state.selected_plr_id = clicked_plr_id


#############################################################
######### DOWNLOAD BUTTON ##################################

# # download button
# plr = 'Koepenik'

# csv = df_milieu.to_csv().encode("utf-8")

# st.download_button(
#     label="Steckbrief Download",
#     data=csv,
#     file_name=f"{plr}.csv",
#     mime="text/csv",
#     type="primary",
#     icon=":material/download:"
# )

################################################################
######## EXPANDING CONTAINER AFTER BUTTONPRESS #################

# initialize state
if "show_profile" not in st.session_state:
    st.session_state.show_profile = False

# container height for right container
HEADER_FOOTER_OVERHEAD_PX = 126  # rough space for title, divider, legend etc.
RIGHT_CONTAINER_HEIGHT_PX = MAP_HEIGHT_PX + HEADER_FOOTER_OVERHEAD_PX

# toggle it on click
with right_col:
    with st.container(key="white_container_right", border=True, height=RIGHT_CONTAINER_HEIGHT_PX):

        # look up the row mathing the currently selected PLR
        selected_row = gdf.loc[gdf["plr_id"] == st.session_state.selected_plr_id].iloc[0]
        
        PLR = selected_row["plr_name"]
        plr_id = selected_row["plr_id"]
        cluster_status = selected_row["cluster_status"]
        ms_status = selected_row["ms_status"]
        
        st.markdown("### Short Profile")

        # gray boy for text
        with st.container(key="profile_textbox", border=False):
            st.markdown("*- Please click on a planning area in the map -*", text_alignment="center")
            st.markdown("")
            st.markdown(f"**Name of planning area**: {PLR}")
            st.markdown(f"**Identification number of planning area**: {plr_id}")
            st.markdown(f"**Gentrification cluster**: {cluster_status}")
            st.markdown(f"**Milieu protection status**: {ms_status}")
        
        if st.button(label="↓ Open profile", type="primary"):
            st.session_state.show_profile = not st.session_state.show_profile

# render content based on state
if st.session_state.show_profile:
    with st.container(key="white_container_profile", border=True):
        st.subheader("Profile of Planning Area")
        
        # download button
        csv = df_final.to_csv().encode("utf-8")

        st.download_button(
            label="Download profile",
            data=csv,
            file_name=f"{PLR}.csv",
            mime="text/csv",
            type="primary",
            icon=":material/download:"
        )