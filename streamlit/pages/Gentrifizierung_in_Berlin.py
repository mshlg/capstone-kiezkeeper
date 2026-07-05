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
st.markdown("# KiezKeeper :small[Frühwarnung für Berliner Kieze]")
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
# import Milieuschutzareale
df_milieu = pd.read_csv("target_milieuschutz.csv")
df_milieu = df_milieu.drop(["plr_name", "bez", "milieu_anteil"], axis=1)
df_milieu["plr_id"] = df_milieu["plr_id"].astype(str).str.zfill(8)

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
        tolerance=20,
        preserve_topology=True
    )

    # Convert geometries to web map coordinates
    plr_geo = plr_geo.to_crs(epsg=4326)

    # Return the prepared geodata
    return plr_geo


# Load the prepared PLR geometries
plr_geo = load_plr_geometries()

# Make sure the PLR ID in the data table has the same format
df_milieu["plr_id"] = df_milieu["plr_id"].astype(str).str.zfill(8)

# Join the geodata with the Milieuschutz data
gdf = plr_geo.merge(
    df_milieu,
    on="plr_id",
    how="left"
)

# Create a readable map categories
gdf["milieu_status"] = "Kein Milieuschutz"
gdf.loc[gdf["milieu_majoritaet"] == 1, "milieu_status"] = "Milieuschutz"
gdf.loc[gdf["milieu_majoritaet"].isna(), "milieu_status"] = "Keine Daten"

# Numeric code per status, needed for the single shared trace
# (this is what makes hover work the same for ALL areas, not just protected ones)
status_to_code = {"Keine Daten": -1, "Kein Milieuschutz": 0, "Milieuschutz": 1}
gdf["milieu_code"] = gdf["milieu_status"].map(status_to_code)

# Reset the index and create a unique ID for Plotly
gdf = gdf.reset_index(drop=True)
geojson = json.loads(gdf.to_json())

# Convert the GeoDataFrame to GeoJSON for Plotly
geojson = json.loads(gdf.to_json())

# Calculate center from geometries
min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
center_lon = (min_lon + max_lon) / 2
center_lat = (min_lat + max_lat) / 2

# --- HIER die alte Zeile ersetzen ---
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

# Discrete colorscale for the 3 status categories
colorscale = [
    [0.00, "#4b5563"],  # Keine Daten
    [0.33, "#4b5563"],
    [0.34, "#fac4c4"],  # Kein Milieuschutz
    [0.66, "#fac4c4"],
    [0.67, "#d62828"],  # Milieuschutz
    [1.00, "#d62828"],
]

# Create map
# Single Choroplethmapbox trace instead of one trace per category,
# so hover works consistently for every area
fig = go.Figure(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=gdf["plr_id"],
        z=gdf["milieu_code"],
        zmin=-1,
        zmax=1,
        featureidkey="properties.plr_id",
        colorscale=colorscale,
        showscale=False,
        marker_opacity=0.75,
        marker_line_width=0.4,
        marker_line_color="grey",
        customdata=gdf[["plr_name", "plr_id", "milieu_status"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "PLR-ID: %{customdata[1]}<br>"
            "Status: %{customdata[2]}"
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
    uirevision="constant",        # width stays responsive
)

# Show map on half the page
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    with st.container(key="white_container_left", border=True):
        st.subheader("Karte von Berlin")
        st.plotly_chart(fig, width="stretch", config={"responsive": True})

        # legend
        st.markdown(
            """
            <div style="display:flex; gap:16px; font-size:1rem; margin-top:8px;">
              <span><span style="display:inline-block;width:15px;height:15px;background:#d62828;border-radius:2px;"></span> Milieuschutz</span>
              <span><span style="display:inline-block;width:15px;height:15px;background:#fac4c4;border-radius:2px;"></span> Kein Milieuschutz</span>
              <span><span style="display:inline-block;width:15px;height:15px;background:#4b5563;border-radius:2px;"></span> Keine Daten</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        PLR = "Koepenik"
        plr_id = 12345678
        st.subheader("Kurzprofil für Planungsraum")

        # gray boy for text
        with st.container(key="profile_textbox", border=False):
            st.markdown(f"PLR Name: {PLR}")
            st.markdown(f"PLR ID: {plr_id}")
        
        if st.button(label="↓ Steckbrief öffnen", type="primary"):
            st.session_state.show_profile = not st.session_state.show_profile

# render content based on state
if st.session_state.show_profile:
    with st.container(key="white_container_profile", border=True):
        st.subheader("Planungsraum-Steckbrief")
        # download button
        plr = 'Koepenik'

        csv = df_milieu.to_csv().encode("utf-8")

        st.download_button(
            label="Steckbrief Download",
            data=csv,
            file_name=f"{plr}.csv",
            mime="text/csv",
            type="primary",
            icon=":material/download:"
        )