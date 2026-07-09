##################################################
#### PREREQUISITES###############################

# load libraries
import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import io
import os
import geopandas as gpd
import contextily as cx
from pywaffle import Waffle
from fpdf import FPDF

# page config
st.set_page_config(layout="wide")

# title
st.logo('kiezkeeper_vector_logo.svg', size="large")
st.markdown("# KiezKeeper :small[Data-Driven Detection of Gentrification in Berlin]")
st.markdown("***")

# container styling
css = """
.st-key-white_container_selection,
.st-key-white_container_profile{
    background: rgba(255, 255, 255);
}
.st-key-profile_textbox,
.st-key-plot_container,
.st-key-bar_description_box,
.st-key-waffle_description_box,
.st-key-bar_plot_container,
.st-key-waffle_container{
    background: rgba(245, 244, 244);
    padding: 16px;
}
/* center chart inside its (narrower) column -- flex columns need
   align-items for cross-axis centering, not justify-content */
.st-key-bar_plot_container,
.st-key-waffle_container{
    align-items: center;
}

/* stretch the map thumbnail to match the facts grid's height, then crop
   the image (not stretch it) to fill that height on any screen size */
div[data-testid="stHorizontalBlock"]:has(.st-key-map_container) {
    align-items: stretch;
}
div[data-testid="stHorizontalBlock"]:has(.st-key-map_container) [data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"]:has(.st-key-map_container) [data-testid="stElementContainer"],
div[data-testid="stHorizontalBlock"]:has(.st-key-map_container) [data-testid="stLayoutWrapper"] {
    height: 100%;
}
.st-key-map_container{
    height: 100% !important;
}
.st-key-map_container img{
    height: 100% !important;
    width: 100% !important;
    object-fit: cover;
    display: block;
}
"""
st.html(f"<style>{css}</style>")

##########################################################
########## LOAD DATA ####################################

# load final dataset
df_final = pd.read_csv("../data/final_datasets/df_clusters_milieuschutz.csv")
df_final["plr_id"] = df_final["plr_id"].astype(str).str.zfill(8)

# load precomputed profile table
plot_df = pd.read_csv("data/plot_df.csv", dtype={"plr_id": str})
plot_df["plr_id"] = plot_df["plr_id"].str.zfill(8)

# load watchlist
watchlist_df = pd.read_csv("../notebooks/Module2_classification/watchlist_display.csv", dtype={"plr_id": str})
watchlist_df["plr_id"] = watchlist_df["plr_id"].str.zfill(8)

# load building age shares
building_age_df = pd.read_csv("data/building_age_df.csv", dtype={"plr_id": str})
building_age_df["plr_id"] = building_age_df["plr_id"].str.zfill(8)

# load PLR geometries, only for the location map excerpt (cached)
@st.cache_data
def load_plr_geometries():
    plr_geo = gpd.read_file("plr_geometries.gpkg")
    plr_geo["plr_id"] = plr_geo["plr_id"].astype(str).str.zfill(8)
    return plr_geo[["plr_id", "plr_name", "geometry"]]

plr_geo = load_plr_geometries()

# cluster code -> readable label
cluster_labels = {
    1: "City core",
    3: "City belt",
    2: "Disadvantaged outskirts",
    0: "Affluent outskirts",
}

##########################################################
########## PLR SELECTION ################################

# group PLRs by district for the dropdown
grouped_by_bez = df_final.groupby("bez")

with st.container(key="white_container_selection", border=True):
    st.subheader("Profile of Planning Area (PLR)")
    selected_bez = st.selectbox(
        label="**Select a District:**",
        options=sorted(df_final["bez"].unique())
    )

    # PLRs belonging to the selected district only
    plrs_of_bez = grouped_by_bez.get_group(selected_bez)

    # id -> name mapping, scoped to this district
    plr_dict = dict(zip(plrs_of_bez["plr_id"], plrs_of_bez["plr_name"]))
    sorted_plr_ids = sorted(plr_dict.keys(), key=lambda x: plr_dict[x])

    selected_plr_id = st.selectbox(
        label="**Select a PLR (only PLRs with available data):**",
        options=sorted_plr_ids,
        format_func=lambda x: plr_dict[x],  # show the name, return the id
    )

############################################################
############## GENERAL DETAILS ##############################

# dropdown only offers PLRs from df_final, so this row always exists
selected_row = df_final.loc[df_final["plr_id"] == selected_plr_id].iloc[0]
matching_plot_df_rows = plot_df.loc[plot_df["plr_id"] == selected_plr_id]
matching_watchlist_df_rows = watchlist_df.loc[watchlist_df["plr_id"] == selected_plr_id]

PLR = selected_row["plr_name"]
plr_id = selected_row["plr_id"]
bez = selected_row["bez"]

# resident count, with fallback
if not matching_plot_df_rows.empty and pd.notna(matching_plot_df_rows.iloc[0]["res_count"]):
    res_count = int(matching_plot_df_rows.iloc[0]["res_count"])
else:
    res_count = "No data available"

# cluster + milieu protection status
cluster_status = cluster_labels.get(selected_row["cluster_4k"], "No data available")
ms_status = (
    "Proportion of milieu protected area more than 50% of PLR"
    if selected_row["ms_over50"] == 1
    else "Proportion of milieu protected area less than 50% of PLR"
)

# similarity score only applies to undesignated areas
if selected_row["ms_over50"] == 0:
    similarity_score = round(selected_row["oof_prob"], 2)
else:
    similarity_score = "Proportion of milieu protected area already more than 50% of PLR"

proportion_milieu = f'{round(selected_row["ms_portion"] * 100, 1)}%'
on_watchlist_flag = bool(selected_row["on_watchlist"])
watchlist = "On watchlist" if on_watchlist_flag else "Not on watchlist"

# watchlist rank only relevant if on the watchlist
if on_watchlist_flag and not matching_watchlist_df_rows.empty:
    selected_row_from_watchlist_df = matching_watchlist_df_rows.iloc[0]
    watchlist_rank = selected_row_from_watchlist_df["rank"]
    nearest_designated = selected_row_from_watchlist_df["nearest_designated"]
else:
    watchlist_rank = "Not on watchlist"
    nearest_designated = "n/a"

############################################################
########### PROFILE/CLUSTER DETAILS ########################

# cache the four cluster description texts
@st.cache_data
def load_profile_texts():
    texts = {}
    for name in ["city_belt", "city_core", "disadvantaged_outskirts", "affluent_outskirts"]:
        with open(f"texts/profile_{name}.md", encoding="utf-8") as f:
            texts[name] = f.read()
    return texts

profile_texts = load_profile_texts()

# cluster_status uses spaces, file keys use underscores
cluster_key = cluster_status.lower().replace(" ", "_")
cluster_profile_text = profile_texts.get(cluster_key, "No data available")

###########################################################
########## WATCHLIST DETAILS ##############################

# cache the three milieu-protection-status texts
@st.cache_data
def load_watchlist_texts():
    texts = {}
    for name in ["undesignated", "designated", "watchlist"]:
        with open(f"texts/profile_{name}.md", encoding="utf-8") as f:
            texts[name] = f.read()
    return texts

watchlist_texts = load_watchlist_texts()

# pick + fill the matching milieu-protection text
if selected_row["ms_over50"] == 1:
    watchlist_text = watchlist_texts["designated"]  # no placeholders
elif on_watchlist_flag:
    watchlist_text = watchlist_texts["watchlist"].format(
        rank=watchlist_rank,
        score=similarity_score,
        nearest_designated=nearest_designated,
    )
else:
    watchlist_text = watchlist_texts["undesignated"].format(
        score=similarity_score,
    )

############################################################
########### PROFILE PLOTS: BAR GRAPHS #######################

# plot_df column -> display label
labels_display = {
    "Rent_level_(€_m²)": "Rent level (€/m²)",
    "AirBnB_density_(per_1000_Apts.)": "AirBnB density (per 1000 Apts.)",
    "Share_of_single-parent_households": "Share of single-parent households",
    "Share_of_benefit_recipients": "Share of benefit recipients",
    "Business_exit_rate": "Business exit rate",
    "Share_of_upscale_gastronomy": "Share of upscale gastronomy",
}

# display label -> reference year
years_by_label = {
    "Rent level (€/m²)": "2025",
    "AirBnB density (per 1000 Apts.)": "2025",
    "Share of single-parent households": "2024",
    "Share of benefit recipients": "2024",
    "Business exit rate": "2026",
    "Share of upscale gastronomy": "2026",
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

# two variables per dimension
selected_vars = {
    "re":  ["Rent_level_(€_m²)", "AirBnB_density_(per_1000_Apts.)"],
    "soc": ["Share_of_single-parent_households", "Share_of_benefit_recipients"],
    "com": ["Business_exit_rate", "Share_of_upscale_gastronomy"],
}

# font sizes
dimension_fontsize = 16
var_title_fontsize = 16
legend_fontsize = 14
y_label_fontsize = 14

dims = ["re", "soc", "com"]

# 2x3 bar chart (PLR vs district/Berlin median)
def build_profile_figure(row):
    fig_profile, axes = plt.subplots(2, 3, figsize=(4.2 * 3, 4.5 * 2))

    for col, dim in enumerate(dims):
        for row_idx in range(2):
            ax = axes[row_idx][col]
            var = selected_vars[dim][row_idx]
            var_label = labels_display[var]

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
            ax.set_ylabel(var_label, fontsize=y_label_fontsize)

            if row_idx == 0:
                ax.text(0.5, 1.28, f"Dimension: {dimension_labels[dim]}",
                        transform=ax.transAxes, ha="center",
                        fontsize=dimension_fontsize, fontweight="bold",
                        color=dimension_colors[dim])
                ax.text(0.5, 1.05, f"{var_label} ({years_by_label[var_label]})",
                        transform=ax.transAxes, ha="center",
                        fontsize=var_title_fontsize, fontweight="normal",
                        color="black")
            else:
                ax.set_title(f"{var_label} ({years_by_label[var_label]})",
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


matching_plot_rows = plot_df.loc[plot_df["plr_id"] == selected_plr_id]

if matching_plot_rows.empty:
    fig_profile = None
    st.info("No detailed data available for this planning area.")
else:
    profile_row = matching_plot_rows.iloc[0]
    fig_profile = build_profile_figure(profile_row)

#############################################################
############## WAFFLE PLOT ###################################

# building age shares for the selected PLR
matching_building_age_rows = building_age_df.loc[building_age_df["plr_id"] == selected_plr_id]

if matching_building_age_rows.empty:
    fig_waffle = None
else:
    plr_building_age = matching_building_age_rows.iloc[0]

    # apartment count
    if not matching_plot_df_rows.empty:
        n_apartments = int(matching_plot_df_rows.iloc[0]["n_apartments"])
    else:
        n_apartments = 0

    # group into three age bands
    pre_war = (
        plr_building_age["re_anteil_vor_1919"]
        + plr_building_age["re_anteil_y1919_1948"]
    )
    post_war = (
        plr_building_age["re_anteil_y1949_1978"]
        + plr_building_age["re_anteil_y1979_1990"]
        + plr_building_age["re_anteil_y1991_2000"]
        + plr_building_age["re_anteil_y2001_2010"]
    )
    new_construction = plr_building_age["re_anteil_y2011_plus"]

    # convert to whole percentages summing to 100
    values = [round(pre_war * 100), round(post_war * 100), round(new_construction * 100)]
    values[-1] += 100 - sum(values)

    waffle_data = {
        "Pre-war (<1949)": values[0],
        "Post-war (1949-2010)": values[1],
        "New construction (2011+)": values[2],
    }

    # waffle chart figure
    fig_waffle = plt.figure(
        FigureClass=Waffle,
        rows=10,
        values=waffle_data,
        colors=["#704B24", "#A06B34", "#E3D3C2"],
        title={
            "label": f"{n_apartments:,} apartments (each square = 1%)",
            "loc": "center",
            "fontsize": 14,
            "fontweight": "bold",
            "color": "black",
            "pad": 10,
        },
        legend={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.05),
            "ncol": 1,
            "framealpha": 0,
            "fontsize": 11,
        },
        figsize=(7, 7.5),
        tight={"pad": 2.5, "rect": (0, 0, 1, 0.93)},
    )

    # colored header
    waffle_ax = fig_waffle.axes[0]
    waffle_ax.text(
        0.5, 1.10, f"Dimension: {dimension_labels['re']}",
        transform=waffle_ax.transAxes, ha="center",
        fontsize=dimension_fontsize, fontweight="bold",
        color=dimension_colors["re"],
    )


#############################################################
############### PROFILE CONTAINER ############################

# bar chart description text
BAR_CHART_DESCRIPTION = (
    "Each planning area (PLR) is characterised across the three dimensions — real estate, social, and commercial — "
    "with every indicator shown against two reference lines: the median of its district (grey) and of Berlin as a whole (black). "
    "This dual benchmark places each PLR both in its local and in its city-wide context. "
    "The variables are chosen to capture the mechanisms through which gentrification becomes visible at the neighbourhood level. "
    "In the real estate dimension, rent level (€/m²) tracks the price pressure that drives displacement, while Airbnb density (listings per 1,000 apartments) "
    "measures the withdrawal of housing from the regular market through short-term letting — an early and spatially concentrated signal of touristic upgrading. "
    "The social dimension captures displacement pressure on vulnerable residents: the share of benefit recipients and of single-parent households "
    "identify two groups that are very exposed to displacement. The commercial dimension reflects the transformation of the local economy: "
    "the business exit rate captures the turnover and closure of established businesses, and the share of upscale gastronomy indicates the "
    "commercial upgrading that typically accompanies — and reinforces — residential gentrification."
)

# waffle chart description text
WAFFLE_CHART_DESCRIPTION = (
    "This waffle chart shows the age structure of the housing stock in this planning area, split into three "
    "construction eras: pre-war (built before 1949), post-war (1949-2010), and new construction (2011 onwards). "
    "Each square represents one percent of the area's apartments, with the total apartment count shown in the title. "
    "Building age is a useful proxy for gentrification potential: a high pre-war share often marks the characterful "
    "older stock — high ceilings, ornate facades, central locations — that tends to attract renovation and rising "
    "prices, the same building fabric that defines the City belt profile. A high share of new construction, by "
    "contrast, points to an area shaped more by recent development than by neighbourhood change."
)

# location map fixed
MAP_THUMB_WIDTH_IN = 6.0
MAP_THUMB_HEIGHT_IN = 3.0

# builds a small real-map excerpt (OSM tiles) around the selected PLR
@st.cache_data(show_spinner=False)
def build_location_map(plr_id):
    row = plr_geo.loc[plr_geo["plr_id"] == plr_id]
    if row.empty:
        return None

    fig, ax = plt.subplots(figsize=(MAP_THUMB_WIDTH_IN, MAP_THUMB_HEIGHT_IN))
    row.plot(ax=ax, color="#8B0000", alpha=0.2, zorder=2)
    row.boundary.plot(ax=ax, color="#8B0000", linewidth=2, zorder=3)

    # zoom out a bit for street context
    minx, miny, maxx, maxy = row.total_bounds
    center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
    span_x = (maxx - minx) * 1.6
    span_y = (maxy - miny) * 1.6

    # match crop aspect to figure aspect so the map itself isn't distorted
    # (CSS crops the rendered image further to fit the actual box height)
    fig_aspect = MAP_THUMB_HEIGHT_IN / MAP_THUMB_WIDTH_IN
    box_aspect = span_y / span_x if span_x else fig_aspect
    if box_aspect < fig_aspect:
        span_y = span_x * fig_aspect
    else:
        span_x = span_y / fig_aspect

    ax.set_xlim(center_x - span_x / 2, center_x + span_x / 2)
    ax.set_ylim(center_y - span_y / 2, center_y + span_y / 2)
    ax.set_aspect("equal")

    # real OSM basemap, falls back to a plain outline without internet
    try:
        cx.add_basemap(ax, crs=plr_geo.crs, source=cx.providers.OpenStreetMap.Mapnik, attribution=False)
    except Exception:
        pass

    ax.set_axis_off()
    fig.tight_layout(pad=0)
    return fig

# render the profile
with st.container(key="white_container_profile", border=True):
    with st.container(key="profile_textbox", border=False):
        st.markdown(f"### {PLR}")

        facts_col, map_col = st.columns([2, 1], gap="small")

        # facts
        with facts_col:
            profile_rows = {
                "Identification number": plr_id,
                "District": bez,
                "Resident count (12/2025)": res_count,
                "Milieu protection status": ms_status,
                "% of milieu protection": proportion_milieu,
                "Gentrification profile": cluster_status,
                "Watchlist status": watchlist,
            }
            rows_html = "".join(
                f'<div style="font-weight:600;">{label}</div><div>{value}</div>'
                for label, value in profile_rows.items()
            )
            st.markdown(
                f"""
                <div style="display:grid; grid-template-columns:auto 1fr; column-gap:12px; row-gap:8px;">
                    {rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        # map on the side
        with map_col:
            with st.container(key="map_container", border=False):
                fig_location = build_location_map(selected_plr_id)
                if fig_location is not None:
                    st.pyplot(fig_location, width="stretch")

        st.markdown("")
        st.markdown(cluster_profile_text)
        st.markdown("")
        st.markdown(watchlist_text)

    # bar chart + description
    with st.container(key="plot_container", border=False):
        bar_col, bar_desc_col = st.columns([3, 2], gap="small")
        with bar_col:
            with st.container(key="bar_plot_container", border=False):
                if fig_profile is not None:
                    st.pyplot(fig_profile, width="stretch")
        with bar_desc_col:
            with st.container(key="bar_description_box", border=False):
                st.markdown(f"##### Description of Bar Plots \n\n {BAR_CHART_DESCRIPTION}", text_alignment="justify")

        # waffle chart + description
        waffle_col, waffle_desc_col = st.columns([0.7, 2], gap="small")
        with waffle_col:
            with st.container(key="waffle_container", border=False):
                if fig_waffle is not None:
                    st.pyplot(fig_waffle, width="stretch")
        with waffle_desc_col:
            with st.container(key="waffle_description_box", border=False):
                st.markdown(f"##### Description of Waffle Chart \n\n {WAFFLE_CHART_DESCRIPTION}", text_alignment="justify")


############################################################
########### DOWNLOAD PROFILE #################################

# reuse matplotlib's bundled Unicode font (Helvetica can't do em dashes etc.)
DEJAVU_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
DEJAVU_REGULAR = os.path.join(DEJAVU_DIR, "DejaVuSans.ttf")
DEJAVU_BOLD = os.path.join(DEJAVU_DIR, "DejaVuSans-Bold.ttf")

# turn "- **" bullets into plain bold lines for fpdf's markdown mode
def clean_body_for_pdf(text):
    return text.replace("- **", "**").replace("·", "-")

# renders a markdown block as a real bold heading + body
def add_markdown_section(pdf, markdown_text, title_size=13, body_size=10):
    lines = markdown_text.strip().split("\n")
    if lines and lines[0].lstrip().startswith("#"):
        title = lines[0].lstrip("#").strip()
        body_lines = lines[1:]
    else:
        title = None
        body_lines = lines

    if title:
        pdf.set_font("DejaVu", "B", title_size)
        pdf.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    pdf.set_font("DejaVu", "", body_size)
    body_text = "\n".join(body_lines).strip()
    pdf.multi_cell(0, 6, clean_body_for_pdf(body_text), markdown=True)

# builds a one-PLR PDF export
def build_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", DEJAVU_REGULAR)
    pdf.add_font("DejaVu", "B", DEJAVU_BOLD)

    # page 1: key facts + profile texts
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, f"Profile: {PLR} ({plr_id})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("DejaVu", "", 11)
    for label, value in profile_rows.items():
        pdf.cell(0, 7, f"{label}: {value}", new_x="LMARGIN", new_y="NEXT")

    if fig_location is not None:
        pdf.ln(4)
        buf = io.BytesIO()
        fig_location.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        pdf.image(buf, x=10, w=80)

    pdf.ln(8)
    add_markdown_section(pdf, cluster_profile_text)
    pdf.ln(8)
    add_markdown_section(pdf, watchlist_text)

    # page 2: bar chart
    if fig_profile is not None:
        pdf.add_page()
        buf = io.BytesIO()
        fig_profile.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        pdf.image(buf, x=10, w=190)
        pdf.ln(4)
        pdf.set_font("DejaVu", "B", 13)
        pdf.cell(0, 8, "Description", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("DejaVu", "", 10)
        pdf.multi_cell(0, 6, BAR_CHART_DESCRIPTION)

    # page 3: waffle chart
    if fig_waffle is not None:
        pdf.add_page()
        buf = io.BytesIO()
        fig_waffle.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        waffle_pdf_width = 90
        page_width = 210  # A4
        x_centered = (page_width - waffle_pdf_width) / 2
        pdf.image(buf, x=x_centered, w=waffle_pdf_width)
        pdf.ln(4)
        pdf.set_font("DejaVu", "B", 13)
        pdf.cell(0, 8, "Description", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("DejaVu", "", 10)
        pdf.multi_cell(0, 6, WAFFLE_CHART_DESCRIPTION)

    return bytes(pdf.output())

pdf_bytes = build_pdf()

st.download_button(
    label="Download profile",
    data=pdf_bytes,
    file_name=f"{PLR}.pdf",
    mime="application/pdf",
    type="primary",
    icon=":material/download:",
)