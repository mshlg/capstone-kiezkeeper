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
from pywaffle import Waffle

# set page to wide format
st.set_page_config(layout="wide")

# set title
st.logo('kiezkeeper_vector_logo.svg', size="large")
st.markdown("# KiezKeeper :small[Data-Driven Detection of Gentrification in Berlin]")
st.markdown("***")

# set style for containers
css = """
.st-key-white_container_selection,
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
########## PLR SELECTION ###################################

# load final dataset
df_final = pd.read_csv("../data/final_datasets/df_clusters_milieuschutz.csv")
df_final["plr_id"] = df_final["plr_id"].astype(str).str.zfill(8)

# group plrs by bez:
grouped_by_bez = df_final.groupby("bez")

with st.container(key="white_container_selection", border=True):
        st.subheader("Profile of Planning Area (PLR)")
        selected_bez = st.selectbox(
            label="Select a District",
            options=sorted(df_final["bez"].unique())
        )

        # only the PLRs belonging to the selected district
        plrs_of_bez = grouped_by_bez.get_group(selected_bez)

        # id -> name mapping, but scoped to just this district
        plr_dict = dict(zip(plrs_of_bez["plr_id"], plrs_of_bez["plr_name"]))
        sorted_plr_ids = sorted(plr_dict.keys(), key=lambda x: plr_dict[x])

        selected_plr_id = st.selectbox(
            label="Select a Planning Area",
            options=sorted_plr_ids,
            format_func=lambda x: plr_dict[x],   # shows the name, but returns the id
        )

############################################################
########### PROFILE TEXT ##################################

with st.container(key="white_container_profile", border=True):
        with st.container(key="profile_textbox", border=False):
                st.markdown(f"#### Profile for {selected_plr_id}")






############################################################
########### PROFILE PLOTS ##################################



############################################################
########### DOWNLOAD PROFILE ###############################


# # Download button
#         csv = df_final.to_csv().encode("utf-8")

#         st.download_button(
#             label="Download profile",
#             data=csv,
#             file_name=f"{PLR}.csv",
#             mime="text/csv",
#             type="primary",
#             icon=":material/download:"
#         )