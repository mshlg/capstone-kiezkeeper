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

plr_dict = dict(zip(df_final["plr_id"], df_final["plr_name"]))

# sort plrs alphabetically
sorted_plr_ids = sorted(plr_dict.keys(), key=lambda x: plr_dict[x])

with st.container(key="white_container_profile", border=True):
        st.subheader("Profile of Planning Area")
        selected_plr_id = st.selectbox(
            label="Select a Planning Area",
            options=sorted_plr_ids,
            format_func=lambda x: plr_dict[x],   # shows the name, but returns the id
        )