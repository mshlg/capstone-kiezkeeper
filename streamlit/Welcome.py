# import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# white textbox
css = """
.st-key-white_textbox{
    background: rgba(255, 255, 255);
    padding: 16px;
"""
st.html(f"<style>{css}</style>")


# Set title and head page
st.logo('kiezkeeper_vector_logo.svg', size="large")
st.title("Welcome to ...")
st.image('kiezkeeper_vector_real_cut.svg')
st.markdown("***")

st.markdown(
    """
    <div style="font-size: 1.3rem; line-height: 1.6;">

    **The problem**: In Berlin, tenant protection against displacement (known as Milieuschutz, §172) is patchwork,
    depending both on political will and a long administrative process, and therefore usually arrives late.
    By the time the process recognises that a neighbourhood is changing, much of that change has already happened.
    And because each of the city's twelve districts decides on its own, there has never been one shared, city-wide picture of which areas are under pressure.

    **KiezKeeper turns this around**: instead of reacting after the fact, it looks across all of Berlin at once.
    It learns what the city's already-protected neighbourhoods have in common on three dimensions: their social make-up,
    their housing market, and their local business landscape. From these patterns, KiezKeeper does two things:
    it sorts every neighbourhood into a profile that captures what kind of area it is, and it flags the unprotected ones whose profile most closely resembles the areas already under protection.

    **Together this gives Berlin a single, consistent view: turning twelve separate district perspectives into one shared, city-wide map.**

    **What it is — and what it isn't**: KiezKeeper measures resemblance, not risk. An area on the watchlist looks like the neighbourhoods
    a Berlin district has already chosen to protect — that is a reason to take a closer look, not proof that displacement is happening,
    and not a judgement about the people who live there. The aim is not to measure change once it is over.
    It is to give the whole city a way to act while there is still a neighbourhood to keep.

    </div>
    """,
    unsafe_allow_html=True,
)