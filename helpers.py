# helpers.py
import pandas as pd, numpy as np
import streamlit as st
import plotly.graph_objects as go

COLORS = {
    "primary":  "#081B48",
    "secondary":"#519CDD",
    "background":"#FFFFFF"
}

def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans', Arial, sans-serif !important; }
    </style>
    """, unsafe_allow_html=True)

def apply_plotly_style(fig: go.Figure, title:str=None):
    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font=dict(family="Noto Sans, Arial", color=COLORS["primary"]),
        title=dict(text=title or "", x=0.0, xanchor="left", font=dict(size=22))
    )
    return fig

@st.cache_data(show_spinner=False)
def load_data(path="data/analyse_donnees.xlsx"):
    df = pd.read_excel(path, sheet_name="Feuil2")
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0":"Ticker"})
    df = df.dropna(how="all").drop_duplicates(subset="Ticker")
    df = df.replace([np.inf, -np.inf], np.nan)
    indice = df.iloc[[0]].copy()
    actions = df.iloc[1:].copy().reset_index(drop=True)
    return actions, indice

def get_filtered(actions: pd.DataFrame, sector=None, country=None, esg_rule=None):
    df = actions.copy()
    if sector:
        df = df[df["Sector (1)"].isin(sector)]
    if country:
        df = df[df["Risk Country"].isin(country)]
    if esg_rule == "ESG >= médiane secteur" and "ESG Score" in df.columns:
        med = df.groupby("Sector (1)")["ESG Score"].median()
        df = df[df["ESG Score"] >= df["Sector (1)"].map(med)]
    return df.reset_index(drop=True)
