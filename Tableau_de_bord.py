import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from pathlib import Path

from helpers import load_data, inject_theme, apply_plotly_style, get_filtered, COLORS

# === Configuration et thème ===
st.set_page_config(page_title="SX5E – Dashboard", layout="wide")
inject_theme()

import base64
from pathlib import Path

# === Logo dans la sidebar ===
def place_logo_sidebar(path: str = "logo.jpeg", height_px: int = 110):
    p = Path(path)
    if not p.exists():
        st.warning(f"Logo introuvable : {p.resolve()}")
        return
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    ext = p.suffix.replace(".", "") or "png"

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebar"] {{
                position: relative;
            }}
            [data-testid="stSidebar"]::before {{
                content: "";
                display: block;
                background-color: white;        /* ✅ fond blanc autour du logo */
                background-image: url("data:image/{ext};base64,{b64}");
                background-repeat: no-repeat;
                background-size: contain;
                background-position: center;
                height: {height_px + 20}px;      /* un peu plus haut pour le cadre */
                margin: 15px 10px 0px 10px;
                border-radius: 10px;             /* ✅ coins arrondis sur le cadre */
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);  /* ✅ ombre douce */
                padding: 8px;                    /* espace entre logo et bord */
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
# --- appel ---
place_logo_sidebar("logo.jpeg", height_px=110)

# === Titre principal ===
st.title("Tableau de bord")

st.markdown("<br>", unsafe_allow_html=True)

# === Chargement des données ===
actions, indice = load_data("analyse données.xlsx")

# === Filtres ===
with st.sidebar:
    st.header("Filtres")
    sectors = sorted(actions["Sector (1)"].dropna().unique().tolist())
    countries = sorted(actions["Risk Country"].dropna().unique().tolist())
    s_pick = st.multiselect("Secteurs", sectors)
    c_pick = st.multiselect("Pays", countries)
    esg = st.selectbox("Règle ESG", ["Aucune", "ESG >= médiane secteur"], index=0)
    st.divider()
    st.markdown("Par Maxence Chachuat pour l'équipe ESR d'Amundi")


fdf = get_filtered(actions, s_pick, c_pick, esg if esg != "Aucune" else None)
universe = fdf.copy()

# === Colonnes principales ===
COL_RET = "1 Year Total Return - Previous"
COL_VOL = "Volatility 360 Day Calc"
COL_ESG = "ESG Score"
COL_UP  = "Upside with Target Price from Analyst"

# Conversion Upside en %
if COL_UP in universe.columns and universe[COL_UP].notna().any():
    p95 = np.nanpercentile(pd.to_numeric(universe[COL_UP], errors="coerce"), 95)
    if p95 <= 5:
        universe[COL_UP] = pd.to_numeric(universe[COL_UP], errors="coerce") * 100

# === Fonctions utilitaires ===
def nanmean(s): return float(np.nanmean(pd.to_numeric(s, errors="coerce")))
def nanmedian(s): return float(np.nanmedian(pd.to_numeric(s, errors="coerce")))

# === KPIs ===
ret_mean,  ret_med  = nanmean(universe[COL_RET]), nanmedian(universe[COL_RET])
vol_mean,  vol_med  = nanmean(universe[COL_VOL]), nanmedian(universe[COL_VOL])
esg_mean,  esg_med  = (nanmean(universe[COL_ESG]) if COL_ESG in universe else np.nan,
                       nanmedian(universe[COL_ESG]) if COL_ESG in universe else np.nan)
sharpe_series = (pd.to_numeric(universe[COL_RET], errors="coerce") /
                 pd.to_numeric(universe[COL_VOL], errors="coerce")).replace([np.inf, -np.inf], np.nan)
sharpe_mean, sharpe_med = float(np.nanmean(sharpe_series)), float(np.nanmedian(sharpe_series))

# === Tableau KPI ===
kpi_table = pd.DataFrame({
    "Indicateur": ["Rendement 1 an", "Volatilité 1 an", "Score ESG", "Sharpe Proxy"],
    "Moyenne":    [f"{ret_mean:.2f} %", f"{vol_mean:.2f} %",
                   (f"{esg_mean:.2f}" if not np.isnan(esg_mean) else "—"),
                   f"{sharpe_mean:.2f}"],
    "Médiane":    [f"{ret_med:.2f} %", f"{vol_med:.2f} %",
                   (f"{esg_med:.2f}" if not np.isnan(esg_med) else "—"),
                   f"{sharpe_med:.2f}"],
})

fig_kpi = go.Figure(data=[go.Table(
    header=dict(
        values=["<b>Indicateur</b>", "<b>Moyenne</b>", "<b>Médiane</b>"],
        fill_color=COLORS["primary"],
        font=dict(color="white", family="Noto Sans, Arial"),
        align="left",
        height=30
    ),
    cells=dict(
        values=[kpi_table[c] for c in ["Indicateur", "Moyenne", "Médiane"]],
        fill_color=COLORS["background"],
        font=dict(color=COLORS["primary"], family="Noto Sans, Arial", size=13),
        align="left",
        height=28
    )
)])
fig_kpi = apply_plotly_style(fig_kpi, title="KPI - Vue d'ensemble (univers SX5E)")
fig_kpi.update_layout(margin=dict(l=0, r=0, t=50, b=5), height=200)
st.plotly_chart(fig_kpi, use_container_width=True)

st.divider()

# === 1) Carte Risque / Rendement ===
fig_risk_return = px.scatter(
    universe,
    x=COL_VOL, y=COL_RET,
    color="Sector (1)",
    hover_data={
        "Ticker": True,
        "Sector (1)": True,
        "Risk Country": True,
        COL_RET: ":.2f",
        COL_VOL: ":.2f",
        COL_UP: ":.2f" if COL_UP in universe else False
    },
    template="plotly_white",
    color_discrete_sequence=px.colors.sequential.Blues,
)
x_med, y_med = float(universe[COL_VOL].median()), float(universe[COL_RET].median())
fig_risk_return.add_hline(y=y_med, line_dash="dot", line_color="rgba(0,0,0,0.25)")
fig_risk_return.add_vline(x=x_med, line_dash="dot", line_color="rgba(0,0,0,0.25)")
fig_risk_return.update_traces(marker=dict(size=9, line=dict(width=0)))
fig_risk_return = apply_plotly_style(fig_risk_return, title="Carte Risque / Rendement")
st.plotly_chart(fig_risk_return, use_container_width=True)

st.divider()

# === 2) Répartition par secteur ===
sect_col = "Sector (1)"
tmp = (universe
    .assign(**{sect_col: universe[sect_col].fillna("Inconnu")})
    .groupby(sect_col, as_index=False)
    .agg(n_titles=("Ticker", "count"))
    .sort_values("n_titles", ascending=False)
)

TOP_N = 12
if len(tmp) > TOP_N:
    top = tmp.head(TOP_N - 1)
    other = pd.DataFrame({sect_col: ["Autres"], "n_titles": [tmp.iloc[TOP_N-1:]["n_titles"].sum()]})
    tmp = pd.concat([top, other], ignore_index=True)

fig_sector_donut = px.pie(
    tmp,
    names=sect_col,
    values="n_titles",
    hole=0.6,
    color=sect_col,
    color_discrete_sequence=px.colors.sequential.Blues[::-1],
)
fig_sector_donut.update_traces(
    textposition="inside",
    textinfo="label+percent",
    hovertemplate="<b>%{label}</b><br>Nombre d'actifs: %{value}<br>Part: %{percent}<extra></extra>"
)
fig_sector_donut = apply_plotly_style(fig_sector_donut, title="Répartition par secteur")
fig_sector_donut.update_layout(showlegend=False, margin=dict(l=0, r=0, t=50, b=5), height=420)
st.plotly_chart(fig_sector_donut, use_container_width=True)

st.divider()

# === 3) Top / Bottom 5 ===
fdf = universe.copy()
top5 = fdf.nlargest(5, COL_RET).copy()
bottom5 = fdf.nsmallest(5, COL_RET).copy()
top5["Groupe"], bottom5["Groupe"] = "Top 5", "Bottom 5"
tb = pd.concat([top5, bottom5], ignore_index=True)
tb = tb.sort_values(by=["Groupe", COL_RET], ascending=[True, True])

color_map = {"Top 5": COLORS["secondary"], "Bottom 5": COLORS["primary"]}
tb["Color"] = tb["Groupe"].map(color_map)

fig_top_bottom = go.Figure()
for grp in ["Top 5", "Bottom 5"]:
    sub = tb[tb["Groupe"] == grp]
    fig_top_bottom.add_trace(go.Bar(
        x=sub[COL_RET],
        y=sub["Ticker"],
        orientation="h",
        name=grp,
        marker_color=color_map[grp],
        customdata=np.stack([sub["Sector (1)"], sub["Risk Country"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Secteur: %{customdata[0]}<br>Pays: %{customdata[1]}<br>Rendement 1Y: %{x:.2f}%<extra></extra>"
    ))
fig_top_bottom.update_layout(
    barmode="relative",
    xaxis_title="Rendement 1 an (%)",
    yaxis_title="Ticker",
    legend_title_text="Groupe",
)
fig_top_bottom = apply_plotly_style(fig_top_bottom, title="Top 5 / Bottom 5 - Rendement 1 an")
st.plotly_chart(fig_top_bottom, use_container_width=True)

st.divider()

# === 4) Synthèse "5 titres à regarder" ===
fdf["Sharpe_proxy"] = (fdf[COL_RET] / fdf[COL_VOL]).replace([np.inf, -np.inf], np.nan)
if "BEst P/E Ratio" in fdf and "Price / Earnings - 5 Year Average" in fdf:
    fdf["PE_premium_5Y_%"] = (fdf["BEst P/E Ratio"] - fdf["Price / Earnings - 5 Year Average"]) / fdf["Price / Earnings - 5 Year Average"] * 100


shortlist = []
def add_best(df, sort_col, ascending, label):
    cand = df.dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending)
    for _, row in cand.iterrows():
        if row["Ticker"] not in [s["Ticker"] for s in shortlist]:
            shortlist.append({
                "Raison": label,
                "Ticker": row["Ticker"],
                "Secteur": row.get("Sector (1)", None),
                "Pays": row.get("Risk Country", None),
                "Rendement 1Y (%)": row.get(COL_RET, np.nan),
                "Vol 360j (%)": row.get(COL_VOL, np.nan),
                "Sharpe proxy": row.get("Sharpe_proxy", np.nan),
                "ESG": row.get(COL_ESG, np.nan),
                "Upside (%)": row.get(COL_UP, np.nan),
            })
            break


add_best(fdf, COL_RET, False, "Meilleur rendement 1Y")
add_best(fdf, "Sharpe_proxy", False, "Meilleur ratio rendement/risque")
if COL_ESG in fdf: add_best(fdf, COL_ESG, False, "Meilleur score ESG                ")
if "PE_premium_5Y_%" in fdf: add_best(fdf, "PE_premium_5Y_%", True, "Plus forte décote P/E vs 5Y")
if COL_UP in fdf: add_best(fdf, COL_UP, False, "Plus fort Upside analystes")


summary5_df = pd.DataFrame(shortlist)


# Mise en forme des pourcentages
for c in ["Rendement 1Y (%)", "Vol 360j (%)", "Upside (%)"]:
    if c in summary5_df:
        summary5_df[c] = summary5_df[c].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

# Mise en forme du Sharpe proxy (arrondi à 2 décimales)
if "Sharpe proxy" in summary5_df:
    summary5_df["Sharpe proxy"] = summary5_df["Sharpe proxy"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")


fig_summary5 = go.Figure(data=[go.Table(
    header=dict(
        values=[f"<b>{h}</b>" for h in summary5_df.columns],
        fill_color=COLORS["primary"],
        font=dict(color="white", family="Noto Sans, Arial"),
        align="left",
        height=28
    ),
    cells=dict(
        values=[summary5_df[c] for c in summary5_df.columns],
        fill_color=COLORS["background"],
        font=dict(color=COLORS["primary"], family="Noto Sans, Arial", size=13),
        align="left",
        height=26
    )
)])
fig_summary5 = apply_plotly_style(fig_summary5, title="Synthèse - 5 titres à regarder")
fig_summary5.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=360)
st.plotly_chart(fig_summary5, use_container_width=True)
