import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helpers import load_data, get_filtered, inject_theme, apply_plotly_style, COLORS

inject_theme()
st.title("Analyse extra-financière (ESG)")

st.markdown("<br>", unsafe_allow_html=True)


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

# --- Données & filtres ---
actions, indice = load_data("data/analyse données.xlsx")
with st.sidebar:
    st.header("Filtres")
    sectors = sorted(actions["Sector (1)"].dropna().unique())
    countries = sorted(actions["Risk Country"].dropna().unique())
    s_pick = st.multiselect("Secteurs", sectors)
    c_pick = st.multiselect("Pays", countries)
    esg_rule = st.selectbox("Règle ESG", ["Aucune", "ESG >= médiane secteur"], index=0)
    st.divider()
    st.markdown("Par Maxence Chachuat pour l'équipe ESR d'Amundi")

fdf = get_filtered(actions, s_pick, c_pick, esg_rule if esg_rule != "Aucune" else None)

COL_ESG   = "ESG Score"
COL_SECT  = "Sector (1)"
COL_RET   = "1 Year Total Return - Previous"
COL_VOL   = "Volatility 360 Day Calc"
COL_UP    = "Upside with Target Price from Analyst"
COL_CTRY  = "Risk Country"


st.markdown("L'analyse révèle une déconnexion totale entre les critères ESG et les mécanismes de création de valeur (rendement, risque, upside analyst). Contrairement aux hypothèses de finance durable, les entreprises de ce jeu de données avec la meilleure gouvernance n'offrent ni réduction de volatilité, ni surperformance, ni support analystes. Au contraire, certains clusters majeurs concentrent des titres ignorant les critères ESG mais affichant un potentiel d'upside élevé, suggérant que l'ESG reste avant tout un filtre de conformité normative sans impact sur les fondamentaux de marché. L'Immobilier et la Technologie constituent les exceptions avec des scores ESG élevés, tandis que les secteurs traditionnels (Matériaux, Santé) restent largement sous-équipés en gouvernance.")   

# =========================
# 5.1) Distribution des scores ESG (barres affinées)
# =========================
if COL_ESG in fdf.columns and fdf[COL_ESG].notna().any():
    df = fdf[[COL_ESG]].dropna().copy()
    s = pd.to_numeric(df[COL_ESG], errors="coerce")

    fig_esg_dist = px.histogram(
        df, x=COL_ESG, nbins=40,
        template="plotly_white",
        color_discrete_sequence=[COLORS["secondary"]]
    )
    fig_esg_dist.update_traces(marker_line_width=0, opacity=0.85)
    fig_esg_dist.add_vline(x=float(s.mean()),   line_dash="dash", line_color=COLORS["primary"])
    fig_esg_dist.add_vline(x=float(s.median()), line_dash="dot",  line_color="rgba(0,0,0,0.35)")

    fig_esg_dist.update_layout(
        xaxis_title="Score ESG",
        yaxis_title="Effectif",
        margin=dict(l=0, r=0, t=60, b=50),
        height=420,
        bargap=0.25
    )
    fig_esg_dist = apply_plotly_style(fig_esg_dist, title="Distribution des scores ESG (barres affinées)")
    st.plotly_chart(fig_esg_dist, use_container_width=True)
else:
    st.info("Aucun score ESG disponible après filtres.")

st.divider()

# =========================
# 5.2) Score ESG médian par secteur
# =========================
if all(c in fdf.columns for c in [COL_ESG, COL_SECT]) and fdf[COL_ESG].notna().any():
    df = fdf[[COL_ESG, COL_SECT]].dropna().copy()
    df[COL_SECT] = df[COL_SECT].fillna("Inconnu")

    agg = (
        df.groupby(COL_SECT, as_index=False)
          .agg(esg_median=(COL_ESG, "median"), esg_mean=(COL_ESG, "mean"), n=(COL_ESG, "count"))
          .sort_values("esg_median", ascending=False)
    )

    fig_esg_sector = px.bar(
        agg, x=COL_SECT, y="esg_median",
        template="plotly_white",
        color_discrete_sequence=[COLORS["primary"]],
        hover_data={"esg_mean":":.2f","n":True}
    )
    fig_esg_sector.update_layout(
        xaxis_title="Secteur",
        yaxis_title="Score ESG médian",
        margin=dict(l=0, r=0, t=60, b=120),
        height=520
    )
    fig_esg_sector = apply_plotly_style(fig_esg_sector, title="Score ESG médian par secteur")
    st.plotly_chart(fig_esg_sector, use_container_width=True)
else:
    st.info("Impossible d’agréger l’ESG par secteur (colonnes manquantes ou vides).")

st.divider()

# =========================
# 5.3) ESG vs performance (avec tendance)
# =========================
if all(c in fdf.columns for c in [COL_ESG, COL_RET, COL_SECT]):
    df = fdf[[COL_ESG, COL_RET, COL_SECT, "Ticker", COL_CTRY]].dropna().copy()
    df[COL_SECT] = df[COL_SECT].fillna("Inconnu")

    if len(df) >= 2:
        x = pd.to_numeric(df[COL_ESG], errors="coerce").to_numpy()
        y = pd.to_numeric(df[COL_RET], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[m], y[m]
        slope, intercept = (np.polyfit(xv, yv, 1) if len(xv) >= 2 else (0.0, float(np.nanmean(yv)) if len(yv) else 0.0))
        R = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) >= 2 else np.nan

        fig_esg_ret = px.scatter(
            df, x=COL_ESG, y=COL_RET, color=COL_SECT,
            hover_data={"Ticker":True, COL_CTRY:True, COL_SECT:True, COL_ESG:":.2f", COL_RET:":.2f"},
            template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_esg_ret.update_traces(marker=dict(size=9, opacity=0.85))

        if len(xv) >= 2:
            x_line = np.linspace(xv.min(), xv.max(), 100)
            y_line = slope * x_line + intercept
            fig_esg_ret.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=COLORS["primary"], width=2, dash="dash"),
                hoverinfo="skip", showlegend=False
            ))

        r_text = f"(R = {R:.2f})" if np.isfinite(R) else ""
        fig_esg_ret.update_layout(
            xaxis_title="Score ESG",
            yaxis_title="Rendement 1 an (%)",
            margin=dict(l=0, r=0, t=70, b=40),
            height=480
        )
        fig_esg_ret = apply_plotly_style(fig_esg_ret, title=f"ESG vs performance financière {r_text}")
        st.plotly_chart(fig_esg_ret, use_container_width=True)
    else:
        st.info("Échantillon insuffisant pour tracer ESG vs performance.")
else:
    st.info("Colonnes nécessaires absentes (ESG, Retour, Secteur).")

st.divider()

# =========================
# 5.4) ESG vs volatilité (avec tendance)
# =========================
if all(c in fdf.columns for c in [COL_ESG, COL_VOL, COL_SECT]):
    df = fdf[[COL_ESG, COL_VOL, COL_SECT, "Ticker", COL_CTRY]].dropna().copy()
    df[COL_SECT] = df[COL_SECT].fillna("Inconnu")

    if len(df) >= 2:
        x = pd.to_numeric(df[COL_ESG], errors="coerce").to_numpy()
        y = pd.to_numeric(df[COL_VOL], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[m], y[m]
        slope, intercept = (np.polyfit(xv, yv, 1) if len(xv) >= 2 else (0.0, float(np.nanmean(yv)) if len(yv) else 0.0))
        R = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) >= 2 else np.nan

        fig_esg_vol = px.scatter(
            df, x=COL_ESG, y=COL_VOL, color=COL_SECT,
            hover_data={"Ticker":True, COL_CTRY:True, COL_SECT:True, COL_ESG:":.2f", COL_VOL:":.2f"},
            template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_esg_vol.update_traces(marker=dict(size=9, opacity=0.85))

        if len(xv) >= 2:
            x_line = np.linspace(xv.min(), xv.max(), 100)
            y_line = slope * x_line + intercept
            fig_esg_vol.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=COLORS["primary"], width=2, dash="dash"),
                hoverinfo="skip", showlegend=False
            ))

        r_text = f"(R = {R:.2f})" if np.isfinite(R) else ""
        fig_esg_vol.update_layout(
            xaxis_title="Score ESG",
            yaxis_title="Volatilité 1 an (%)",
            margin=dict(l=0, r=0, t=70, b=40),
            height=480
        )
        fig_esg_vol = apply_plotly_style(fig_esg_vol, title=f"ESG vs risque (volatilité) {r_text}")
        st.plotly_chart(fig_esg_vol, use_container_width=True)
    else:
        st.info("Échantillon insuffisant pour tracer ESG vs volatilité.")
else:
    st.info("Colonnes nécessaires absentes (ESG, Volatilité, Secteur).")

st.divider()

# =========================
# 5.5) Matrice ESG vs Upside (quantiles)
# =========================
if all(c in fdf.columns for c in [COL_ESG, COL_UP]):
    df = fdf[[COL_ESG, COL_UP, COL_SECT]].dropna().copy()
    df[COL_SECT] = df[COL_SECT].fillna("Inconnu")

    # Upside en % si nécessaire
    p95 = np.nanpercentile(pd.to_numeric(df[COL_UP], errors="coerce"), 95)
    if p95 <= 5:
        df[COL_UP] = pd.to_numeric(df[COL_UP], errors="coerce") * 100.0

    # Quantiles ESG / Upside
    if df[COL_ESG].notna().sum() >= 5 and df[COL_UP].notna().sum() >= 5:
        df["ESG_bin"] = pd.qcut(pd.to_numeric(df[COL_ESG], errors="coerce"), 5,
                                labels=["Très bas","Bas","Moyen","Haut","Très haut"])
        df["Upside_bin"] = pd.qcut(pd.to_numeric(df[COL_UP], errors="coerce"), 5,
                                   labels=["Très faible","Faible","Modéré","Élevé","Très élevé"])

        heat = df.groupby(["ESG_bin", "Upside_bin"], as_index=False, observed=True).size()

        fig_esg_up = px.density_heatmap(
            heat, x="ESG_bin", y="Upside_bin", z="size",
            color_continuous_scale="Blues", text_auto=True,
            template="plotly_white"
        )
        fig_esg_up.update_layout(
            xaxis_title="Score ESG (quantiles)",
            yaxis_title="Upside analystes (quantiles)",
            coloraxis_colorbar=dict(title="Nombre d'actifs"),
            margin=dict(l=0, r=0, t=60, b=60),
            height=500
        )
        fig_esg_up = apply_plotly_style(fig_esg_up, title="Matrice ESG vs Upside - lecture combinée valorisation / durabilité")
        st.plotly_chart(fig_esg_up, use_container_width=True)
    else:
        st.info("Échantillon insuffisant pour former des quantiles ESG/Upside.")
else:
    st.info("Colonnes nécessaires absentes (ESG, Upside).")
