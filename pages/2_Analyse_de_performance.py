import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helpers import load_data, get_filtered, inject_theme, apply_plotly_style, COLORS

inject_theme()
st.title("Analyse de performance")

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

# --- Data & filtres ---
actions, indice = load_data("data/analyse données.xlsx")
with st.sidebar:
    st.header("Filtres")
    sectors = sorted(actions["Sector (1)"].dropna().unique())
    countries = sorted(actions["Risk Country"].dropna().unique())
    s_pick = st.multiselect("Secteurs", sectors)
    c_pick = st.multiselect("Pays", countries)
    esg = st.selectbox("Règle ESG", ["Aucune", "ESG >= médiane secteur"], index=0)
    st.divider()
    st.markdown("Par Maxence Chachuat pour l'équipe ESR d'Amundi")

fdf = get_filtered(actions, s_pick, c_pick, esg if esg != "Aucune" else None)

col_ret = "1 Year Total Return - Previous"
col_vol = "Volatility 360 Day Calc"
col_sector = "Sector (1)"
col_country = "Risk Country"
col_pe = "BEst P/E Ratio"

st.markdown("L'indice SX5E révèle une dynamique de performance profondément non-linéaire où les stratégies de faible volatilité offrent un rendement attractif ajusté au risque, les secteurs défensifs fournissent une stabilité relative tandis que les secteurs cycliques concentrent une volatilité extrême, et où la géographie demeure un facteur de différenciation majeur indépendant des valuations. Cette configuration suggère qu'une exposition diversifiée géographiquement mais sélective sectoriellement reste optimale.")


# ========================
# 2.1) Distribution des rendements 1 an
# ========================
if col_ret in fdf.columns and fdf[col_ret].notna().any():
    s = pd.to_numeric(fdf[col_ret], errors="coerce").dropna()
    ret_min, ret_max = float(s.min()), float(s.max())
    ret_mean, ret_med = float(s.mean()), float(s.median())

    fig_perf_dist = px.histogram(
        fdf, x=col_ret,
        nbins=35,  # barres plus fines
        template="plotly_white",
        color_discrete_sequence=[COLORS["secondary"]],
    )
    fig_perf_dist.update_traces(marker_line_width=0, opacity=0.85)

    # Lignes de moyenne / médiane
    fig_perf_dist.add_vline(x=ret_med, line_dash="dot", line_color="rgba(0,0,0,0.35)")
    fig_perf_dist.add_vline(x=ret_mean, line_dash="dash", line_color=COLORS["primary"])

    # Style
    fig_perf_dist = apply_plotly_style(fig_perf_dist, title="Distribution des rendements 1 an")
    fig_perf_dist.update_layout(
        xaxis_title="Rendement 1 an (%)",
        yaxis_title="Effectif",
        margin=dict(l=0, r=0, t=70, b=50),
        height=420,
        bargap=0.15,
        annotations=[
            dict(
                text=f"Min {ret_min:.2f}%  |  Moy {ret_mean:.2f}%  |  Med {ret_med:.2f}%  |  Max {ret_max:.2f}%",
                xref="paper", yref="paper", x=0.0, y=1.10, showarrow=False,
                font=dict(family="Noto Sans, Arial", size=12, color="rgba(0,0,0,0.75)")
            )
        ]
    )
    st.plotly_chart(fig_perf_dist, use_container_width=True)
else:
    st.info("Aucune donnée valide pour les rendements 1 an avec les filtres actuels.")

st.divider()

# ========================
# 2.2) Box plot par secteur
# ========================
if all(c in fdf.columns for c in [col_ret, col_sector]) and fdf[col_ret].notna().any():
    df_sec = fdf.copy()
    df_sec[col_sector] = df_sec[col_sector].fillna("Inconnu")
    fig_box_sector = px.box(
        df_sec, x=col_sector, y=col_ret, points=False,
        template="plotly_white",
        color_discrete_sequence=[COLORS["secondary"]],
    )
    fig_box_sector.update_layout(
        xaxis_title="Secteur",
        yaxis_title="Rendement 1 an (%)",
        margin=dict(l=0, r=0, t=50, b=120),
        height=520
    )
    fig_box_sector = apply_plotly_style(fig_box_sector, title="Dispersion des rendements par secteur")
    st.plotly_chart(fig_box_sector, use_container_width=True)
else:
    st.info("Données insuffisantes pour le box plot sectoriel.")


st.divider()

# ========================
# 2.3) Rendement moyen par pays
# ========================

if all(c in fdf.columns for c in [col_ret, col_country]) and fdf[col_ret].notna().any():
    df_cty = fdf.copy()
    df_cty[col_country] = df_cty[col_country].fillna("Inconnu")
    agg = (
        df_cty.groupby(col_country, as_index=False)
              .agg(ret_mean=(col_ret, "mean"), ret_median=(col_ret, "median"), n=("Ticker", "count"))
              .sort_values("ret_mean", ascending=False)
    )
    fig_country_bar = px.bar(
        agg, x=col_country, y="ret_mean",
        template="plotly_white",
        color_discrete_sequence=[COLORS["secondary"]],
        hover_data={"ret_mean":":.2f", "ret_median":":.2f", "n": True}
    )
    fig_country_bar.update_layout(
        xaxis_title="Pays (Risk Country)",
        yaxis_title="Rendement moyen 1 an (%)",
        margin=dict(l=0, r=0, t=50, b=80),
        height=520
    )
    fig_country_bar = apply_plotly_style(fig_country_bar, title="Rendement moyen par pays")
    st.plotly_chart(fig_country_bar, use_container_width=True)
else:
    st.info("Données insuffisantes pour l’agrégat par pays.")

st.divider()

# ========================
# 2.4) Rendement moyen par décile de volatilité
# ========================
if all(c in fdf.columns for c in [col_ret, col_vol]) and fdf[col_vol].notna().any():
    dfq = fdf[[col_ret, col_vol]].dropna().copy()
    if len(dfq) >= 10:
        labels = [f"D{i}" for i in range(1, 11)]
        # Rang pour stabiliser en cas d'égalités
        dfq["Vol_Decile"] = pd.qcut(
            dfq[col_vol].rank(method="first"),
            q=10, labels=labels, duplicates="drop"
        )
        dfq["Vol_Decile"] = pd.Categorical(dfq["Vol_Decile"], categories=labels, ordered=True)

        by_dec = (
            dfq.groupby("Vol_Decile", observed=True, as_index=False)
               .agg(ret_mean=(col_ret, "mean"), n=(col_ret, "count"))
               .sort_values("Vol_Decile")
        )

        fig_ret_by_decile = px.bar(
            by_dec, x="Vol_Decile", y="ret_mean",
            template="plotly_white",
            color_discrete_sequence=[COLORS["primary"]],
            hover_data={"ret_mean":":.2f", "n": True}
        )
        fig_ret_by_decile.update_layout(
            xaxis_title="Déciles de volatilité (D1 = plus faible)",
            yaxis_title="Rendement moyen 1 an (%)",
            margin=dict(l=0, r=0, t=50, b=20),
            height=420
        )
        fig_ret_by_decile = apply_plotly_style(fig_ret_by_decile, title="Rendement moyen par décile de volatilité")
        st.plotly_chart(fig_ret_by_decile, use_container_width=True)
    else:
        st.info("Échantillon insuffisant pour former 10 déciles.")
else:
    st.info("Données insuffisantes pour l’analyse par déciles de volatilité.")

st.divider()

# ========================
# 2.5) Scatter Rendement vs P/E (+ tendance)
# ========================
if all(c in fdf.columns for c in [col_ret, col_pe, col_sector]):
    plot_df = fdf[[col_ret, col_pe, col_sector, "Ticker", col_country]].dropna().copy()
    plot_df[col_sector] = plot_df[col_sector].fillna("Inconnu")

    if len(plot_df) >= 2:
        # Régression simple sur valeurs numériques valides
        x = pd.to_numeric(plot_df[col_pe], errors="coerce").to_numpy()
        y = pd.to_numeric(plot_df[col_ret], errors="coerce").to_numpy()
        msk = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[msk], y[msk]

        fig_ret_vs_pe = px.scatter(
            plot_df, x=col_pe, y=col_ret, color=col_sector,
            hover_data={"Ticker": True, col_country: True, col_sector: True, col_pe:":.2f", col_ret:":.2f"},
            template="plotly_white",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_ret_vs_pe.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0)))

        if len(xv) >= 2:
            slope, intercept = np.polyfit(xv, yv, 1)
            R = float(np.corrcoef(xv, yv)[0, 1])
            x_line = np.linspace(xv.min(), xv.max(), 100)
            y_line = slope * x_line + intercept
            fig_ret_vs_pe.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines", name="Trend",
                line=dict(color=COLORS["primary"], width=2, dash="dash"),
                hoverinfo="skip", showlegend=False
            ))
            trend_direction = "négative" if slope < 0 else "positive"
            r_text = f"(R = {R:.2f}, pente {trend_direction})" if np.isfinite(R) else ""
        else:
            r_text = ""

        fig_ret_vs_pe.update_layout(
            xaxis_title="P/E (BEst)",
            yaxis_title="Rendement 1 an (%)",
            legend_title_text="Secteur",
            margin=dict(l=0, r=0, t=70, b=40),
            height=480
        )
        fig_ret_vs_pe = apply_plotly_style(fig_ret_vs_pe, title=f"Rendement 1 an vs P/E {r_text}")
        st.plotly_chart(fig_ret_vs_pe, use_container_width=True)
    else:
        st.info("Données insuffisantes pour tracer le scatter P/E vs rendement.")
else:
    st.info("Colonnes manquantes pour P/E vs rendement (vérifie P/E / Secteur / Retour).")

