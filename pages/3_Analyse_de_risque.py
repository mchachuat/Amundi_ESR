import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helpers import load_data, get_filtered, inject_theme, apply_plotly_style, COLORS

inject_theme()
st.title("Analyse de risque")

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
actions, indice = load_data("data/analyse données.xlsx")
with st.sidebar:
    st.header("Filtres")
    s_pick = st.multiselect("Secteurs", sorted(actions["Sector (1)"].dropna().unique()))
    st.divider()
    st.markdown("Par Maxence Chachuat pour l'équipe ESR d'Amundi")
fdf = get_filtered(actions, s_pick, None)

# Pour simplifier la suite
universe = fdf.copy()
col_ret = "1 Year Total Return - Previous"
col_vol = "Volatility 360 Day Calc"
col_iv  = "12 Month Put Implied Volatility"

st.markdown("L'univers d'investissement se caractérise par une concentration du risque-bêta dans les secteurs cycliques (Énergie, Technologie, Industriels) qui amplifient les chocs de marché, une prime de risque insuffisante au regard de la volatilité réalisée (sauf extrêmes), une asymétrie downside accentuée pour les titres volatiles, et une anticipation pessimiste du marché des options reflétée par la prime IV-Vol positive. Cette architecture suggère une exposition systémique élevée avec une compensation adéquate uniquement pour les positions extrêmes, rendant la sélection sectorielle et la gestion du downside critiques.")

# =========================
# 3.1) IV vs Vol (prime IV − Vol)
# =========================
if all(c in universe.columns for c in [col_vol, col_iv]) and universe[[col_vol, col_iv]].notna().any().any():
    plot_df = universe[[col_vol, col_iv, "Ticker", "Sector (1)", "Risk Country"]].dropna().copy()
    plot_df["Prime_IV_Vol"] = pd.to_numeric(plot_df[col_iv], errors="coerce") - pd.to_numeric(plot_df[col_vol], errors="coerce")

    if len(plot_df) > 0:
        fig_iv_vs_vol = px.scatter(
            plot_df,
            x=col_vol, y=col_iv,
            color="Sector (1)",
            hover_data={
                "Ticker": True,
                "Risk Country": True,
                "Sector (1)": True,
                col_vol:":.2f", col_iv:":.2f",
                "Prime_IV_Vol":":.2f"
            },
            template="plotly_white",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_iv_vs_vol.update_traces(marker=dict(size=9, opacity=0.9, line=dict(width=0)))

        # Diagonale y = x
        xy_min = float(np.nanmin([plot_df[col_vol].min(), plot_df[col_iv].min()]))
        xy_max = float(np.nanmax([plot_df[col_vol].max(), plot_df[col_iv].max()]))
        fig_iv_vs_vol.add_trace(go.Scatter(
            x=[xy_min, xy_max], y=[xy_min, xy_max],
            mode="lines", line=dict(color="rgba(0,0,0,0.25)", dash="dot"), name="y = x",
            hoverinfo="skip", showlegend=False
        ))

        fig_iv_vs_vol.update_layout(
            xaxis_title="Volatilité 1 an (%)",
            yaxis_title="Vol implicite 12m (%)",
            legend_title_text="Secteur",
            margin=dict(l=0, r=0, t=60, b=40),
            height=480
        )
        fig_iv_vs_vol = apply_plotly_style(fig_iv_vs_vol, title="Vol implicite vs Vol historique - prime de volatilité (IV − Vol)")
        st.plotly_chart(fig_iv_vs_vol, use_container_width=True)
    else:
        st.info("Aucune paire (Vol, IV) complète après filtres.")
else:
    st.info("Colonnes manquantes ou données insuffisantes pour IV vs Vol.")

st.divider()

# =========================
# 3.2) Tableau Top/Bottom prime (IV − Vol)
# =========================
if all(c in universe.columns for c in [col_vol, col_iv]):
    tb = (
        universe[[ "Ticker", "Sector (1)", "Risk Country", col_vol, col_iv ]]
        .dropna()
        .assign(Prime=lambda d: pd.to_numeric(d[col_iv], errors="coerce") - pd.to_numeric(d[col_vol], errors="coerce"))
    )
    if len(tb) > 0:
        top10 = tb.nlargest(10, "Prime").copy()
        bot10 = tb.nsmallest(10, "Prime").copy()
        top10["Groupe"] = "Top 10 prime (IV−Vol)"
        bot10["Groupe"] = "Bottom 10 prime (IV−Vol)"
        tbl = pd.concat([top10, bot10], ignore_index=True)

        # Mise en forme (affichage en %)
        for c in [col_vol, col_iv, "Prime"]:
            tbl[c] = pd.to_numeric(tbl[c], errors="coerce").map(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

        fig_premium_tbl = go.Figure(data=[go.Table(
            header=dict(
                values=[ "<b>Groupe</b>", "<b>Ticker</b>", "<b>Secteur</b>", "<b>Pays</b>", "<b>Vol 1 an</b>", "<b>IV 12m</b>", "<b>Prime (IV−Vol)</b>" ],
                fill_color=COLORS["primary"], font=dict(color="white", family="Noto Sans, Arial"), align="left", height=30
            ),
            cells=dict(
                values=[ tbl["Groupe"], tbl["Ticker"], tbl["Sector (1)"], tbl["Risk Country"], tbl[col_vol], tbl[col_iv], tbl["Prime"] ],
                fill_color=COLORS["background"], font=dict(color=COLORS["primary"], family="Noto Sans, Arial", size=13),
                align="left", height=26
            )
        )])
        fig_premium_tbl = apply_plotly_style(fig_premium_tbl, title="Prime de volatilité - extrêmes (Top/Bottom 10)")
        fig_premium_tbl.update_layout(margin=dict(l=0, r=0, t=60, b=5), height=420)
        st.plotly_chart(fig_premium_tbl, use_container_width=True)
    else:
        st.info("Aucune donnée exploitable pour la prime IV−Vol.")
else:
    st.info("Colonnes manquantes pour calculer la prime IV−Vol.")

st.divider()

# =========================
# 3.3) Distribution des volatilités
# =========================
if col_vol in universe.columns and universe[col_vol].notna().any():
    s = pd.to_numeric(universe[col_vol], errors="coerce").dropna()
    fig_vol_dist = px.histogram(
        universe, x=col_vol, nbins=30, template="plotly_white",
        color_discrete_sequence=[COLORS["secondary"]]
    )
    fig_vol_dist.update_traces(marker_line_width=0, opacity=0.85)
    fig_vol_dist.add_vline(x=float(s.median()), line_dash="dot", line_color="rgba(0,0,0,0.35)")
    fig_vol_dist.add_vline(x=float(s.mean()),   line_dash="dash", line_color=COLORS["primary"])
    fig_vol_dist.update_layout(
        xaxis_title="Volatilité 1 an (%)",
        yaxis_title="Effectif",
        margin=dict(l=0, r=0, t=60, b=40),
        height=420
    )
    fig_vol_dist = apply_plotly_style(fig_vol_dist, title="Distribution des volatilités (1 an)")
    st.plotly_chart(fig_vol_dist, use_container_width=True)
else:
    st.info("Données insuffisantes pour tracer la distribution des volatilités.")

st.divider()

# =========================
# 3.4) Bêta (proxy) — médiane par secteur + extrêmes
# =========================
if col_vol in indice.columns and pd.to_numeric(indice[col_vol].squeeze(), errors="coerce") is not None:
    vol_index = pd.to_numeric(indice[col_vol].squeeze(), errors="coerce")
    vol_index = float(vol_index) if np.isfinite(vol_index) else np.nan

    beta_df = universe[["Ticker", "Sector (1)", "Risk Country", col_vol]].dropna().copy()
    if np.isfinite(vol_index) and len(beta_df) > 0:
        beta_df["beta_proxy"] = pd.to_numeric(beta_df[col_vol], errors="coerce") / vol_index

        # Agrégation par secteur — médiane
        agg_beta = (
            beta_df.groupby("Sector (1)", as_index=False)
                   .agg(beta_median=("beta_proxy", "median"), n=("Ticker", "count"))
                   .sort_values("beta_median", ascending=False)
        )

        fig_beta_sector = px.bar(
            agg_beta, x="Sector (1)", y="beta_median",
            template="plotly_white",
            color_discrete_sequence=[COLORS["primary"]],
            hover_data={"beta_median":":.2f","n":True}
        )
        fig_beta_sector.update_layout(
            xaxis_title="Secteur",
            yaxis_title="β (proxy = Vol titre / Vol indice, médiane sectorielle)",
            margin=dict(l=0, r=0, t=60, b=120),
            height=520
        )
        fig_beta_sector = apply_plotly_style(fig_beta_sector, title="Bêta (proxy) médian par secteur")
        st.plotly_chart(fig_beta_sector, use_container_width=True)

        st.divider()
        
        # Table Top/Bottom 10 β titres
        tb = pd.concat([
            beta_df.nlargest(10, "beta_proxy").assign(Groupe="Top 10 β"),
            beta_df.nsmallest(10, "beta_proxy").assign(Groupe="Bottom 10 β")
        ], ignore_index=True)
        tb["beta_proxy"] = tb["beta_proxy"].map(lambda x: f"{x:.2f}")
        fig_beta_tbl = go.Figure(data=[go.Table(
            header=dict(
                values=["<b>Groupe</b>", "<b>Ticker</b>", "<b>Secteur</b>", "<b>Pays</b>", "<b>β (proxy)</b>"],
                fill_color=COLORS["primary"], font=dict(color="white", family="Noto Sans, Arial"),
                align="left", height=30
            ),
            cells=dict(
                values=[tb["Groupe"], tb["Ticker"], tb["Sector (1)"], tb["Risk Country"], tb["beta_proxy"]],
                fill_color=COLORS["background"], font=dict(color=COLORS["primary"], family="Noto Sans, Arial", size=13),
                align="left", height=26
            )
        )])
        fig_beta_tbl = apply_plotly_style(fig_beta_tbl, title="β (proxy) extrêmes par titre")
        fig_beta_tbl.update_layout(margin=dict(l=0, r=0, t=60, b=5), height=420)
        st.plotly_chart(fig_beta_tbl, use_container_width=True)
    else:
        st.info("Impossible de calculer le bêta proxy (volatilité indice manquante ou aucune action valide).")
else:
    st.info("La volatilité de l’indice est manquante.")

st.divider()

# =========================
# 3.5) Frontière efficiente simplifiée (nuage + enveloppe supérieure)
# =========================
if all(c in universe.columns for c in [col_ret, col_vol]) and universe[[col_ret, col_vol]].notna().any().any():
    df = universe[[col_ret, col_vol, "Ticker", "Sector (1)"]].dropna().copy()
    df.rename(columns={col_ret:"ret", col_vol:"vol"}, inplace=True)
    pts = df[["vol","ret"]].to_numpy()
    order = np.argsort(pts[:,0])
    pts = pts[order]

    def upper_hull(points):
        hull = []
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        for p in points:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0:
                hull.pop()
            hull.append(tuple(p))
        # Nettoyage: conserver l'enveloppe croissante en rendement
        hull = [hull[0]] + [p for i,p in enumerate(hull[1:-1], start=1) if p[1]>=hull[i-1][1]] + [hull[-1]]
        return np.array(hull)

    frontier = upper_hull(pts)
    frontier = frontier[np.argsort(frontier[:,0])]

    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(
        x=df["vol"], y=df["ret"], mode="markers",
        marker=dict(size=8, color="rgba(81,156,221,0.7)"),
        name="Titres", text=df["Ticker"],
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.2f}%<br>Rend: %{y:.2f}%<extra></extra>"
    ))
    if len(frontier) >= 2:
        fig_frontier.add_trace(go.Scatter(
            x=frontier[:,0], y=frontier[:,1], mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=6, color=COLORS["primary"]),
            name="Frontière (simplifiée)", hoverinfo="skip"
        ))

    fig_frontier.update_layout(
        xaxis_title="Volatilité 1 an (%)",
        yaxis_title="Rendement 1 an (%)",
        margin=dict(l=0, r=0, t=60, b=40),
        height=500,
        legend_title_text=""
    )
    fig_frontier = apply_plotly_style(fig_frontier, title="Frontière efficiente (nuage titres + enveloppe supérieure)")
    st.plotly_chart(fig_frontier, use_container_width=True)
else:
    st.info("Données insuffisantes pour tracer la frontière efficiente simplifiée.")

st.divider()

# =========================
# 3.6) Heatmap de corrélations (facteurs & risque)
# =========================
cols = {
    "Rendement 1 an (%)":          "1 Year Total Return - Previous",
    "Volatilité 1 an (%)":         "Volatility 360 Day Calc",
    "IV 12m (%)":                  "12 Month Put Implied Volatility",
    "Upside (%)":                  "Upside with Target Price from Analyst",
    "ESG (score)":                 "ESG Score",
    "P/E (BEst)":                  "BEst P/E Ratio",
    "LTG EPS (%)":                 "BEst LTG EPS"
}
avail = [v for v in cols.values() if v in universe.columns]
if len(avail) >= 2:
    dfc = universe[avail].copy()

    # Upside en % si besoin
    if "Upside with Target Price from Analyst" in dfc.columns:
        p95 = np.nanpercentile(pd.to_numeric(dfc["Upside with Target Price from Analyst"], errors="coerce"), 95)
        if p95 <= 5:
            dfc["Upside with Target Price from Analyst"] = pd.to_numeric(dfc["Upside with Target Price from Analyst"], errors="coerce") * 100.0

    # Renommer colonnes pour affichage
    rename_map = {v:k for k,v in cols.items() if v in dfc.columns}
    dfc = dfc.rename(columns=rename_map)

    # Conversion numérique et corrélation
    dfc = dfc.apply(pd.to_numeric, errors="coerce")
    if dfc.dropna(how="all").shape[1] >= 2:
        corr = dfc.corr(method="pearson").round(2)
        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto", color_continuous_scale="Blues",
            template="plotly_white"
        )
        fig_corr.update_layout(
            coloraxis_colorbar=dict(title="ρ"),
            margin=dict(l=0, r=0, t=60, b=5),
            height=520
        )
        fig_corr = apply_plotly_style(fig_corr, title="Corrélations - risque & facteurs")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Pas assez de colonnes numériques pour calculer une corrélation.")
else:
    st.info("Colonnes insuffisantes pour construire une heatmap de corrélation.")
