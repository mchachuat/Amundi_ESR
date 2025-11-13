import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helpers import load_data, get_filtered, inject_theme, apply_plotly_style, COLORS

inject_theme()
st.title("Analyse sectorielle")

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
actions, indice = load_data("data/analyse données.xlsx")  # <- nom de fichier sans accents
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

# --- Alias colonnes ---
SECT = "Sector (1)"
RET  = "1 Year Total Return - Previous"
VOL  = "Volatility 360 Day Calc"
ESG  = "ESG Score"
UP   = "Upside with Target Price from Analyst"
PE   = "BEst P/E Ratio"
LTG  = "BEst LTG EPS"
CTRY = "Risk Country"

# Prépa univers
df = fdf.copy()
df[SECT] = df[SECT].fillna("Inconnu")

# Upside en % si nécessaire
if UP in df.columns and df[UP].notna().any():
    p95 = np.nanpercentile(pd.to_numeric(df[UP], errors="coerce"), 95)
    if p95 <= 5:
        df[UP] = pd.to_numeric(df[UP], errors="coerce") * 100.0

# PEG ratio si P/E et LTG existent
if PE in df.columns and LTG in df.columns:
    pe_num  = pd.to_numeric(df[PE], errors="coerce")
    ltg_num = pd.to_numeric(df[LTG], errors="coerce")
    df["PEG"] = np.where(ltg_num > 0, pe_num / ltg_num, np.nan)
else:
    df["PEG"] = np.nan

# Sharpe proxy
if all(c in df.columns for c in [RET, VOL]):
    ret_num = pd.to_numeric(df[RET], errors="coerce")
    vol_num = pd.to_numeric(df[VOL], errors="coerce")
    df["sharpe_proxy"] = (ret_num / vol_num).replace([np.inf, -np.inf], np.nan)
else:
    df["sharpe_proxy"] = np.nan

st.markdown("L'analyse révèle une hiérarchie sectorielle où les Financières dominent en rentabilité absolue malgré une valorisation élevée, tandis que les Industriels offrent le meilleur compromis rendement-risque. Les secteurs défensifs (Utilities, Energy) affichent des décotes de valorisation substantielles suggérant une revalorisation potentielle, tandis que Consumer Discretionary et Information Technologie constituent des pièges ayant subi des déceptions structurelles. La volatilité sectorielle reste le principal driver de performance ajusté au risque, surpassant largement les considérations ESG ou de croissance long-terme.")   

# =========================
# 6.1) KPI sectoriels (médianes & effectifs)
# =========================
if SECT in df.columns:
    agg = (
        df.groupby(SECT, as_index=False, observed=True)
          .agg(n=("Ticker","count"),
               ret_med=(RET,"median"),
               vol_med=(VOL,"median"),
               sharpe_med=("sharpe_proxy","median"))
    )

    # Ajouts dynamiques (médianes ESG/UP/PE/PEG si dispo)
    def merge_median(source_col, out_col, base):
        if source_col in df.columns:
            m = df.groupby(SECT, as_index=False, observed=True)[source_col].median().rename(columns={source_col: out_col})
            return base.merge(m, on=SECT, how="left")
        return base

    agg = merge_median(ESG, "esg_med", agg)
    agg = merge_median(UP,  "up_med",  agg)
    agg = merge_median(PE,  "pe_med",  agg)
    agg = merge_median("PEG", "peg_med", agg)

    # Mise en forme
    tbl = agg.sort_values("n", ascending=False).copy()
    fmt_pct = lambda x: (f"{x:.2f} %" if pd.notna(x) else "—")
    fmt_num = lambda x: (f"{x:.2f}" if pd.notna(x) else "—")

    for c in ["ret_med", "vol_med", "up_med"]:
        if c in tbl.columns: tbl[c] = tbl[c].map(fmt_pct)
    for c in ["sharpe_med", "esg_med", "pe_med", "peg_med"]:
        if c in tbl.columns: tbl[c] = tbl[c].map(fmt_num)
    tbl["n"] = tbl["n"].astype(int)

    # Colonnes à afficher (ordre fixe + filtre dispo)
    cols_display = [SECT, "n", "ret_med", "vol_med", "sharpe_med", "esg_med", "up_med", "pe_med", "peg_med"]
    cols_display = [c for c in cols_display if c in tbl.columns]

    headers_map = {
        SECT: "Secteur",
        "n": "n",
        "ret_med": "Rend. méd.",
        "vol_med": "Vol méd.",
        "sharpe_med": "Sharpe méd.",
        "esg_med": "ESG méd.",
        "up_med": "Upside méd.",
        "pe_med": "P/E méd.",
        "peg_med": "PEG méd."
    }
    headers = [f"<b>{headers_map[c]}</b>" for c in cols_display]

    fig_sector_kpi = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color=COLORS["primary"],
            font=dict(color="white", family="Noto Sans, Arial"),
            align="left",
            height=30
        ),
        cells=dict(
            values=[tbl[c] for c in cols_display],
            fill_color=COLORS["background"],
            font=dict(color=COLORS["primary"], family="Noto Sans, Arial", size=13),
            align="left",
            height=26
        )
    )])
    fig_sector_kpi = apply_plotly_style(fig_sector_kpi, title="KPI sectoriels - médianes et effectifs")
    fig_sector_kpi.update_layout(margin=dict(l=0, r=0, t=60, b=5), height=460)
    st.plotly_chart(fig_sector_kpi, use_container_width=True)
else:
    st.info("Colonne secteur manquante.")

st.divider()

# =========================
# 6.2) Carte Risque / Rendement — centroïdes sectoriels
# =========================
if all(c in df.columns for c in [SECT, RET, VOL]):
    df_rr = df[[SECT, RET, VOL]].dropna().copy()
    if len(df_rr):
        centroids = (
            df_rr.groupby(SECT, as_index=False, observed=True)
                 .agg(ret_med=(RET, "median"), vol_med=(VOL, "median"), n=(SECT,"count"))
        )
        fig_sector_rr = px.scatter(
            centroids,
            x="vol_med", y="ret_med",
            size="n", size_max=40,
            color_discrete_sequence=[COLORS["secondary"]],
            template="plotly_white",
            hover_data={"n":True, "vol_med":":.2f", "ret_med":":.2f"},
            text=SECT
        )
        fig_sector_rr.update_traces(textposition="top center", marker=dict(line=dict(width=0)))
        fig_sector_rr.update_layout(
            xaxis_title="Volatilité 1 an (médiane, %)",
            yaxis_title="Rendement 1 an (médiane, %)",
            margin=dict(l=0, r=0, t=60, b=40),
            height=480, showlegend=False
        )
        fig_sector_rr = apply_plotly_style(fig_sector_rr, title="Carte Risque / Rendement - centroïdes sectoriels")
        st.plotly_chart(fig_sector_rr, use_container_width=True)
    else:
        st.info("Données insuffisantes pour les centroïdes sectoriels.")
else:
    st.info("Colonnes nécessaires absentes (Secteur, Rendement, Volatilité).")

st.divider()

# =========================
# 6.3) Return vs Vol (médian) — comparaison sectorielle
# =========================
if all(c in df.columns for c in [SECT, RET, VOL]):
    df_rv = df[[SECT, RET, VOL]].dropna().copy()
    if len(df_rv):
        agg = df_rv.groupby(SECT, as_index=False, observed=True).agg(ret_med=(RET,"median"), vol_med=(VOL,"median")).sort_values("ret_med", ascending=False)
        fig_group = go.Figure()
        fig_group.add_bar(x=agg[SECT], y=agg["ret_med"], name="Rendement médian 1 an", marker_color=COLORS["secondary"])
        fig_group.add_bar(x=agg[SECT], y=agg["vol_med"], name="Volatilité médiane 1 an", marker_color=COLORS["primary"])
        fig_group.update_layout(
            barmode="group",
            xaxis_title="Secteur",
            yaxis_title="%",
            margin=dict(l=0, r=0, t=60, b=120),
            height=520
        )
        fig_group = apply_plotly_style(fig_group, title="Return vs Vol (médian) - comparaison sectorielle")
        st.plotly_chart(fig_group, use_container_width=True)
    else:
        st.info("Aucune donnée suffisante pour comparer rendement/volatilité par secteur.")

st.divider()

# =========================
# 6.4) Prime/Décote P/E vs 5 ans — médiane sectorielle
# =========================
if all(c in df.columns for c in [SECT, PE, "Price / Earnings - 5 Year Average"]):
    PE5 = "Price / Earnings - 5 Year Average"
    df_pe = df[[SECT, PE, PE5]].dropna().copy()
    if len(df_pe):
        df_pe["PE_premium_5Y_%"] = (pd.to_numeric(df_pe[PE], errors="coerce") - pd.to_numeric(df_pe[PE5], errors="coerce")) \
                                   / pd.to_numeric(df_pe[PE5], errors="coerce") * 100.0
        agg = df_pe.groupby(SECT, as_index=False, observed=True).agg(premium_med=("PE_premium_5Y_%","median"), n=(PE,"count")).sort_values("premium_med")
        fig_prem = px.bar(
            agg, x=SECT, y="premium_med",
            template="plotly_white",
            color_discrete_sequence=[COLORS["primary"]],
            hover_data={"premium_med":":.2f","n":True}
        )
        fig_prem.update_layout(
            xaxis_title="Secteur",
            yaxis_title="Prime/Décote P/E vs 5 ans (médiane, %)",
            margin=dict(l=0, r=0, t=60, b=120),
            height=520
        )
        fig_prem = apply_plotly_style(fig_prem, title="Prime/Décote P/E vs 5 ans médiane sectorielle")
        st.plotly_chart(fig_prem, use_container_width=True)
    else:
        st.info("Pas assez de données pour calculer la prime/décote P/E sectorielle.")
else:
    st.info("Colonnes nécessaires absentes (P/E & P/E 5Y).")

st.divider()

# =========================
# 6.5) ESG (médian) vs Return (médian) — bulle par secteur
# =========================
if all(c in df.columns for c in [SECT, ESG, RET]):
    df_esgret = df[[SECT, ESG, RET]].copy()
    df_esgret[SECT] = df_esgret[SECT].fillna("Inconnu")
    df_valid = df_esgret.dropna(subset=[ESG, RET])
    if len(df_valid):
        agg = (
            df_valid.groupby(SECT, as_index=False, observed=True)
                    .agg(esg_med=(ESG, "median"), ret_med=(RET, "median"))
                    .merge(
                        df_valid.groupby(SECT, observed=True).size().reset_index(name="n"),
                        on=SECT, how="left"
                    )
                    .sort_values("ret_med", ascending=False)
        )
        fig_esg_ret_sec = px.scatter(
            agg, x="esg_med", y="ret_med",
            size="n", size_max=40,
            template="plotly_white",
            color_discrete_sequence=[COLORS["secondary"]],
            hover_data={"n": True, "esg_med":":.2f", "ret_med":":.2f"},
            text=SECT
        )
        fig_esg_ret_sec.update_traces(textposition="top center", marker=dict(line=dict(width=0)))
        fig_esg_ret_sec.update_layout(
            xaxis_title="ESG (médian)",
            yaxis_title="Rendement 1 an (médian, %)",
            margin=dict(l=0, r=0, t=60, b=40),
            height=480,
            showlegend=False
        )
        fig_esg_ret_sec = apply_plotly_style(fig_esg_ret_sec, title="ESG (médian) vs Return (médian) par secteur")
        st.plotly_chart(fig_esg_ret_sec, use_container_width=True)
    else:
        st.info("Données insuffisantes pour ESG vs Return sectoriel.")

st.divider()

# =========================
# 6.6) Radar sectoriel (z-scores) avec sélecteur
# =========================
if all(c in df.columns for c in [SECT, RET, VOL, ESG, UP, PE]):
    df_radar = df[[SECT, RET, VOL, ESG, UP, PE]].copy()

    # Upside en % si nécessaire
    if df_radar[UP].notna().any():
        p95 = np.nanpercentile(pd.to_numeric(df_radar[UP], errors="coerce"), 95)
        if p95 <= 5:
            df_radar[UP] = pd.to_numeric(df_radar[UP], errors="coerce") * 100.0

    metrics = {"Return 1Y (%)": RET, "Vol 1Y (%)": VOL, "ESG": ESG, "Upside (%)": UP, "P/E": PE}
    sec_med = df_radar.groupby(SECT, observed=True).median(numeric_only=True)[list(metrics.values())]

    def zscore(series: pd.Series) -> pd.Series:
        mu = np.nanmedian(series)
        sigma = np.nanstd(series)
        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0
        return (series - mu) / sigma

    sec_z = sec_med.apply(zscore)
    sec_z = sec_z.rename(columns={v: k for k, v in metrics.items()})

    if not sec_z.empty:
        cats = list(sec_z.columns)
        cats_closed = cats + [cats[0]]

        fig_radar = go.Figure()
        # Référence univers
        fig_radar.add_trace(go.Scatterpolar(
            r=[0]*len(cats_closed), theta=cats_closed,
            name="Univers (réf=0)",
            line=dict(color="rgba(0,0,0,0.35)", dash="dot"),
            hoverinfo="skip", showlegend=True
        ))

        sectors_list = list(sec_z.index)
        for idx, sect in enumerate(sectors_list):
            row = sec_z.loc[sect]
            vals = row.values.tolist() + [row.values.tolist()[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats_closed, fill="toself",
                name=sect,
                line=dict(color="#081B48"),
                fillcolor="rgba(8,27,72,0.30)",
                visible=(idx == 0)
            ))

        # Dropdown repositionné
        buttons = []
        for i, sect in enumerate(sectors_list):
            visible = [True] + [False]*len(sectors_list)
            visible[i+1] = True
            buttons.append(dict(
                label=sect,
                method="update",
                args=[{"visible": visible},
                      {"title": f"Profil sectoriel (z-scores) — {sect} vs univers"}],
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, showgrid=True, gridcolor="rgba(0,0,0,0.06)")),
            margin=dict(l=0, r=0, t=60, b=40),
            height=520,
            showlegend=True,
            updatemenus=[dict(
                type="dropdown",
                x=0.02, y=1.02,
                xanchor="left", yanchor="top",
                buttons=buttons,
                bgcolor="white",
                bordercolor="rgba(0,0,0,0.1)"
            )]
        )
        initial_title = f"Profil sectoriel (z-scores) — {sectors_list[0]} vs univers"
        fig_radar = apply_plotly_style(fig_radar, title=initial_title)
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Données insuffisantes pour le radar sectoriel.")
else:
    st.info("Colonnes nécessaires absentes pour le radar sectoriel.")
