import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helpers import load_data, get_filtered, inject_theme, apply_plotly_style, COLORS

inject_theme()
st.title("Valorisation et attentes")

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
    esg = st.selectbox("Règle ESG", ["Aucune", "ESG >= médiane secteur"], index=0)
    st.divider()
    st.markdown("Par Maxence Chachuat pour l'équipe ESR d'Amundi")

fdf = get_filtered(actions, s_pick, c_pick, esg if esg != "Aucune" else None)

# --- Alias colonnes ---
PE   = "BEst P/E Ratio"
LTG  = "BEst LTG EPS"
SECT = "Sector (1)"
UPS  = "Upside with Target Price from Analyst"
PE5  = "Price / Earnings - 5 Year Average"
CTRY = "Risk Country"

st.markdown("L'univers affiche une fragmentation entre titres en prime structurelle (Industriels, Technologie) et ceux systématiquement décotés (Énergie, Financières), avec les analystes projetant une revalorisation des cycles défensifs et une normalisation des attentes croissance. Le PEG bas identifie des poches de valeur résiliente (Financière, Énergie française), tandis que la majorité des titres affiche une croissance attendue décalée de leurs valorisations, générant peu d'upside analyst proche du cours actuel. L'exposition value via l'Énergie et l'Immobilier offre le meilleur ratio upside-valorisation, contrastant avec les secteurs cycliques chers offrant peu de soutien analytique.")

# =========================
# 4.1) Scatter P/E vs LTG EPS (+ tendance)
# =========================
if all(c in fdf.columns for c in [PE, LTG, SECT]):
    df = fdf[[PE, LTG, SECT, "Ticker", CTRY]].dropna().copy()
    df[SECT] = df[SECT].fillna("Inconnu")

    if len(df) >= 2:
        x = pd.to_numeric(df[LTG], errors="coerce").to_numpy()
        y = pd.to_numeric(df[PE],  errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[m], y[m]
        slope, intercept = (np.polyfit(xv, yv, 1) if len(xv) >= 2 else (0.0, float(np.nanmean(yv)) if len(yv) else 0.0))
        R = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) >= 2 else np.nan

        fig_pe_ltg = px.scatter(
            df, x=LTG, y=PE, color=SECT,
            hover_data={"Ticker":True, CTRY:True, SECT:True, LTG:":.2f", PE:":.2f"},
            template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_pe_ltg.update_traces(marker=dict(size=9, opacity=0.9, line=dict(width=0)))

        if len(xv) >= 2:
            x_line = np.linspace(xv.min(), xv.max(), 100)
            y_line = slope * x_line + intercept
            fig_pe_ltg.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=COLORS["primary"], width=2, dash="dash"),
                hoverinfo="skip", showlegend=False
            ))

        r_text = f"(R = {R:.2f})" if np.isfinite(R) else ""
        fig_pe_ltg.update_layout(
            xaxis_title="LTG EPS (croissance attendue, %)",
            yaxis_title="P/E (BEst)",
            legend_title_text="Secteur",
            margin=dict(l=0, r=0, t=70, b=40), height=480
        )
        fig_pe_ltg = apply_plotly_style(fig_pe_ltg, title=f"P/E vs LTG EPS - valorisation vs croissance {r_text}")
        st.plotly_chart(fig_pe_ltg, use_container_width=True)
    else:
        st.info("Données insuffisantes pour tracer le scatter P/E vs LTG.")
else:
    st.info("Colonnes nécessaires absentes (P/E, LTG, Secteur).")

st.divider()

# =========================
# 4.2) P/E actuel vs moyenne 5 ans — primes et décotes (Top 20)
# =========================
if all(c in fdf.columns for c in [PE, PE5]):
    df = fdf[["Ticker", SECT, CTRY, PE, PE5]].dropna().copy()
    if len(df) > 0:
        df["PE_premium_5Y_%"] = (pd.to_numeric(df[PE], errors="coerce") - pd.to_numeric(df[PE5], errors="coerce")) \
                                / pd.to_numeric(df[PE5], errors="coerce") * 100.0

        top_premium  = df.nlargest(10, "PE_premium_5Y_%").assign(Groupe="Prime (Top10)")
        top_discount = df.nsmallest(10, "PE_premium_5Y_%").assign(Groupe="Décote (Top10)")
        plot_df = pd.concat([top_premium, top_discount], ignore_index=True)
        plot_df = plot_df.sort_values(["Groupe", "PE_premium_5Y_%"], ascending=[False, False])

        fig_pe_premium = go.Figure()
        for grp, color in [("Prime (Top10)", COLORS["secondary"]), ("Décote (Top10)", COLORS["primary"])]:
            sub = plot_df[plot_df["Groupe"] == grp]
            fig_pe_premium.add_trace(go.Bar(
                x=sub["Ticker"], y=sub["PE_premium_5Y_%"], name=grp,
                marker_color=color,
                customdata=np.stack([sub[PE], sub[PE5]], axis=-1),
                hovertemplate="<b>%{x}</b><br>Prime/Décote: %{y:.2f}%<br>PE: %{customdata[0]:.2f} | PE 5Y: %{customdata[1]:.2f}<extra></extra>"
            ))

        fig_pe_premium.update_layout(
            xaxis_title="Ticker", yaxis_title="Prime/Décote vs 5 ans (%)",
            barmode="group", margin=dict(l=0, r=0, t=60, b=120), height=520,
            legend_title_text=""
        )
        fig_pe_premium = apply_plotly_style(fig_pe_premium, title="P/E actuel vs moyenne 5 ans - primes et décotes (Top 20)")
        st.plotly_chart(fig_pe_premium, use_container_width=True)
    else:
        st.info("Aucune donnée exploitable (P/E ou P/E 5Y manquants).")
else:
    st.info("Colonnes nécessaires absentes (P/E actuel et P/E 5 ans).")

st.divider()

# =========================
# 4.3) Distribution des P/E par secteur (box plot)
# =========================
if all(c in fdf.columns for c in [PE, SECT]):
    df = fdf[[PE, SECT]].dropna().copy()
    df[SECT] = df[SECT].fillna("Inconnu")

    if len(df) > 0:
        fig_pe_box = px.box(
            df, x=SECT, y=PE, points=False, template="plotly_white",
            color_discrete_sequence=[COLORS["secondary"]]
        )
        fig_pe_box.update_layout(
            xaxis_title="Secteur", yaxis_title="P/E (BEst)",
            margin=dict(l=0, r=0, t=60, b=120), height=520
        )
        fig_pe_box = apply_plotly_style(fig_pe_box, title="Distribution des P/E par secteur")
        st.plotly_chart(fig_pe_box, use_container_width=True)
    else:
        st.info("Aucune donnée P/E valide après filtres.")
else:
    st.info("Colonnes nécessaires absentes (P/E, Secteur).")

st.divider()

# =========================
# 4.4) PEG ratio — extrêmes + distribution robuste
# =========================
if all(c in fdf.columns for c in [PE, LTG, SECT, CTRY]):
    df = fdf[["Ticker", SECT, CTRY, PE, LTG]].dropna().copy()
    if len(df) > 0:
        # PEG = PE / LTG ; attention LTG <= 0 → non interprétable
        df["PEG"] = pd.to_numeric(df[PE], errors="coerce") / pd.to_numeric(df[LTG], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["PEG"])

        if len(df) > 0:
            # Table Top/Bottom 10
            tb = pd.concat([
                df[df[LTG] > 0].nsmallest(10, "PEG").assign(Groupe="Top 10 PEG (bas)"),
                df[df[LTG] > 0].nlargest(10, "PEG").assign(Groupe="Top 10 PEG (haut)")
            ], ignore_index=True)

            tb_disp = tb.copy()
            for c in [PE, LTG, "PEG"]:
                tb_disp[c] = pd.to_numeric(tb_disp[c], errors="coerce").map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")

            fig_peg_tbl = go.Figure(data=[go.Table(
                header=dict(
                    values=["<b>Groupe</b>","<b>Ticker</b>","<b>Secteur</b>","<b>Pays</b>","<b>P/E</b>","<b>LTG EPS</b>","<b>PEG</b>"],
                    fill_color=COLORS["primary"], font=dict(color="white", family="Noto Sans, Arial"),
                    align="left", height=30
                ),
                cells=dict(
                    values=[tb_disp["Groupe"], tb_disp["Ticker"], tb_disp[SECT], tb_disp[CTRY], tb_disp[PE], tb_disp[LTG], tb_disp["PEG"]],
                    fill_color=COLORS["background"], font=dict(color=COLORS["primary"], family="Noto Sans, Arial", size=13),
                    align="left", height=26
                )
            )])
            fig_peg_tbl = apply_plotly_style(fig_peg_tbl, title="PEG ratio extrêmes (Top/Bottom 10)")
            fig_peg_tbl.update_layout(margin=dict(l=0, r=0, t=60, b=5), height=420)
            st.plotly_chart(fig_peg_tbl, use_container_width=True)

            st.divider()

            # Distribution robuste (LTG > 0, clip [p1, p99])
            df_dist = df[pd.to_numeric(df[LTG], errors="coerce") > 0].copy()
            df_dist = df_dist.replace([np.inf, -np.inf], np.nan).dropna(subset=["PEG"])
            if len(df_dist) > 1:
                p1, p99 = np.nanpercentile(df_dist["PEG"], [1, 99])
                df_plot = df_dist[(df_dist["PEG"] >= p1) & (df_dist["PEG"] <= p99)].copy()

                excluded_total = len(df) - len(df_plot)
                excluded_reason = {
                    "LTG ≤ 0 ou NaN": int((pd.to_numeric(df[LTG], errors="coerce") <= 0).sum()),
                    "PEG hors [p1,p99]": int(len(df_dist) - len(df_plot))
                }

                fig_peg_hist = px.histogram(
                    df_plot, x="PEG", nbins=30, template="plotly_white",
                    color_discrete_sequence=[COLORS["secondary"]]
                )
                fig_peg_hist.update_traces(marker_line_width=0, opacity=0.85)
                fig_peg_hist.add_vline(x=float(df_plot["PEG"].median()), line_dash="dot", line_color="rgba(0,0,0,0.35)")
                fig_peg_hist.add_vline(x=float(df_plot["PEG"].mean()),   line_dash="dash", line_color=COLORS["primary"])
                fig_peg_hist.update_layout(
                    xaxis_title="PEG (LTG > 0, clip [p1, p99])",
                    yaxis_title="Effectif",
                    margin=dict(l=0, r=0, t=70, b=40),
                    height=420,
                    bargap=0.15,
                    annotations=[
                        dict(
                            text=f"Exclus: total {excluded_total} | LTG≤0/NaN: {excluded_reason['LTG ≤ 0 ou NaN']} | Hors [p1,p99]: {excluded_reason['PEG hors [p1,p99]']}",
                            xref="paper", yref="paper", x=0.0, y=1.10, showarrow=False,
                            font=dict(family='Noto Sans, Arial', size=12, color='rgba(0,0,0,0.7)')
                        )
                    ]
                )
                fig_peg_hist = apply_plotly_style(fig_peg_hist, title="Distribution du PEG ratio")
                st.plotly_chart(fig_peg_hist, use_container_width=True)
            else:
                st.info("Échantillon insuffisant pour une distribution de PEG robuste.")
        else:
            st.info("PEG non calculable (valeurs invalides ou insuffisantes).")
    else:
        st.info("Aucune donnée disponible pour le calcul du PEG.")
else:
    st.info("Colonnes nécessaires absentes (P/E, LTG, Secteur, Pays).")

st.divider()

# =========================
# 4.5) Upside potentiel (médian) par secteur
# =========================
if all(c in fdf.columns for c in [UPS, SECT]):
    df = fdf[[UPS, SECT]].dropna().copy()
    df[SECT] = df[SECT].fillna("Inconnu")

    # Normaliser en % si nécessaire
    p95 = np.nanpercentile(pd.to_numeric(df[UPS], errors="coerce"), 95)
    if p95 <= 5:
        df[UPS] = pd.to_numeric(df[UPS], errors="coerce") * 100.0

    agg = df.groupby(SECT, as_index=False).agg(up_median=(UPS, "median"), n=(UPS, "count")).sort_values("up_median", ascending=False)

    fig_up_sector = px.bar(
        agg, x=SECT, y="up_median",
        template="plotly_white",
        color_discrete_sequence=[COLORS["primary"]],
        hover_data={"up_median":":.2f","n":True}
    )
    fig_up_sector.update_layout(
        xaxis_title="Secteur", yaxis_title="Upside médian (%)",
        margin=dict(l=0, r=0, t=60, b=120), height=520
    )
    fig_up_sector = apply_plotly_style(fig_up_sector, title="Upside potentiel (médian) par secteur")
    st.plotly_chart(fig_up_sector, use_container_width=True)
else:
    st.info("Colonnes nécessaires absentes (Upside, Secteur).")

st.divider()

# =========================
# 4.6) Upside analystes vs P/E (+ tendance)
# =========================
if all(c in fdf.columns for c in [PE, UPS, SECT, CTRY]):
    df = fdf[[PE, UPS, SECT, "Ticker", CTRY]].dropna().copy()
    df[SECT] = df[SECT].fillna("Inconnu")

    # Normaliser Upside en %
    p95 = np.nanpercentile(pd.to_numeric(df[UPS], errors="coerce"), 95)
    if p95 <= 5:
        df[UPS] = pd.to_numeric(df[UPS], errors="coerce") * 100.0

    if len(df) >= 2:
        x = pd.to_numeric(df[PE],  errors="coerce").to_numpy()
        y = pd.to_numeric(df[UPS], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[m], y[m]
        slope, intercept = (np.polyfit(xv, yv, 1) if len(xv) >= 2 else (0.0, float(np.nanmean(yv)) if len(yv) else 0.0))
        R = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) >= 2 else np.nan

        fig_up_pe = px.scatter(
            df, x=PE, y=UPS, color=SECT,
            hover_data={"Ticker":True, CTRY:True, SECT:True, PE:":.2f", UPS:":.2f"},
            template="plotly_white",
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_up_pe.update_traces(marker=dict(size=9, opacity=0.9, line=dict(width=0)))

        if len(xv) >= 2:
            x_line = np.linspace(xv.min(), xv.max(), 100)
            y_line = slope * x_line + intercept
            fig_up_pe.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=COLORS["primary"], width=2, dash="dash"),
                hoverinfo="skip", showlegend=False
            ))

        r_text = f"(R = {R:.2f})" if np.isfinite(R) else ""
        fig_up_pe.update_layout(
            xaxis_title="P/E (BEst)", yaxis_title="Upside analystes (%)",
            legend_title_text="Secteur",
            margin=dict(l=0, r=0, t=70, b=40), height=480
        )
        fig_up_pe = apply_plotly_style(fig_up_pe, title=f"Upside analystes vs P/E - décote vs attentes {r_text}")
        st.plotly_chart(fig_up_pe, use_container_width=True)
    else:
        st.info("Données insuffisantes pour tracer Upside vs P/E.")
else:
    st.info("Colonnes nécessaires absentes (P/E, Upside, Secteur, Pays).")
