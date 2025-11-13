# Portefeuille.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from helpers import inject_theme, apply_plotly_style, COLORS

inject_theme()
st.title("Portefeuille")

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

# === Filtres ===
with st.sidebar:
    st.markdown("Par Maxence Chachuat pour l'équipe ESR d'Amundi")

# ===== Données du portefeuille (telles que fournies) =====
rows = [
    {"Ticker":"TotalEnergies SE","Sector (1)":"Energy","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":18.26,"Vol 1Y":20.98,"Beta (proxy)":1.30},
    {"Ticker":"Air Liquide SA","Sector (1)":"Materials","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":4.77,"Vol 1Y":19.55,"Beta (proxy)":1.21},
    {"Ticker":"Enel SpA","Sector (1)":"Utilities","Risk Country":"ITALY","weight":0.08,"mu_ex_ante":4.60,"Vol 1Y":18.47,"Beta (proxy)":1.14},
    {"Ticker":"Dassault Systèmes SE","Sector (1)":"Information Technology","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":13.16,"Vol 1Y":27.14,"Beta (proxy)":1.68},
    {"Ticker":"Unibail-Rodamco-Westfield","Sector (1)":"Real Estate","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":15.16,"Vol 1Y":25.19,"Beta (proxy)":1.56},
    {"Ticker":"Sanofi SA","Sector (1)":"Health Care","Risk Country":"USA","weight":0.08,"mu_ex_ante":20.10,"Vol 1Y":20.22,"Beta (proxy)":1.25},
    {"Ticker":"Orange SA","Sector (1)":"Communication Services","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":7.35,"Vol 1Y":15.72,"Beta (proxy)":0.97},
    {"Ticker":"Publicis Groupe SA","Sector (1)":"Communication Services","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":13.12,"Vol 1Y":23.38,"Beta (proxy)":1.44},
    {"Ticker":"Pandora A/S","Sector (1)":"Consumer Discretionary","Risk Country":"CHINA","weight":0.08,"mu_ex_ante":17.24,"Vol 1Y":33.11,"Beta (proxy)":2.04},
    {"Ticker":"AXA SA","Sector (1)":"Financials","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":4.37,"Vol 1Y":19.76,"Beta (proxy)":1.22},
    {"Ticker":"Industria de Diseño Textil, S.A. (Inditex)","Sector (1)":"Consumer Discretionary","Risk Country":"SPAIN","weight":0.08,"mu_ex_ante":5.14,"Vol 1Y":23.29,"Beta (proxy)":1.44},
    {"Ticker":"Danone SA","Sector (1)":"Consumer Staples","Risk Country":"FRANCE","weight":0.08,"mu_ex_ante":0.28,"Vol 1Y":15.13,"Beta (proxy)":0.93},
    {"Ticker":"Allianz SE","Sector (1)":"Financials","Risk Country":"GERMANY","weight":0.04,"mu_ex_ante":1.20,"Vol 1Y":17.93,"Beta (proxy)":1.11},
    {"Ticker":"Siemens AG","Sector (1)":"Industrials","Risk Country":"GERMANY","weight":0.01,"mu_ex_ante":5.11,"Vol 1Y":29.67,"Beta (proxy)":1.83},
]

port = pd.DataFrame(rows)


# --- Explication méthodologique (haut de page) ---
st.markdown("""
Ce processus décrit la construction d'un portefeuille d'actions européennes ESG conçu pour maximiser le ratio de Sharpe ex ante. L'univers de base est constitué du jeu de données fourni pour cet exercice. La méthodologie combine des approches quantitatives robustes : filtrage ESG sectoriel, analyse factorielle standardisée, estimation des rendements anticipés, et optimisation mathématique sous contraintes de diversification. J'ai choisi de combiner des critères ESG avec la maximisation du ratio de Sharpe ex ante pour construire un portefeuille éligible à l'épargne salariale et retraite tout en optimisant le rendement ajusté au risque. Le ratio de Sharpe reste la métrique de référence en gestion quantitative moderne car elle garantit une allocation optimale.

## Méthodologie de construction du portefeuille

### Filtrage Sectoriel ESG

Pour chaque secteur, on ne conserve que les actions dont le score ESG est supérieur ou égal à la médiane sectorielle :

$$\\text{Conserver action } i \\text{ si } \\text{ESG}_i \\geq \\text{Médiane}(\\text{ESG}_{\\text{secteur}})$$

Pour garantir au moins une action par secteur, une réintégration hiérarchique est appliquée aux secteurs devenant vides : meilleur ESG, puis meilleur rendement 1 an, puis plus faible volatilité, puis première action disponible.

### Calcul des Z-Scores Sectoriels

Pour chaque facteur financier, on standardise par secteur selon une z-score robuste :

$$Z_f^{(i)} = \\frac{x_f^{(i)} - \\text{Médiane}(x_f)}{\\text{MAD}(x_f)}$$

où MAD est l'écart absolu médian. Les principaux facteurs sont : momentum ajusté du risque, valeur (P/E négatif), PEG négatif, et faible volatilité.

### Pré-Sélection par Score Composite

Le score composite synthétique est la moyenne simple des z-scores disponibles :

$$S_{\\text{composite}}^{(i)} = \\frac{1}{n_{\\text{facteurs}}} \\sum_{f} Z_f^{(i)}$$

Dans chaque secteur, on sélectionne les $K$ meilleures actions selon ce score, où :

$$K = \\max\\left(1, \\left\\lceil \\frac{N_{\\text{TARGET}}}{\\text{nombre de secteurs}} \\right\\rceil\\right)$$

### Estimation du Rendement Anticipé Ex Ante

Le rendement ex ante combine deux sources via un **barycentre pondéré à parts égales** :

**Composante factorielle :** Pour chaque secteur, régression ridge du rendement 1 an sur les z-scores :

$$\\mu_{\\text{factoriel}}^{(i)} = \\beta_0 + \\sum_f \\beta_f \\cdot Z_f^{(i)}$$

**Composante upside :** Estimation directe des analystes.

**Barycentre 50/50 :**

$$\\mu_{\\text{ex ante}}^{(i)} = 0.5 \\cdot \\mu_{\\text{factoriel}}^{(i)} + 0.5 \\cdot \\text{Upside}^{(i)}$$

(ou l'une ou l'autre si disponibilité partielle). Clipping final entre percentiles 2.5 et 97.5 pour robustesse.

### Volatilité Attendue et Matrice de Covariance

Selon un modèle single-index, le bêta proxy de chaque action est :

$$\\beta_i = \\frac{\\sigma_i}{\\sigma_{\\text{indice}}}$$

La variance spécifique est :

$$\\text{Var}_{\\text{spécifique}}^{(i)} = \\sigma_i^2 - \\beta_i^2 \\cdot \\sigma_{\\text{indice}}^2$$

La matrice de covariance :

$$\\Sigma = \\beta \\beta^T \\sigma_{\\text{indice}}^2 + \\text{diag}(\\text{Var}_{\\text{spécifique}}) + 10^{-8} I$$

### Optimisation Markowitz Contrainte

On maximise le ratio de Sharpe ex ante :

$$\\text{Sharpe} = \\frac{w^T \\mu - r_f}{\\sqrt{w^T \\Sigma w}}$$

via minimisation sur une grille de paramètres $\\lambda$ :

$$\\min_w \\left[ -(w^T \\mu - \\lambda w^T \\Sigma w) \\right]$$

**Contraintes :**
- $\\sum_i w_i = 1$ (poids total = 100%)
- $0 \\leq w_i \\leq 0.08$ (max 8% par action)
- $\\sum_{i \\in s} w_i \\geq 0.001$ pour chaque secteur $s$ (diversification minimale)

Algorithme SLSQP avec warm-start. La solution optimale est celle maximisant le Sharpe ex ante.

## Limites et Perspectives
            
### Limites

Cette approche, bien que structurée et robuste, présente plusieurs limitations. Les estimations des rendements anticipés reposent sur des données historiques et des estimations d'analystes qui peuvent être biaisées ou obsolètes. Le modèle single-index simplifie les corrélations réelles et ne capture pas les dépendances sectorielles complexes. Le filtrage ESG utilise une médiane sectorielle arbitraire qui peut ne pas refléter des standards ESG absolus. Les contraintes sectorielles rigides peuvent forcer l'inclusion de secteurs peu attrayants.

### Perspectives d'amélioration

Pour renforcer cette méthodologie, plusieurs extensions pourraient être explorées : intégrer des modèles GARCH ou à changements de régimes pour une volatilité dynamique, utiliser l'apprentissage machine pour pondérer adapativement les facteurs, remplacer le modèle single-index par des approches multi-facteurs (Fama-French, APT), adapter les critères ESG à chaque secteur, mettre en place un backtesting rigoureux avec validation croisée, intégrer des données macro-économiques et sentimentales, et ajuster dynamiquement les contraintes selon les conditions de marché. Une approche complémentaire combinant analyse quantitative, jugement expert, et suivi actif des risques maximiserait les chances de surperformance.

### Conclusion

Cette méthodologie offre un cadre rigoureux et bien encadré pour la sélection et la pondération quantitatives des actions selon des critères factoriels et ESG robustes. Son succès dépend de la fiabilité des données et de la stabilité des estimations. Une approche complémentaire combinant analyse quantitative, jugement expert, et suivi actif des risques maximiserait les chances de surperformance face à un indice de référence.
""")


st.divider()

# ===== KPIs (ex-ante) – tels que fournis =====
ER_p = 9.89     # %
SD_p = 21.76    # %
SH_p = 0.45

c1,c2,c3 = st.columns(3)
c1.metric("E[R] (ex-ante)", f"{ER_p:.2f} %")
c2.metric("Vol (ex-ante)", f"{SD_p:.2f} %")
c3.metric("Sharpe (ex-ante)", f"{SH_p:.2f}")

st.divider()

# ===== Tableau formaté =====
tbl = port.copy()
for c in ["weight","mu_ex_ante","Vol 1Y"]:
    tbl[c] = tbl[c].map(lambda x: f"{x*100:.2f} %" if c=="weight" else f"{x:.2f} %")
tbl["Beta (proxy)"] = port["Beta (proxy)"].map(lambda x: f"{x:.2f}")

fig_tbl = go.Figure(data=[go.Table(
    header=dict(
        values=[f"<b>{h}</b>" for h in ["Nom","Secteur","Pays","Poids","µ ex-ante","Vol 1Y","Beta (proxy)"]],
        fill_color=COLORS["primary"],
        font=dict(color="white", family="Noto Sans, Arial"),
        align="left",
        height=30
    ),
    cells=dict(
        values=[tbl["Ticker"], tbl["Sector (1)"], tbl["Risk Country"], tbl["weight"], tbl["mu_ex_ante"], tbl["Vol 1Y"], tbl["Beta (proxy)"]],
        fill_color="white",
        font=dict(color=COLORS["primary"], family="Noto Sans, Arial", size=13),
        align="left",
        height=26
    )
)])
fig_tbl = apply_plotly_style(fig_tbl, title="Composition du portefeuille")
st.plotly_chart(fig_tbl, use_container_width=True)

st.divider()

# ===== Graphique 1 : Poids par titre =====
fig_w_bar = px.bar(
    port.sort_values("weight", ascending=False),
    x="Ticker", y="weight",
    template="plotly_white",
    color_discrete_sequence=[COLORS["primary"]],
    hover_data={"weight":":.4f","mu_ex_ante":":.2f","Vol 1Y":":.2f","Beta (proxy)":":.2f"}
)
fig_w_bar.update_yaxes(title="Poids", tickformat=".0%")
fig_w_bar.update_xaxes(title="Ticker")
fig_w_bar = apply_plotly_style(fig_w_bar, title="Poids par titre")
st.plotly_chart(fig_w_bar, use_container_width=True)

st.divider()

# ===== Graphique 2 : Répartition sectorielle =====
by_sector = port.groupby("Sector (1)", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
fig_sector = px.pie(
    by_sector, names="Sector (1)", values="weight",
    hole=0.55, template="plotly_white",
    color_discrete_sequence=px.colors.sequential.Blues
)
fig_sector.update_traces(
    textposition="inside",
    texttemplate="%{label}<br>%{percent:.0%}",
    hovertemplate="%{label}<br>%{value:.2%}<extra></extra>"
)
fig_sector = apply_plotly_style(fig_sector, title="Répartition sectorielle du portefeuille")
st.plotly_chart(fig_sector, use_container_width=True)

st.divider()

# ===== Graphique 3 : Scatter diagnostic µ vs contribution au risque =====
# Reconstruction d'une matrice de covariance Σ avec un modèle single-index :
# Sigma = beta beta^T * sigma_m^2 + diag(spec_var) avec
# sigma_m^2 estimée par médiane((Vol 1Y / beta)^2), spec_var >= 0.

w = port["weight"].values.astype(float)               # (k,)
mu_pc = port["mu_ex_ante"].values.astype(float)       # % annuels
sigma_i = port["Vol 1Y"].values.astype(float)         # % annuels
beta_i  = port["Beta (proxy)"].values.astype(float)

valid = np.isfinite(sigma_i) & np.isfinite(beta_i) & (beta_i > 0)
sigma_m = np.nanmedian(sigma_i[valid] / beta_i[valid]) if valid.any() else np.nan
sigma_m2 = float(sigma_m**2) if np.isfinite(sigma_m) else float(np.nanmedian(sigma_i**2))

spec_var = np.clip(sigma_i**2 - (beta_i**2) * sigma_m2, a_min=0.0, a_max=None)
Sigma = np.outer(beta_i, beta_i) * sigma_m2 + np.diag(spec_var)
Sigma = Sigma + 1e-8*np.eye(len(port))  # régularisation numérique

Sigma_w = Sigma @ w
tot_var = float(w @ Sigma_w)
cr = (w * Sigma_w) / tot_var if tot_var > 0 else np.zeros_like(w)

diag_df = pd.DataFrame({
    "Ticker": port["Ticker"],
    "Poids": w,
    "mu": mu_pc,            # en %
    "CRisk": cr             # part de variance
})
fig_diag = px.scatter(
    diag_df, x="mu", y="CRisk", size="Poids", hover_name="Ticker",
    template="plotly_white",
    color_discrete_sequence=[COLORS["secondary"]]
)
fig_diag.update_layout(
    xaxis_title="µ ex-ante (%)",
    yaxis_title="Contribution au risque (part de variance)",
    height=460, margin=dict(l=0, r=0, t=60, b=40)
)
fig_diag = apply_plotly_style(fig_diag, title="Diagnostic - µ vs contribution au risque")
st.plotly_chart(fig_diag, use_container_width=True)

st.divider()

import streamlit as st
import numpy as np
import time

import streamlit as st
import numpy as np
import time

# ===== Graphique 4 : Monte Carlo — trajectoires (base 100) =====
# Utilise µ (en % annuels) et Σ (en %^2 annuels) → passe en décimaux & journalier.

# Initialiser le seed dans session_state
if 'monte_carlo_seed' not in st.session_state:
    st.session_state.monte_carlo_seed = 123

N_SIMS_PATHS = 500
N_SHOW       = 60
N_DAYS       = 252
V0           = 100.0

# Utiliser le seed depuis session_state
np.random.seed(st.session_state.monte_carlo_seed)

# % -> décimaux
muA = mu_pc / 100.0
SA  = Sigma / (100.0**2)

# Journalier
mu_d = muA / N_DAYS
S_d  = SA  / N_DAYS

# Projection PSD (sécurité numérique)
eigval, eigvec = np.linalg.eigh(S_d)
eigval = np.clip(eigval, a_min=1e-12, a_max=None)
S_d_psd = (eigvec @ np.diag(eigval) @ eigvec.T).astype(float)

k = len(w)
Z = np.random.multivariate_normal(mean=np.zeros(k), cov=S_d_psd, size=(N_DAYS, N_SIMS_PATHS))
drift_adj = (mu_d - 0.5 * np.diag(S_d_psd))
Rlog = Z + drift_adj
rp_log_daily = np.tensordot(Rlog, w, axes=([2],[0]))  # (days, sims)
V_paths = V0 * np.exp(rp_log_daily.cumsum(axis=0))

# Fan chart + quelques trajectoires
t = np.arange(1, N_DAYS+1)
p50 = np.percentile(V_paths, 50, axis=1)
p05 = np.percentile(V_paths, 5,  axis=1)
p95 = np.percentile(V_paths, 95, axis=1)

fig_mc = go.Figure()
fig_mc.add_trace(go.Scatter(x=t, y=p95, line=dict(width=0), hoverinfo="skip", showlegend=False))
fig_mc.add_trace(go.Scatter(x=t, y=p05, line=dict(width=0), fill='tonexty',
                            fillcolor='rgba(8,27,72,0.08)', name='Bandes 5–95 %',
                            line_color='rgba(0,0,0,0)', hoverinfo="skip"))
fig_mc.add_trace(go.Scatter(x=t, y=p50, mode='lines',
                            line=dict(color=COLORS["primary"], width=2), name='Médiane'))
for j in range(min(N_SHOW, N_SIMS_PATHS)):
    fig_mc.add_trace(go.Scatter(x=t, y=V_paths[:, j], mode='lines',
                                line=dict(color='rgba(81,156,221,0.35)', width=1),
                                showlegend=False))
fig_mc.update_layout(
    xaxis_title="Jours ouvrés",
    yaxis_title="Valeur du portefeuille (base 100)",
    margin=dict(l=0, r=0, t=60, b=40),
    height=520
)
fig_mc = apply_plotly_style(fig_mc, title="Simulation de Monte Carlo")
st.plotly_chart(fig_mc, use_container_width=True)

# Bouton pour relancer la simulation (placé APRÈS le graphique)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎲 Relancer la simulation Monte Carlo", use_container_width=True):
        st.session_state.monte_carlo_seed = int(time.time() * 1000) % 100000
        st.rerun()


