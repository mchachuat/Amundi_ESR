
  <h1>📊 Exercices pour l'entretien de Stage Assistant Gestion Multi-Asset</h1>

  <p>
    Cette application <strong>Streamlit</strong> fournit une plateforme complète d’analyse
    financière, extra-financière (ESG), sectorielle et d’optimisation de portefeuille
    pour l’univers <strong>EuroStoxx 50 étendu (SX5E)</strong>.
  </p>
  <p>
    Elle offre un tableau de bord interactif intégrant analyses statistiques, filtrages ESG,
    visualisations avancées, optimisation de portefeuille et simulations de Monte Carlo.
  </p>

  <hr />

  <h2>🚀 Fonctionnalités principales</h2>

  <h3>1. Tableau de bord principal</h3>
  <ul>
    <li>Vue d’ensemble des KPI de l’univers (rendement 1 an, volatilité, score ESG, Sharpe proxy).</li>
    <li>Carte risque / rendement.</li>
    <li>Répartition sectorielle et géographique.</li>
  </ul>

  <h3>2. Analyse de performance</h3>
  <ul>
    <li>Distribution des rendements (histogrammes, boxplots).</li>
    <li>Comparaison des performances par secteur et par pays.</li>
    <li>Visualisation de la relation rendement / volatilité.</li>
  </ul>

  <h3>3. Analyse de risque</h3>
  <ul>
    <li>Matrice de corrélation interactive (facteurs &amp; risques).</li>
    <li>Distribution des volatilités et prime de volatilité (IV – Vol).</li>
    <li>Analyse des bêtas proxy par secteur et par titre.</li>
  </ul>

  <h3>4. Valorisation &amp; attentes</h3>
  <ul>
    <li>Analyse des multiples de valorisation (P/E, prime/décote vs moyenne 5 ans).</li>
    <li>PEG ratio (P/E / LTG EPS) — distributions et extrêmes.</li>
    <li>Upside des analystes (niveau absolu et par secteur).</li>
  </ul>

  <h3>5. Analyse extra-financière ESG</h3>
  <ul>
    <li>Distribution des scores ESG sur l’univers.</li>
    <li>Scores ESG médians par secteur.</li>
    <li>Relations ESG vs performance (rendement, risque, upside).</li>
    <li>Matrice ESG vs Upside (lecture combinée durabilité / valorisation).</li>
  </ul>

  <h3>6. Analyse sectorielle</h3>
  <ul>
    <li>KPI sectoriels (rendement, volatilité, Sharpe proxy, ESG, Upside, P/E, PEG… en médianes).</li>
    <li>Carte risque/rendement des centroïdes sectoriels.</li>
    <li>Prime/décote P/E vs 5 ans par secteur.</li>
    <li>Profils radar par secteur (z-scores multi-facteurs).</li>
  </ul>

  <h3>7. Portefeuille</h3>
  <ul>
    <li>Résumé pédagogique de la méthodologie (filtres ESG, facteurs, optimisation).</li>
    <li>Tableau final du portefeuille :
      <ul>
        <li><code>Ticker</code>, <code>Secteur</code>, <code>Pays</code>, <code>Poids</code>, <code>µ ex-ante</code>, <code>Vol 1Y</code>, <code>Beta (proxy)</code>.</li>
      </ul>
    </li>
    <li>Graphiques de diagnostic :
      <ul>
        <li>Barres de poids par titre.</li>
        <li>Donut de répartition sectorielle.</li>
        <li>Scatter <strong>µ ex-ante vs contribution au risque</strong>.</li>
        <li>Simulation de Monte Carlo des trajectoires du portefeuille (fan chart + paths).</li>
      </ul>
    </li>
  </ul>

  <hr />

  <h2>🗂️ Structure du projet</h2>

  <pre><code>project/
├── pages/
│   ├── Analyse_de_performance.py
│   ├── Analyse_de_risque.py
│   ├── Valorisation_et_attentes.py
│   ├── Analyse_extra_financiere_ESG.py
│   ├── Analyse_sectorielle.py
│   ├── Portefeuille.py
│
├── helpers.py
├── data/
│   └── analyse_donnees.xlsx
│
├── logo.jpeg
├── Tableau_de_bord.py
├── README.md
└── requirements.txt
</code></pre>

  <hr />

  <h2>🔧 Installation</h2>

  <h3>1. Cloner le projet</h3>
    <pre><code>git clone https://github.com/mchachuat/Amundi_ESR
  cd Amundi_ESR
</code></pre>

  <h3>2. Installer les dépendances</h3>
  <pre><code>pip install -r requirements.txt
</code></pre>

  <h3>3. Lancer l’application</h3>
  <pre><code>streamlit run Tableau_de_bord.py
</code></pre>

  <hr />

  <h2>📌 Technologies utilisées</h2>

  <ul>
    <li>Python 3.9+</li>
    <li>Streamlit</li>
    <li>Pandas, NumPy</li>
    <li>Plotly (visualisation interactive)</li>
    <li>scikit-learn (prétraitements / modèles)</li>
    <li>SciPy (optimisation)</li>
    <li>statsmodels / régressions</li>
  </ul>

  <hr />

  <h2>👤 Auteur</h2>

  <p>
    Application développée par <strong>Maxence Chachuat</strong><br />
    Pour l’équipe <strong>ESR - Amundi Investment Solutions</strong>.
  </p>

  <hr />

  <h2>📄 Licence</h2>

  <p>
    Projet interne - usage non commercial.<br />
    Toute reproduction ou diffusion est soumise à autorisation.
  </p>

</body>
</html>
