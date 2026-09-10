import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/boosted_cp_presentation.slides.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boosted Conformal Prediction Intervals
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Contexte

    - Méthodes précédentes comme la prédiction conforme (CP) et ses variantes (NCP, MCP), ou la régression quantile conforme (CQR) assurent par construction une couverture marginale mais ne permettent pas de garantir d'autres propriétés désirables.
    - But: adapter des intervalles de prédiction conforme à un usage spécifique (e.g. amélioration de la couverture conditionelle, minimisation de la taille des intervalles) à travers des techniques de **gradient boosting** comme XGBoost ou LightGBM.
    - Introduction d'un formalisme d'abstraction de méthodes d'estimation d'incertitude (prédiction conforme, régression quantile)
    - Constitution de fonctions de perte adaptées aux deux usages spécifiques mentionnés ci-dessus
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formalisme

    ### 2.1. Standard CP

    - Split dataset en training set $I_1$, calibration set $I_2$ et test set $I_3$ supposés échangeables
    - Sur le training set, entraîner un modèle de régression $f$ pour produire un score de conformité $E(\cdot, \cdot; f)$ comme $E(x,y;f) = \vert y - \hat{\mu}(x) \vert$, où $f$ est ici simplement $\hat{\mu}$.
    - Evaluer $E(\cdot, \cdot; f)$ sur $I_2$ pour obtenir $\{E_i\}_{i\in I_2}$, puis choisir le $(1-\alpha)$-quantile $Q_{1-\alpha}(E, I_2)$.
    - Pour une nouvelle observation $X_{n+1}$, l'intervalle de prediction conforme est
    $$C_n(X_{n+1}) = \{y \in \mathbb{R} : E(X_{n+1}, y; f) \leq Q_{1-\alpha}(E, I_2) \}$$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.2. Prédiction Conforme Adaptive Localement

    - Lorsque la l'étendue de la distribution $Y$ varie fortement selon $X$, les intervalles prédits par la prédiction conforme standard perdent en utilité
    - Développement de méthodes adaptatives localement comme la CP normalisée, formalisées par
    $$E(x,y;f) = \frac{\vert y - \mu_0(x) \vert}{\sigma_0(x)}, \quad C_n(X_{n+1}) = [\mu_0(X_{n+1}) - \sigma_0(X_{n+1})Q_{1-\alpha}(E,I_2), \mu_0(X_{n+1}) + \sigma_0(X_{n+1})Q_{1-\alpha}(E,I_2)]$$
    où $\mu_0(X)$ est un estimateur de la moyenne conditionelle $\mathbb{E}[Y\vert X]$ et $\sigma_0(X) \gt 0$ est un estimateur de la dispersion de la moyenne conditionelle. Ces deux fonctions peuvent être choisies arbitrairement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3. Régression Quantile Conforme

    - La prédiction conforme adaptative localement produit des intervalles symmétriques par construction
    - Inspirée de la régression quantile, la régression quantile conforme prédit deux quantiles asymmétriques $(\hat{q}_{\alpha/2}(x), \hat{q}_{1-\alpha/2}(x))$ et un score de conformité $E(x,y;f)=\max\{ \hat{q}_{\alpha/2}(x)-y, y-\hat{q}_{1-\alpha/2}(x)\}$.
    - L'intervalle de prédiction conforme est donné par
    $$C_n(X_{n+1}) = [\hat{q}_{\alpha/2}(X_{n+1}) - Q_{1-\alpha}(E,I_2), \hat{q}_{1-\alpha/2}(X_{n+1}) + Q_{1-\alpha}(E,I_2)]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image("CQR.png", style={"margin": "auto"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4. Familles de scores de conformité généralisés

    - On peut finalement formuler une famille de score généralisée, incluant les scores de conformité locaux, CQR et autres variantes:
    $$\mathcal{H}:= \big\{ E(\cdot, \cdot; f): E(x,y;f) = \frac{\max \{ \mu_1(x)-y, y-\mu_2(x) \}}{\sigma(x)}, \mu_1(\cdot) \leq \mu_2(\cdot), \sigma(\cdot) \gt 0 \big\}$$
    $$C_n(X) = [\mu_1(X) - \sigma(X)Q_{1-\alpha}(E,I_2), \mu_2(X) + \sigma(X)Q_{1-\alpha}(E,I_2)]$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Boosted CP

    Etant donné un dataset $\mathcal{D}$ supposé échangeable divisé en un training set $I_1$, un calibration set $I_2$ et un test set $I_3$, la procédure est la suivante:

    1. Définir $E(\cdot, \cdot)$ et initialiser $f_0$; par exemple $f_0 = (\mu_0, \sigma_0)$, $\mu_0 = \hat{y}, \sigma_0 = \text{MAD}(x)$.
    2. Définir une fonction de perte $\ell$ différentiable par rapport à $f$
    3. Trouver le nombre de rounds de boosting $\tau$:
       - Split le en $k$ folds
       - Pour chaque fold $j$: appliquer l'algorithme pendant $T$ rounds (fixe) en boostant sur les $k-1$ autres folds pour obtenir $T+1$ fonctions de score candidates $E_j^{(0)}, ..., E_j^{(0)}$; puis evaluant sur le fold $j$, pour obtenir $\ell(E_j^{(0)}), ..., \ell(E_j^{(0)})$
       - Faire la moyenne des $\ell(E_j^{(t)})$ à travers les $k$ folds et choisir le nombre de rounds qui minimise la perte moyenne $\tau = \argmin_t \frac{1}{k} \sum_{j=1}^k \ell(E_j^t)$

    4. Réentraîner avec $\tau$ rounds de boosting sur tout le dataset
    5. A chaque round de boosting $t$:
       - Evaluer $E_{t-1}(X_i)$, construire l'intervalle $C_{t-1}(X_i)$ à partir du training set $I_1$
       - Calculer les pseudo-résiduels $\frac{\partial\ell}{\partial f_{t-1}}$
       - Fit un arbre peu profond $h_t(X)$ aux gradients (ici simplement 1 couche)
       - Update $f_t = f_{t-1} + \eta h_t(X)$ pour obtenir $E_t(\cdot, \cdot; f_t)$
    6. Calibration sur $I_2$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Fonctions de perte

    ### 4.1. Couverture Conditionelle
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2. Largeur des Intervalles
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
