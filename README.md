# Gold/Silver Forecasting

Recherche reproductible de prévision des rendements journaliers J+1 de l'or et de l'argent.

Le projet compare des baselines, des modèles tabulaires et des architectures séquentielles compactes avec une validation strictement chronologique. Il ne constitue pas un conseil en investissement.

## Installation sur macOS Apple Silicon

Le projet cible Python 3.11, disponible ici via Homebrew :

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

Les dépendances de recherche plus lourdes sont séparées :

```bash
python -m pip install -e '.[research]'
```

Pour activer les fondation models et leurs dépendances :

```bash
python -m pip install -e '.[foundation]'
hf auth login
python scripts/train.py --include-foundation-models
```

Les checkpoints Chronos/TimesFM sont téléchargés séparément depuis Hugging Face. Le benchmark exécuté ici a utilisé `amazon/chronos-bolt-tiny` et `google/timesfm-2.5-200m-pytorch` en local ; ces identifiants sont passés via `--chronos-path` et `--timesfm-path`. Un token Hugging Face valide peut être nécessaire ; il ne doit jamais être écrit dans le dépôt.

## Workflow

```bash
# Télécharger GC=F, SI=F et les variables de marché publiques
make download

# Produire les corrélations Gold/Silver et un résumé textuel
make analysis

# Comparer les modèles, sélectionner séparément Gold/Silver et tester sur les 20 % finaux
make train

# Optionnel : ajouter les adaptateurs Chronos/TimesFM si leurs paquets sont installés.
# Cette option peut télécharger des poids de modèles depuis Hugging Face.
python scripts/train.py --include-foundation-models

# Prédire le prochain rendement avec le bundle final
make predict

# Préparer seulement le bundle final pour Hugging Face
make export-hf
```

Les données brutes et les modèles sont volontairement ignorés par Git. Le fichier `data/raw/manifest.json` conserve les tickers, dates, colonnes et checksum du téléchargement.

## Méthode

- Cibles : `log(close[t+1] / close[t])` pour Gold et Silver.
- Features : retards, momentum, volatilité, moyennes mobiles, drawdown, volume, ratio Gold/Silver, corrélations roulantes et variables de marché.
- Familles : baselines, Ridge/ElasticNet, ExtraTrees, HistGradientBoosting, TSMixer compact, PatchTST compact et TimeMixer compact. Les deux dernières architectures utilisent des fenêtres causales de 32 jours et sont exécutables sur CPU/MPS ; elles sont des implémentations légères de recherche, pas les dépôts officiels complets. XGBoost, FRED, Optuna et Jupyter sont optionnels via `.[research]`.
- Chronos et TimesFM disposent d’adaptateurs univariés optionnels. Ils utilisent uniquement l’historique du rendement de l’actif, alors que les modèles locaux utilisent également les covariables de `X` ; cette différence d’information est documentée dans le rapport. Ils ne sont pas téléchargés implicitement.
- Split : derniers 20 % réservés au test final ; cinq folds `TimeSeriesSplit` sur les premiers 80 %, avec gap d'un jour.
- Sélection : Sharpe annualisé net moyen en validation ; départage par Sharpe médian, drawdown, IC et MAE.
- Le baseline zéro rendement est conservé comme point de comparaison uniquement ; il ne peut jamais être sélectionné comme modèle final.
- Backtest : signal `-1/0/+1`, coût de référence de 10 bps par unité de turnover, analyse de sensibilité 0/5/10/20 bps.
- Comparaison test : `data/processed/gold_test_comparison.csv` et `silver_test_comparison.csv` contiennent le Sharpe net, le turnover, l'IC et un intervalle bootstrap par blocs de cinq jours pour chaque famille. Les scores de sélection restent ceux de la validation ; le test final n'est jamais utilisé pour choisir les hyperparamètres.
- Preuve statistique : `*_statistical_tests.csv` contient le test Diebold–Mariano sur les erreurs de prévision, le bootstrap apparié de différence de Sharpe et la correction Holm–Bonferroni. Une p-value non significative signifie que la supériorité observée ne peut pas être établie sur cet historique.

Le modèle final est sauvegardé dans `models/gold_silver_bundle.joblib` et le rapport dans `reports/experiment_summary.json`.

## Résultat du benchmark local

Sur le cache courant, avec le même horizon J+1, cinq folds walk-forward, `gap=1` et 10 bps :

| Actif | Modèle retenu par validation | Sharpe validation | Sharpe test | Sharpe test 95 % bootstrap |
|---|---|---:|---:|---:|
| Gold | ExtraTrees | 0,485 | 0,779 | [-0,134 ; 1,600] |
| Silver | HistGradientBoosting | 0,165 | 0,025 | [-0,936 ; 0,782] |

PatchTST et TimeMixer ne battent pas ExtraTrees sur Gold dans cette première grille compacte. Chronos-Bolt Tiny et TimesFM 2.5 ont aussi été exécutés réellement : Gold Sharpe test respectivement `-0,409` et `0,275`, contre `0,779` pour ExtraTrees. Cependant, aucune supériorité Gold d'ExtraTrees sur HistGradientBoosting, Chronos ou TimesFM n'est statistiquement significative après Holm–Bonferroni. Le résultat est donc prometteur, mais ne constitue pas une preuve de SOTA.

## FRED

La première exécution fonctionne sans FRED. Pour ajouter des séries FRED, renseigner `FRED_API_KEY` dans `.env`, puis activer `use_fred` et les séries dans `configs/default.yaml`. Les clés ne doivent jamais être committées.

## Publication privée

Initialiser et publier le dépôt GitHub après authentification :

```bash
git init
git add .
git commit -m "Initial reproducible Gold Silver forecasting pipeline"
gh repo create gold-silver-forecasting --private --source=. --remote=origin --push
```

Après entraînement, vérifier le contenu de `hf_export/` puis publier uniquement ce dossier dans un dépôt Hugging Face privé :

```bash
hf auth login
hf upload <namespace>/gold-silver-forecasting hf_export --private
```

Les données Yahoo Finance/FRED ne sont pas redistribuées par ce dépôt ; elles sont téléchargées à la demande.
