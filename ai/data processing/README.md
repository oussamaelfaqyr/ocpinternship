# Youssoufia SCADA — Pipeline de préparation des données (fichier unique)

Un seul fichier de code, `preprocessing_pipeline.py`, qui exécute les 9
étapes dans l'ordre validé (cleaning -> validation -> feature selection ->
split chronologique -> windowing -> encoding -> scaling). Toute la
configuration (chemins, seuils, taille de fenêtre, ratios de split...) reste
dans `params.yaml`, à côté — rien de codé en dur dans le script.

## Lancer

```bash
pip install -r requirements.txt
python preprocessing_pipeline.py params.yaml
```

Place `youssoufia_pandapower_dataset_dataset.csv` dans `data/raw/dataset.csv`
avant de lancer (chemin défini dans `params.yaml: paths.raw`).

## Vérifié de bout en bout

Résultats identiques à la version multi-fichiers précédente (voir
`pipeline_run_log.txt` pour le log complet de l'exécution réelle) :

- 393 lignes `normal` relabelisées en `recovering` (incohérence label/status)
- cutoffs de split ajustés 25200s→25213s / 30600s→30600s pour ne pas couper
  un événement en deux
- 55 429 / 11 814 / 11 845 fenêtres train/val/test, forme `(20, 21)`

## ⚠️ Point à trancher avant la modélisation

Avec le split chronologique, les classes très rares (`under_frequency`,
`source_outage`, `voltage_sag`, `over_frequency` — 11 à 22 occurrences sur
tout le run) ne sont pas forcément présentes dans les trois splits : par
exemple `source_outage`/`under_frequency` n'apparaissent que dans train,
`voltage_sag` seulement dans val/test. Le script logue un `ATTENTION` avec
le nombre exact de fenêtres concernées à l'étape encoding plutôt que de les
avaler silencieusement. Ce n'est pas un bug — c'est la conséquence directe
d'un split temporel strict sur des classes très rares. À décider :
accepter, passer à un split stratifié par événement, ou générer plus de
données pour ces classes côté simulateur.

## Structure des fonctions dans le fichier

```
compute_event_blocks() / event_table()   utilitaires partagés
run_cleaning()                            Étape 1
run_validation()                          Étape 2
run_feature_selection()                   Étape 3
run_split()                               Étape 4
run_windowing()                           Étapes 5-6
run_encoding()                            Étape 7
run_scaling()                             Étape 8
main()                                    orchestrateur (appelle tout dans l'ordre)
```

Chaque fonction peut aussi être appelée seule dans un notebook :
```python
from preprocessing_pipeline import load_params, run_cleaning
params = load_params("params.yaml")
df_clean = run_cleaning(params)
```
