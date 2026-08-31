# OCP Youssoufia — SCADA Digital Twin

Digital twin du réseau 60 kV HTB de l'OCP Youssoufia.
Simule le réseau électrique, injecte des pannes, et les classifie en temps réel via un modèle LSTM.

---

## Structure du projet

```
ocp_internship/
│
├── server.py                          ← Serveur Flask (API REST + UI Web)
├── README.md
│
├── network_model/
│   └── build_network.py               ← Construction du réseau PandaPower 60 kV
│
├── fault_simulator/
│   ├── fault_simulator.py             ← Moteur de simulation pas-à-pas
│   ├── dataset_generator.py           ← Générateur de dataset d'entraînement
│   ├── generate.py                    ← Script de génération (point d'entrée)
│   ├── generate_dataset_pandapower.py ← Générateur alternatif PandaPower
│   ├── __init__.py
│   ├── lg_fault.py / ll_fault.py      ← Types de pannes
│   ├── overload.py / outage_fault.py
│   ├── voltage_sag.py / frequency_fault.py
│   ├── load_profile.py                ← Profil de charge journalier
│   ├── protection_engine.py           ← Logique de protection
│   ├── sc_engine.py                   ← Calcul court-circuit
│   ├── sensor_effects.py              ← Bruit capteur réaliste
│   └── youssoufia_pandapower_*.csv    ← Datasets simulés bruts
│
├── ai/
│   ├── fault_classifier.py            ← Classificateur LSTM (interface serveur)
│   ├── __init__.py
│   ├── data processing/
│   │   ├── preprocessing_pipeline_v2.py   ← Pipeline de prétraitement
│   │   ├── params.yaml / dvc.yaml / dvc.lock
│   │   └── data/processed/
│   │       ├── scaler.joblib          ← StandardScaler entraîné
│   │       ├── label_vocab.json       ← Mapping label → index
│   │       ├── train_final.npz        ← Séquences (20 × 22 features)
│   │       ├── val_final.npz
│   │       └── test_final.npz
│   ├── dataset/
│   │   ├── train.csv                  ← Dataset brut d'entraînement
│   │   └── test_val.csv
│   └── models/
│       ├── train_lstm_hpt.py          ← Entraînement LSTM + Optuna  ← ACTIF
│       ├── train_baseline.py / train_hpt.py / train_cnn_hpt.py
│       ├── benchmark_inference.py     ← Comparaison latence LSTM/GRU/CNN
│       ├── mlflow.db / mlruns/        ← Tracking MLflow
│       └── saved_models/
│           ├── lstm_optuna_best.pt    ← MODELE EN PRODUCTION
│           ├── gru_optuna_best.pt / cnn_optuna_best.pt
│           └── random_forest.pkl
│
├── dashboard/
│   ├── streamlit_app.py               ← Application Streamlit
│   ├── .streamlit/                    ← Config thème
│   └── components/
│       ├── ai_panel.py / ai_confidence.py
│       ├── charts.py / alarms.py
│       ├── network_table.py / sld_viewer.py
│       └── soe_log.py / dataset_stats.py / common.py
│
├── notebooks/
│   ├── eda_notebook.ipynb             ← Analyse exploratoire
│   └── youssoufia_eda.ipynb
│
└── static/                            ← Frontend Web (HTML5/CSS3/JS)
    ├── index.html / app.js / styles.css
```

---

## Lancer le serveur SCADA (Flask + Web UI)

```bash
python server.py
```

Le navigateur s'ouvre automatiquement sur http://127.0.0.1:8080

## Lancer le dashboard analytique (Streamlit)

```bash
cd dashboard
streamlit run streamlit_app.py
```

---

## Modèle IA — LSTM

| Paramètre    | Valeur |
|---|---|
| Fichier      | ai/models/saved_models/lstm_optuna_best.pt |
| Input        | (1, 20, 22) — 20 timesteps x 22 features |
| Output       | 8 classes de pannes |
| Val Macro-F1 | 0.6987 |
| hidden_size  | 128 |
| num_layers   | 2 |

### Classes détectées

| Index | Label | Sévérité |
|---|---|---|
| 0 | normal | — |
| 1 | lg_fault | CRITICAL |
| 2 | ll_fault | CRITICAL |
| 3 | over_frequency | WARNING |
| 4 | overload | WARNING |
| 5 | transformer_trip | CRITICAL |
| 6 | under_frequency | WARNING |
| 7 | voltage_sag | WARNING |

---

## Réentraîner le modèle LSTM

```bash
# 1. Regénérer le dataset
cd fault_simulator
python generate.py

# 2. Prétraitement
cd "../ai/data processing"
python preprocessing_pipeline_v2.py

# 3. Entraînement LSTM avec Optuna
cd "../models"
python train_lstm_hpt.py
```
