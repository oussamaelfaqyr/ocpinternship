# OCP Youssoufia — SCADA Digital Twin & AI Protection System

> **Système Intelligent de Supervision & Diagnostic des Défauts Réseau (60 kV HTB)**  
> **Projet de Fin d'Études (PFE) — Groupe OCP (Site Gantour / Youssoufia)**  
> **Auteur :** Oussama Elfaqyr ([oussamaelfaqyr@gmail.com](mailto:oussamaelfaqyr@gmail.com))

---

## 📖 Guides Disponibles

* 📘 **Guide Non-Technique (Utilisateurs & Opérateurs) :** [`USER_GUIDE.md`](USER_GUIDE.md) — Explications pas-à-pas simplifiées de chaque écran, des indicateurs et de la procédure de démonstration.
* 🛠️ **Fiche Technique Système :** Accessible directement depuis l'application via le bouton **`System Details`**.

---

## ⚡ Démarrage Rapide

### 1. Installation des Dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancement du Serveur SCADA (Flask + Web UI)
```bash
python server.py
```
*Le serveur démarre et ouvre automatiquement l'interface sur :* **`http://127.0.0.1:8080`**

---

## 🧠 Modèle d'Intelligence Artificielle (Production)

| Paramètre | Spécification |
|---|---|
| **Architecture** | **Deep Gated Recurrent Unit (GRU)** — 2 Couches, 256 Unités Cachées |
| **Optimisation** | **Optuna HPO** (30 Trials, TPE Sampler + Median Pruner) |
| **Poids Modèle** | `ai/models/saved_models/gru_optuna_e3b_best.pt` |
| **Fenêtre Temporelle** | 20 timesteps &times; 35 features physiques & ratios normalisés |
| **Exactitude Globale (Accuracy)** | **`82.35 %` 🏆 (Record Absolu)** |
| **F1-Score Pondéré** | **`82.13 %` 🏆** |
| **Validation Macro-F1** | **`77.15 %` 🏆** |
| **Explicabilité IA** | **Gradient &times; Input (SHAP)** calculé en temps réel (< 1 ms) |

### 8 Classes Diagnostiquées par l'IA

| Index | Classe | Nom du Défaut | Sévérité | F1-Score Validé |
|---|---|---|---|:---:|
| 0 | `normal` | Fonctionnement Nominal | Normal | **0.91** |
| 1 | `lg_fault` | Court-Circuit Ligne-Terre (LG) | CRITICAL | **0.62** |
| 2 | `ll_fault` | Court-Circuit Entre Phases (LL) | CRITICAL | **0.57** |
| 3 | `over_frequency` | Sur-Fréquence Réseau (ANSI 81O) | WARNING | **0.94** |
| 4 | `overload` | Surcharge Thermique Transformateur (ANSI 49) | WARNING | **0.57** (Précision 64%) |
| 5 | `transformer_trip` | Déclenchement Transformateur (Buchholz/Diff) | CRITICAL | **0.65** |
| 6 | `under_frequency` | Sous-Fréquence Réseau (ANSI 81U) | WARNING | **0.70** |
| 7 | `voltage_sag` | Creux de Tension Réseau (ANSI 27) | WARNING | **0.93** |

---

## 🛠️ Pipeline d'Ingénierie & Entraînement

```bash
# 1. Prétraitement des 35 features physiques
cd "ai/data processing"
python preprocessing_pipeline_v2.py

# 2. Entraînement et Optimisation HPO (Optuna 30 trials)
cd "../models"
python train_hpt_e3b.py
```

