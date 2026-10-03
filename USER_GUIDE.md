# Guide d'Utilisation — Jumeau Numérique SCADA & IA OCP

> **Système Intelligent de Supervision et de Diagnostic des Défauts Réseau (60 kV HTB)**  
> **Projet de Fin d'Études (PFE) — Groupe OCP (Site Gantour / Youssoufia)**  
> **Auteur :** Oussama Elfaqyr  
> **Contact :** [oussamaelfaqyr@gmail.com](mailto:oussamaelfaqyr@gmail.com)

---

## 1. Présentation Simple du Projet

### Qu'est-ce que ce système ?
Ce projet est un **Jumeau Numérique (Digital Twin)** du réseau électrique haute tension (60 000 Volts) qui alimente les installations minières et industrielles de l'**OCP à Youssoufia**.

Il remplit deux missions principales :
1. **Supervision en direct (SCADA) :** Il surveille en continu les tensions, courants, puissances et fréquences des **11 sous-stations** de distribution électrique du site.
2. **Détection préventive par Intelligence Artificielle (Deep Learning) :** Un algorithme d'IA analyse les signaux électriques toutes les secondes pour identifier instantanément les anomalies (courts-circuits, surcharges de transformateurs, chutes de tension, déviations de fréquence) et conseiller l'opérateur sur l'action à entreprendre.

---

## 2. Prérequis & Installation Rapide

### Ce dont vous avez besoin :
* Un ordinateur équipé de **Python 3.9 à 3.12**
* Un navigateur Web moderne (Google Chrome, Microsoft Edge, Mozilla Firefox, etc.)

### Installation en 2 étapes :

1. **Installer les bibliothèques requises :**
   Ouvrez votre terminal (Invite de commandes Windows ou PowerShell) dans le dossier du projet et tapez :
   ```bash
   pip install -r requirements.txt
   ```

2. **Démarrer l'application :**
   ```bash
   python server.py
   ```
   *Le serveur démarre immédiatement et ouvre automatiquement votre navigateur sur l'adresse :*  
   👉 **`http://127.0.0.1:8080`**

---

## 3. Guide Visuel de l'Interface

L'interface se compose d'un bandeau de commande supérieur, d'une barre d'indicateurs clés (KPIs), d'un sélecteur de sous-station et de **4 onglets de travail**.

---

### A. La Barre de Contrôle Supérieure (Header)

* **Logo & Statut Réseau :** Un voyant vert `SYSTEM RUNNING` indique que la simulation électrique fonctionne normalement en temps réel.
* **Bouton `Pause / Run` :** Permet de figer la simulation à tout moment pour inspecter une mesure précise, puis de la relancer.
* **Bouton `System Details` :** Ouvre la fiche technique complète du projet (architecture du réseau, caractéristiques détaillées du modèle d'IA, tableau comparatif des performances et coordonnées de l'auteur).
* **Bouton Rouge `Inject Fault` :** Permet de simuler un incident sur le réseau pour tester les réactions de l'IA (court-circuit phase-terre, surcharge thermique, baisse de tension, etc.).

---

### B. Le Bandeau d'Indicateurs Clés (KPI Strip)

Affiche en temps réel les grandeurs électriques fondamentales de la sous-station sélectionnée :
* **HV Bus Voltage :** Tension haute tension (normale : ~60.0 kV).
* **LV Bus Voltage :** Tension de distribution procédé (normale : ~5.5 kV).
* **Substation Current :** Courant traversant le transformateur (en Ampères).
* **Active Power (P) & Reactive Power (Q) :** Puissances active (MW) et réactive (Mvar).
* **Grid Frequency :** Fréquence du réseau national (normale : 50.00 Hz).
* **Transformer Load :** Pourcentage de charge du transformateur (alerte si > 80%, critique si > 100%).
* **AI Engine Model :** Sélecteur du modèle d'IA actif (`GRU Optuna E3-B` sélectionné par défaut avec une exactitude de 82.35%).

---

### C. Le Sélecteur de Sous-Stations

Situé juste en dessous des KPIs, il permet de basculer en un clic entre les **11 postes du site OCP** :
* *Laverie/Séchage SN1 & SN2*
* *Mine Mzinda DIS TR*
* *PSF Mine Bouchane SN6*
* *Recette 3 SN3 & Recette 9 SN9*
* *SSP Local ST1 & ST2*
* *U Calcination SN1, SN2 & SN3*

*(Un point rouge apparaît automatiquement sur la sous-station concernée dès qu'une anomalie s'y produit).*

---

## 4. Les 4 Onglets de l'Application

### Onglet 1 : `System State & SCADA Status`
* **État Global du Système :** Indique clairement si le réseau est en état nominal (*Normal Operation*) ou si un défaut est en cours, avec le nom du poste touché.
* **Tableau Général de Télémétrie :** Présente une vue synoptique des 11 sous-stations avec toutes leurs valeurs de tension, courant, puissance et charge thermique actualisées chaque seconde.

---

### Onglet 2 : `Signal Curves & Live Trends`
Affiche **4 graphiques dynamiques en temps réel** pour la sous-station active :
1. **Courbes de Tensions :** Évolution des tensions 60 kV et 5.5 kV.
2. **Courbes de Courant & Charge :** Évolution de l'intensité (A) et de l'échauffement (%).
3. **Courbes de Puissance :** Puissance active (MW) et réactive (Mvar).
4. **Courbe de Fréquence :** Stabilité du réseau électrique en Hertz (Hz).

---

### Onglet 3 : `AI Diagnosis & Protection` (Le Cœur Intelligent)
Cet onglet détaille l'analyse effectuée par l'Intelligence Artificielle :
* **Carte de Diagnostic IA :** Affiche la classe de défaut prédite et le niveau de confiance de l'IA (ex: `98.5%`).
* **Action Recommandée à l'Opérateur :** Conseils opérationnels clairs générés automatiquement (ex: *"Délestage requis sur départ non prioritaire"*, *"Vérifier relais différentiel transformateur"*).
* **Distribution des Probabilités :** Graphique à barres comparant la vraisemblance des 8 états possibles du réseau.
* **Explicabilité IA (SHAP Feature Importance) :** Un graphique qui explique **pourquoi** l'IA a pris sa décision :
  * *Barre verte :* La mesure pousse l'IA à confirmer le défaut.
  * *Barre rouge :* La mesure retient l'IA d'autres hypothèses.
* **Inspecteur de Features :** Affiche les 35 grandeurs et ratios physiques calculés en temps réel.

---

### Onglet 4 : `Active Alarms & SOE Event Log`
* **Alarmes Actives :** Liste priorisée des anomalies en cours (CRITICAL en rouge, WARNING en orange).
* **Journal des Événements (SOE - Sequence of Events) :** Historique horodaté à la seconde près de tous les événements survenus (déclenchements, variations de charge, alertes IA). Comprend une barre de recherche rapide.

---

## 5. Comment Réaliser une Démonstration Pas-à-Pas

Voici le scénario idéal pour tester et présenter l'application :

1. **Vérifier l'état normal :** L'écran affiche un voyant vert et l'IA indique `Normal Operation` à 99% de confiance.
2. **Injecter un défaut :**
   * Cliquez sur le bouton rouge **`Inject Fault`**.
   * Choisissez une sous-station (ex: *Mine Mzinda DIS TR*).
   * Choisissez un type d'incident (ex: *Thermal Overload* ou *Line-to-Ground Fault*).
   * Cliquez sur **`Inject Fault`**.
3. **Observer la réaction en direct :**
   * Le voyant passe à l'orange/rouge.
   * L'onglet **AI Diagnosis** bascule immédiatement sur le défaut identifié.
   * Le graphique **SHAP** montre instantanément quelle grandeur (ex: courant anormal, chute de tension) a déclenché l'alerte.
   * L'onglet **Active Alarms** enregistre l'alarme et affiche l'action recommandée.
4. **Revenir à la normale :** Cliquez sur **`Clear Active Fault`** dans le modal pour réinitialiser le réseau.

---

## 6. Support & Contact

Pour toute question, démonstration ou information complémentaire relative à ce projet :

* **Auteur & Développeur :** Oussama Elfaqyr
* **Email :** [oussamaelfaqyr@gmail.com](mailto:oussamaelfaqyr@gmail.com)
* **Organisme :** OCP Group — Direction de l'Exploitation Gantour / Youssoufia
