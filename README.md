# carla-bvc-emergence

Étude de l'émergence de représentations neuro-mimétiques (BVC) via l'entraînement de deux modèles (SOM et Hebbien compétitif) sur des données de navigation CSV issues de CARLA.

## À propos du projet

Ce dépôt contient le code source dédié à l'étude et l'entraînement de modèles neuronaux pour la navigation spatiale autonome. L'objectif principal est d'observer si des structures bio-inspirées de type **Boundary Vector Cells (BVC)** peuvent émerger spontanément en fonction des stimuli environnementaux.

### Architectures étudiées

Pour favoriser cette émergence naturelle sans supervision stricte, le projet s'appuie sur deux implémentation distinctes :

* **SOM (Self-Organizing Map) :** Une carte topologique auto-organisatrice pour projeter les variables spatiales.
* **Apprentissage Hebbien Compétitif :** Un modèle basé sur la plasticité synaptique "Winner-Takes-All" pour spécialiser les neurones sur des repères ou frontières spécifiques.

### Jeu de données

L'apprentissage s'appuie sur un jeu de données extrait de l'environnement de simulation **CARLA**.

* Les données ont été récoltées en lançant de multiples séquences de navigation (runs) dans le simulateur.
* L'ensemble des métriques est compilé dans un fichier CSV.

> **[TODO]** Préciser ici les dimensions et caractéristiques du dataset :
>
> * Nombre total de runs / lignes dans le CSV.
> * Signification des colonnes principales (ex: `frame_id`, `poi_id`, `town`, `road_id`, `lane_id`, etc..).

---

## Instructions de lancement

### 1. Prérequis et Installation

Le projet repose sur un environnement Python. Les calculs et les cartographies neuronales s'appuient sur l'écosystème scientifique standard (NumPy, Matplotlib, Pandas, ..).

```bash
# Cloner le repository
git clone https://github.com/itzskillzzdk/carla-bvc-emergence.git
cd carla-bvc-emergence

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate # Sur Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Préparation des données

Placer le fichier CSV issu de CARLA à la racine du projet ou dans un répertoire dédié, par exemple `data/`.

L'emplacement et le nom par défaut du fichier CSV est `./run_dataset.csv` (racine du projet). Si vous souhaitez utiliser une configuration différente vous pourrez changer les propriétés liées aux chemins de fichier dans `./som/config.py` ou `./chebb/config.py`.

> **[TODO]**
>
> Le fichier étant trop lourd, préciser un lien de téléchargement externe ou la méthode pour le re-générer via CARLA.

### 3. Entraînement des modèles

Les commandes ci-dessous permettent de lancer l'apprentissage des deux modèles proposés.

**Entraînement de la SOM :**

```bash
python -m som.train
```

**Entraînement de l'apprentissahe Hebbien Compétitif (PAS ENCORE FAIT) :**

```bash
python -m chebb.train
```

---

## Résultats et Visualisations

Cette section est dédiée à l'analyse de l'activité du réseau après entrainement.

> **[TODO]**
>
> * Intégrer des captures d'écran des graphiques Matplotlib (ex: carte d'activation neurones SOM, histogrammes de spécialisations, ..).
> * Ajouter un paragraphe d'analyse : Observe-t-on des neurones qui réagissent sépcifiquement à une frontière située à une distance et un angle donnés (Signature BVC) ?

---

## Améliorations Futures

> **[TODO]**
>
> * Extraction directe des caractéristiques spatiales via un pipeline de vision par ordinateur depuis les caméras de CARLA (en se passant du CSV au profit d'une contrainte temps réel).
> * Couplage des BVC générées avec d'autres structures spatiuales (Grid Cells, Place Cells).
> * Optimisation des calculs (matriciels).

---

## Auteurs

* **Enzo VESQUE** - *Développement et recherche* - [GitHub](https://github.com/itzskillzzdk)
