"""
Interface de haut niveau du modèle SOM.

Cette classe encapsule MiniSom et expose des méthodes orientées vers l'interprétation BVC_like des unités apprises.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from minisom import MiniSom

from .config import SOMConfig

NormalizationParameters = Mapping[str, Any]

class SOM:
    """
    Self-Organizing Map spécialisée pour les entrées :
    
        [
            cos(theta),
            sin(theta),
            log_distance
        ]
        
    Les deux premiers poids représentent une direction allocentrique.
    Le troisième représente une préférence de distance.
    """
    
    INPUT_DIMENSION = 3
    
    def __init__(self, config: SOMConfig):
        self.config = config
        
        self.model = MiniSom(
            x=config.map_width,
            y=config.map_height,
            input_len=self.INPUT_DIMENSION,
            sigma=config.sigma,
            learning_rate=config.learning_rate,
            neighborhood_function="gaussian",
            activation_distance="euclidean",
            topology="rectangular",
            random_seed=config.random_seed,
        )
        
        self.normalization: dict[str, Any] | None = None
        self._is_fitted = False
        self.history = None
    
    def _evaluation_subset(
        self,
        X,
        maximum_size,
        seed
    ):
        """
        Retourne un sous-ensemble fixe pour le calcul des courbes.
        
        Utiliser le même sous-ensemble à chaque checkpoint évite que les variations de la courbe proviennent d'un échantillonnage différent.
        """
        if maximum_size is None or len(X) <= maximum_size:
            return X
        rng = np.random.RandomState(seed)
        indexes = rng.choice(
            len(X),
            size=maximum_size,
            replace=False
        )
        return X[indexes]
    
    # ====================
    # Validation interne
    # ====================
    def _validate_X(self, 
        X:np.ndarray, 
        *, 
        allow_empty: bool = False
    ) -> np.ndarray:
        """
        Valide et convertit une matrice d'entrée.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"X Doitêtre une matrice 2D, reçu shape={X.shape}.")
        if X.shape[1] != self.INPUT_DIMENSION:
            raise ValueError(f"X doit avoir {self.INPUT_DIMENSION} variables, reçu shape={X.shape}.")
        if not allow_empty and len(X) == 0:
            raise ValueError("X ne doit pas être vide.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contient des valeurs NaN ou infinies.")
        
        return X
    
    def _validate_sample(self, sample: np.ndarray) -> np.ndarray:
        """
        Valide un unique vecteur d'entrées.
        """
        sample = np.asarray(sample, dtype=np.float64)
        
        if sample.shape != (self.INPUT_DIMENSION,):
            raise ValueError(f"Un échantillon doit avoir la shape ({self.INPUT_DIMENSION,},), reçu {sample.shape}.")
        if not np.all(np.isfinite(sample)):
            raise ValueError("L'échantillon contient des valeurs non finies.")
        return sample
    
    def _check_fitted(self) -> None:
        """
        Vérifie que la SOM a été entrainée.
        """
        if not self._is_fitted:
            raise RuntimeError("La SOM n'a pas encore été entrainée.")
    
    # ====================
    # Entrainement
    # ====================
    
    def fit(
        self,
        X_train,
        X_validation=None,
        normalization=None,
        verbose=True
    ):
        """
        Entraine la SOM et enregistre les courbes d'apprentissage.
        
        X_train est utilisé pour modifier les poids.
        X_validation n'est jamais utiliser pour modifier les poids.
        """
        
        X_train = self._validate_X(X_train)
        
        if X_validation is not None:
            X_validation = self._validate_X(X_validation)
        
        if normalization is not None:
            self.normalization = dict(normalization)
        
        # Initialisation uniquement à partir du train.
        self.model.random_weights_init(X_train)
        
        # Générateur controlant l'odre des échantillons.
        training_rng = np.random.RandomState(self.config.random_seed + 1)
        
        # Sous-ensembles fixes pour les courbes.
        curve_train = self._evaluation_subset(
            X_train,
            self.config.curve_sample_size,
            self.config.split_seed + 10
        )
        
        curve_validation = None
        
        if X_validation is not None:
            curve_validation = self._evaluation_subset(
                X_validation,
                self.config.curve_sample_size,
                self.config.split_seed
            )
        
        history = {
            "iteration": [],
            "train_quantization_error": [],
            "validation_quantization_error": [],
            "train_topographic_error": [],
            "validation_topographic_error": [],
        }
        
        def record_metrics(iteration):
            train_qe = float(self.model.quantization_error(curve_train))
            train_te = float(self.model.topographic_error(curve_train))
            
            history["iteration"].append(int(iteration))
            history["train_quantization_error"].append(train_qe)
            history["train_topographic_error"].append(train_te)
            
            if curve_validation is not None:
                validation_qe = float(self.model.quantization_error(curve_validation))
                validation_te = float(self.model.topographic_error(curve_validation))
            else:
                validation_qe = None
                validation_te = None
            history["validation_quantization_error"].append(validation_qe)
            history["validation_topographic_error"].append(validation_te)
            
            if verbose:
                message = f"iteration={iteration:>7d} | train_qe={train_qe:6f} | train_te={train_te:6f}"
                if validation_qe is not None:
                    message += f" | validation_qe={validation_qe:6f} | validation_te={validation_te:6f}"
                print(message)
        # Erreur avant la première mise à jour.
        record_metrics(0)
        
        for iteration in range(self.config.iterations):
            sample_index = training_rng.randint(len(X_train))
            sample = X_train[sample_index]
            winning_unit = self.model.winner(sample)
            self.model.update(sample, winning_unit, iteration, self.config.iterations)
            completed_iterations = iteration + 1
            
            must_evaluate = (
                completed_iterations % self.config.evaluation_interval == 0
                or completed_iterations == self.config.iterations
            )
            if must_evaluate:
                record_metrics(completed_iterations)
        self.history = history
        self._is_fitted = True
        return self
    
    # ====================
    # Evaluation
    # ====================
    def evaluate(self, X):
        """
        Evalue la SOM sans modifier ses poids.
        """
        self._check_fitted()
        X = self._validate_X(X)
        dead_mask = self.dead_units(X)
        
        return {
            "number_of_samples": int(len(X)),
            "quantization_error": float(self.model.quantization_error(X)),
            "topographic_error": float(self.model.topographic_error(X)),
            "dead_units": int(dead_mask.sum()),
            "active_units": int((~dead_mask).sum()),
            "unit_occupancy": float(1.0 - dead_mask.mean())
        }
    
    # ====================
    # Inférence
    # ====================
    def winner(self, sample:np.ndarray) -> tuple[int, int]:
        """
        Retourne les coordonées du BMU d'un échantillon.
        """
        self._check_fitted()
        sample = self._validate_sample(sample)
        
        x, y = self.model.winner(sample)
        return int(x), int(y)
    
    def winners(self, X:np.ndarray) -> np.ndarray:
        """
        Retourne les coordonnées BMU de tous les échantillons.
        
        Returns
        -------
        ndarray de shape (N,2).
        
        Chaque ligne contient : [map_x, map_y]
        """
        self._check_fitted()
        X = self._validate_X(X, allow_empty=True)
        if len(X) == 0:
            return np.empty((0,2), dtype=np.int64)
        return np.asarray(
            [self.model.winner(sample) for sample in X],
            dtype=np.int64
        )
        
    # =========================
    # Interprétation BVC-like
    # =========================
    
    def preferred_directions(
        self,
        *,
        undefined_threshold: float = 1e-8,
    ) -> np.ndarray:
        """
        Retourne l'azimut allocentrique préféré de chaque unité.
        
        Returns
        -------
        Matrice (map_width, map_height), en degrés, dans l'intervalle [-180, 180[.
        
        Une unité dont les poids cos/sin ont une norme presque nulle reçoit NaN, car sa direction n'est pas interprétable.
        """
        self._check_fitted()
        
        cos_weight = self.weights[..., 0]
        sin_weight = self.weights[..., 1]
        
        direction_strength = np.hypot(cos_weight, sin_weight)
        directions = np.rad2deg(np.arctan2(sin_weight, cos_weight))
        
        # repli dans l'intervalle [-180, 180[.
        directions = ((directions + 180.0) % 360.0) - 180.0
        
        directions = directions.astype(np.float64)
        directions[direction_strength < undefined_threshold] = np.nan
        return directions
    
    def direction_selectivity(self) -> np.ndarray:
        """
        Retourne la norme des composantes directionnelles de chaque unité.
        
        Une valeur proche de 1 indique une préférence angulaire nette.
        Une valeur proche de 0 indique que la direction est mal définie, par exemple si une unité moyenne les observations provenant de direction opposées.
        """
        self._check_fitted()
        return np.hypot(self.weights[...,0], self.weights[...,1])
    
    def preferred_distances(
        self,
        normalization: NormalizationParameters | None = None,
        *,
        clip: bool = True
    ) -> np.ndarray:
        """
        Retourne la distance préférée de chaque unité, en mètres.
        
        Le troisième poids est inversé selon la tranformation utilisée dans preprocessing.py.
        
        Parameters
        ----------
        normalization: Paramètres de normalisation. Si None, la normalisation enregistrée pendant fit() ou load() est utlisée.
        clip: Lorsque la distance logarithmiqueest normalisée, limite le poids à [0, 1] avant la transformation inverse.
        """
        self._check_fitted()
        
        parameters = (dict(normalization) if normalization is not None else self.normalization)
        if parameters is None:
            raise ValueError("Aucune normalisation disponible. Passe-la à fit(), preferred_distance(), ou charge un modèle qui la contient.")
        required_keys = {
            "epsilon",
            "normalized",
            "log_min",
            "log_max",
        }
        missing = required_keys - set(parameters)
        
        if missing:
            raise ValueError(f"Paramètres de normalisation manquants : {sorted(missing)}")
        
        distance_weight = self.weights[...,2].astype(np.float64)
        
        if parameters["normalized"]:
            if clip:
                distance_weight = np.clip(distance_weight, 0.0, 1.0)
            log_distance = distance_weight * (parameters["log_max"] - parameters["log_min"]) - parameters["log_min"]
        else:
            log_distance = distance_weight
        distance = np.exp(log_distance) - parameters["epsilon"]
        
        # Les valeurs négatifs n'ont pas de sens physique
        return np.maximum(distance, 0.0)
    
    # =========================
    # Activité des unités
    # =========================
    def activation_histogram(
        self,
        X:np.ndarray,
        *,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Compte le nombre d'échantillons associés à chaque unité.
        
        Parameters
        ----------
        normalize: Si True, retourne la proportion d'échantillons au lieu du nombre brut.
        
        Returns
        -------
        Matrice (map_width, map_height).
        """
        self._check_fitted()
        X = self._validate_X(X, allow_empty=True)
        
        histogram = np.zeros(
            (self.config.map_width, self.config.map_height),
            dtype=np.int64
        )
        
        if len(X) == 0:
            if normalize:
                return histogram.astype(np.float64)
            return histogram
        bmus = self.winners(X)
        
        np.add.at(
            histogram,
            (bmus[:, 0], bmus[:, 1]),
            1
        )
        if normalize:
            return histogram.astype(np.float64) / len(X)
        
        return histogram
    
    def dead_units(self, X: np.ndarray) -> np.ndarray:
        """
        Retourne un masque booléen indiquant les unités mortes.
        
        True signifie qu'aucun échantillon X n'a sélectionnée cette unité comme BMU.
        """
        histogram = self.activation_histogram(X)
        return histogram == 0
    
    def dead_unit_coordinate(self, X:np.ndarray) -> np.ndarray:
        """
        Retourne les coordonnées [map_x, map_y] des unités mortes.
        """
        return np.argwhere(self.dead_units(X))
    
    def unit_occupancy(
        self,
        X:np.ndarray
    ) -> float:
        """
        Retourne la proportion d'unités activées au moins une fois. 
        """
        dead_mask = self.dead_units(X)
        return float(1.0 - dead_mask.mean())
    
    # =========================
    # Structure topologique
    # =========================
    def umatrix(
        self,
        *,
        scaling: str = "mean"
    ) -> np.ndarray:
        """
        Retourne la U-Matrix de MiniSom.
        
        Parameters
        ----------
        scaling: "mean" ou "sum"
        """
        self._check_fitted()
        if scaling not in ["mean", "sum"]:
            raise ValueError("scaling doit valoir 'mean' ou 'sum'.")
        return self.model.distance_map(scaling=scaling).copy()
    
    # =========================
    # Métriques
    # =========================
    
    def quantization_error(
        self,
        X: np.ndarray
    ) -> float:
        self._check_fitted()
        X = self._validate_X(X)
        return float(self.model.quantization_error(X))
    
    def topographic_error(
        self,
        X: np.ndarray
    ) -> float:
        """
        Mesure la préservation de la topologie par la SOM.
        """
        self._check_fitted()
        X = self._validate_X(X)
        return float(self.model.topographic_error(X))
    
    # =========================
    # Poids
    # =========================
    @property
    def weights(self) -> np.ndarray:
        """
        Retourne une copie des poids de forme (map_width, map_height, 3)
        """
        return self.model.get_weights().copy()
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def training_history(self):
        if self.history is None:
            raise RuntimeError("Aucun historique disponible. Le modèle doit d'abord être entrainé.")
        return self.history
    
    # =========================
    # Sauvegarde
    # =========================
    def save(self, filename: str | Path | None = None) -> Path:
        """
        Sauvegarde le modèle, sa configuration et sa normalisation.
        """
        self._check_fitted()
        
        if filename is None:
            output_directory = Path(self.config.output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            path = output_directory / "som.joblib"
        else:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "SOM",
            "config": asdict(self.config),
            "normalization": self.normalization,
            "is_fitted": self._is_fitted,
            "history": self.history,
            "model": self.model
        }
        
        joblib.dump(payload, path)
        return path
    
    @classmethod
    def load(cls, filename: str | Path) -> SOM:
        """
        Recharge un modèle SOM complet.
        """
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        
        payload = joblib.load(path)
        required_keys = {
            "config",
            "model",
            "is_fitted"
        }
        missing = required_keys - set(payload)
        if missing:
            raise ValueError(f"Fichier du modèle invalide, clés absentes : {sorted(missing)}")
        
        config = SOMConfig(**payload["config"])
        instance = cls(config)
        
        instance.model = payload["model"]
        instance.normalization = payload.get("normalization")
        instance.history = payload.get("history")
        instance._is_fitted = bool(payload["is_fitted"])
        return instance