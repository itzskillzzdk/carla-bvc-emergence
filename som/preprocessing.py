"""
Prétaitement des entrées du SOM.
"""
from __future__ import annotations
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import SOMConfig

NormalizationParameters = Dict[str, Any]

def preprocess_dataset(df: pd.DataFrame, config: SOMConfig) -> tuple[np.ndarray, NormalizationParameters, pd.DataFrame]:
    """
    Transforme les POIs en vecteurs :
    
        [
            cos(theta),
            sin(theta),
            log(distance)
        ]
    
    Returns
    -------
    X: Matrice de taille (N,3)
    normalization: Paramètres permettant d'inverser la transformation de distance.
    filtered_df: Dataframe contenant uniquement les lignes conservées, avec le même ordre que X.
    """
    
    required_columns = [
        "allocentric_azimuth_deg",
        "estimated_distance_m",
    ]
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour le prétraitement : {sorted(missing)}")
    filtered_df = df.copy()
    
    # Ne retirer une ligne que si une variable réellement utilisée est absente.
    filtered_df = filtered_df.dropna(subset=required_columns)
    
    # Conversion robuste au cas où le CSV contienderait des chaines.
    for column in required_columns:
        filtered_df[column] = pd.to_numeric(filtered_df[column], errors="coerce")
    filtered_df = filtered_df.dropna(subset=required_columns)
    
    distance_mask = (
        (filtered_df["estimated_distance_m"] >= config.min_distance)
        & (filtered_df["estimated_distance_m"] <= config.max_distance)
    )
    
    filtered_df = filtered_df.loc[distance_mask].reset_index(drop=True)
    
    if filtered_df.empty:
        raise ValueError("Aucune observation restante après le filtrage des distances.")
    
    # --------------------
    # Angle allocentrique
    # --------------------
    theta_rad = np.deg2rad(filtered_df["allocentric_azimuth_deg"].to_numpy(dtype=np.float64))
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # --------------------
    # Distance
    # --------------------
    distance = filtered_df["estimated_distance_m"].to_numpy(dtype=np.float64)
    log_distance = np.log(distance + config.epsilon)
    
    log_min = np.log(config.min_distance + config.epsilon)
    log_max = np.log(config.max_distance + config.epsilon)
    
    if config.normalize:
        denominator = log_max - log_min
        if denominator <= 0:
            raise ValueError("max_distance doit être strictement supérieur à min_distance.")
        distance_feature = (log_distance - log_min) / denominator
    else:
        distance_feature = log_distance
    normalization: NormalizationParameters = {
        "epsilon": config.epsilon,
        "normalized": config.normalize,
        "log_min": float(log_min),
        "log_max": float(log_max),
        "min_distance": config.min_distance,
        "max_distance": config.max_distance,
    }
    
    # --------------------
    # Vecteur d'entrée
    # --------------------
    X = np.column_stack(
        (
            cos_theta,
            sin_theta,
            distance_feature,
        )
    ).astype(np.float64)
    
    if not np.all(np.isfinite(X)):
        raise ValueError("Le prétraitement a produit des valeurs non finies.")
    
    return X, normalization, filtered_df