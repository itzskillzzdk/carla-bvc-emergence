from tkinter import N
from typing import Tuple
import pandas as pd
import numpy as np

from .config import SOMConfig


EXPECTED_COLUMNS = [
    "timestamp",
    "id_run",
    "frame_id",
    "poi_id",
    "town",
    "road_id",
    "lane_id",
    "length",
    "mean_curvature",
    "allocentric_azimuth_deg",
    "estimated_distance_m",
    "vehicle_speed_ms",
    "fps",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Charge le dataset SOM.
    """

    df = pd.read_csv(csv_path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing)}"
        )

    return df

def split_dataset(
    df: pd.DataFrame,
    config: SOMConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sépare le dataset en train, validation et test.
    
    Le découpage se fait prioritairement par id_run afin d'éviter qu'un même run soit présent dans plusieurs ensembles.
    """
    ratios_sum = (
        config.train_ratio
        + config.validation_ratio
        + config.test_ratio
    )
    
    if not np.isclose(ratios_sum, 1.0):
        raise ValueError("train_ratio + validation_ratio + test_ratio doit être égal à 1.")
    
    group_column = config.split_group_column
    groups = df[group_column].dropna().unique().copy()
    rng = np.random.RandomState(config.split_seed)
    rng.shuffle(groups)
    
    number_of_groups = len(groups)
    number_of_test_groups = max(1, int(round(number_of_groups * config.test_ratio)))
    number_of_validation_groups = max(1, int(round(number_of_groups * config.validation_ratio)))
    
    test_groups = groups[:number_of_test_groups]
    validation_groups = groups[number_of_test_groups:number_of_test_groups + number_of_validation_groups]
    train_groups = groups[number_of_test_groups + number_of_validation_groups:]
    
    train_df = df[df[group_column].isin(train_groups)].copy()
    validation_df = df[df[group_column].isin(validation_groups)].copy()
    test_df = df[df[group_column].isin(test_groups)].copy()
    
    # print(f"Groupes train      : {sorted(train_groups.tolist())}")
    # print(f"Groupes validation : {sorted(validation_groups.tolist())}")
    # print(f"Groupes test       : {sorted(test_groups.tolist())}")
    
    if train_df.empty:
        raise ValueError("Le jeu d'entrainement est vide.")
    if validation_df.empty:
            raise ValueError("Le jeu de validation est vide.")
    if test_df.empty:
            raise ValueError("Le jeu de test est vide.")
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )