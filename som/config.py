"""
Configuration du projet SOM.
"""
from dataclasses import dataclass


@dataclass
class SOMConfig:
    # Dataset
    csv_path: str = "run_dataset.csv"

    # Prétraitement
    epsilon: float = 1e-6
    normalize: bool = True

    # Filtrage
    min_distance: float = 2.0
    max_distance: float = 50.0
    
    # Découpage
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    split_seed:int = 123
    split_group_column:str = "id_run"

    # SOM
    map_width: int = 24
    map_height: int = 3

    sigma: float = 2.0
    learning_rate: float = 0.5

    random_seed: int = 42
    iterations: int = 100_000
    
    # Suivi de l'apprentissage
    evaluation_interval: int = 1_000
    
    # Limite le cout du calcul des courbes.
    # Les métriques finales seront calculées sur tous les échantillons.
    curve_sample_size: int = 20_000

    # Résultats
    output_directory: str = "results"