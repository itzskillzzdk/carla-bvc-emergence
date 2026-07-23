import json
from pathlib import Path

from .config import SOMConfig
from .dataset import load_dataset, split_dataset
from .model import SOM
from .preprocessing import preprocess_dataset
from .visualization import (
    plot_quantization_error_history,
    plot_topographic_error_history,
    save_all_visualizations,
)


def main():
    config = SOMConfig()

    output_directory = Path(
        config.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------
    # Chargement et découpage
    # ----------------------------------------------------

    dataframe = load_dataset(config.csv_path)

    (train_dataframe, validation_dataframe, test_dataframe) = split_dataset(dataframe, config)

    print(f"Brut       : {len(dataframe)}")
    print(f"Train      : {len(train_dataframe)}")
    print(f"Validation : {len(validation_dataframe)}")
    print(f"Test       : {len(test_dataframe)}")

    # ----------------------------------------------------
    # Prétraitement
    # ----------------------------------------------------

    (X_train, normalization, filtered_train) = preprocess_dataset(train_dataframe, config)

    (X_validation, _, filtered_validation) = preprocess_dataset(validation_dataframe, config)

    (X_test, _, filtered_test) = preprocess_dataset(test_dataframe, config)

    print()
    print(
        f"Après filtrage : "
        f"train={len(X_train)}, "
        f"validation={len(X_validation)}, "
        f"test={len(X_test)}"
    )

    # ----------------------------------------------------
    # Entraînement
    # ----------------------------------------------------

    som = SOM(config)

    som.fit(
        X_train=X_train,
        X_validation=X_validation,
        normalization=normalization,
        verbose=True,
    )

    # ----------------------------------------------------
    # Évaluation
    # ----------------------------------------------------

    train_metrics = som.evaluate(X_train)

    validation_metrics = som.evaluate(X_validation)

    # Test seulement après la fin de l'apprentissage.
    test_metrics = som.evaluate(X_test)

    all_metrics = {
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    print()
    print("Métriques finales :")

    for split_name, metrics in all_metrics.items():
        print(
            f"{split_name:>10s} | "
            f"QE={metrics['quantization_error']:.6f} | "
            f"TE={metrics['topographic_error']:.6f} | "
            f"occupation={100 * metrics['unit_occupancy']:.2f}%"
        )

    with (
        output_directory / "metrics.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    with (
        output_directory / "training_history.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(som.training_history, file, indent=2)

    # ----------------------------------------------------
    # Sauvegarde
    # ----------------------------------------------------

    som.save()

    filtered_train.to_csv(
        output_directory / "train_dataset.csv",
        index=False,
    )

    filtered_validation.to_csv(
        output_directory / "validation_dataset.csv",
        index=False,
    )

    filtered_test.to_csv(
        output_directory / "test_dataset.csv",
        index=False,
    )

    # ----------------------------------------------------
    # Visualisations intrinsèques de la SOM
    # ----------------------------------------------------

    save_all_visualizations(
        som,
        X_train,
        output_directory,
    )

    # ----------------------------------------------------
    # Courbes d'apprentissage
    # ----------------------------------------------------

    figures_directory = (
        output_directory / "figures"
    )

    plot_quantization_error_history(
        som.training_history,
        figures_directory
        / "quantization_error_history.png",
        test_metrics=test_metrics,
    )

    plot_topographic_error_history(
        som.training_history,
        figures_directory
        / "topographic_error_history.png",
        test_metrics=test_metrics,
    )


if __name__ == "__main__":
    main()