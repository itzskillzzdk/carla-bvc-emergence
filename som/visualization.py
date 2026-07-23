"""
Visualisation des résultats du modèle SOM.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import SOM


def _plot_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    output_path: str | Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    integer_annotations: bool = False
) -> Path:
    """
    Trace une matrice de la SOM.

    La transposition permet d'afficher :
        - map_width sur l'axe horizontal ;
        - map_height sur l'axe vertical.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(13, 4))

    image = ax.imshow(
        matrix.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("Coordonnée x de la SOM")
    ax.set_ylabel("Coordonnée y de la SOM")

    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[1]))

    fig.colorbar(
        image,
        ax=ax,
        label=colorbar_label,
    )

    if annotate:
        for map_x in range(matrix.shape[0]):
            for map_y in range(matrix.shape[1]):
                value = matrix[map_x, map_y]

                if not np.isfinite(value):
                    label = "NaN"
                elif integer_annotations:
                    label = f"{int(value)}"
                else:
                    label = f"{value:.1f}"

                ax.text(
                    map_x,
                    map_y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def plot_preferred_directions(
    som: SOM,
    output_path: str | Path
) -> Path:
    """
    Trace la direction allocentrique préférée de chaque unité.
    """

    return _plot_matrix(
        som.preferred_directions(),
        title="Directions allocentriques préférées",
        colorbar_label="Azimut préféré [°]",
        output_path=output_path,
        cmap="twilight_shifted",
        vmin=-180.0,
        vmax=180.0,
        annotate=True,
    )


def plot_direction_selectivity(
    som: SOM,
    output_path: str | Path
) -> Path:
    """
    Trace la force de la préférence directionnelle.
    """

    return _plot_matrix(
        som.direction_selectivity(),
        title="Sélectivité directionnelle",
        colorbar_label="Norme des poids directionnels",
        output_path=output_path,
        cmap="viridis",
        vmin=0.0,
        annotate=True,
    )


def plot_preferred_distances(
    som: SOM,
    output_path: str | Path
) -> Path:
    """
    Trace la distance préférée de chaque unité.
    """

    return _plot_matrix(
        som.preferred_distances(),
        title="Distances préférées",
        colorbar_label="Distance préférée [m]",
        output_path=output_path,
        cmap="viridis",
        annotate=True,
    )


def plot_activation_histogram(
    som: SOM,
    X: np.ndarray,
    output_path: str | Path
) -> Path:
    """
    Trace le nombre de fois où chaque unité est BMU.
    """

    return _plot_matrix(
        som.activation_histogram(X),
        title="Histogramme d'activation",
        colorbar_label="Nombre d'échantillons",
        output_path=output_path,
        cmap="magma",
        annotate=True,
        integer_annotations=True,
    )


def plot_dead_units(
    som: SOM,
    X: np.ndarray,
    output_path: str | Path
) -> Path:
    """
    Trace les unités mortes.

    1 = unité morte
    0 = unité utilisée
    """

    return _plot_matrix(
        som.dead_units(X).astype(int),
        title="Unités mortes",
        colorbar_label="Unité morte",
        output_path=output_path,
        cmap="binary",
        vmin=0,
        vmax=1,
        annotate=True,
        integer_annotations=True,
    )


def plot_umatrix(
    som: SOM,
    output_path: str | Path
) -> Path:
    """
    Trace la U-Matrix.
    """

    return _plot_matrix(
        som.umatrix(),
        title="U-Matrix",
        colorbar_label="Distance moyenne aux unités voisines",
        output_path=output_path,
        cmap="bone",
        annotate=True,
    )

def plot_quantization_error_history(
    history,
    output_path,
    test_metrics=None,
):
    """
    Trace les erreurs de quantification train et validation.

    Le test est représenté uniquement par un point final.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    iterations = np.asarray(
        history["iteration"]
    )

    train_error = np.asarray(
        history["train_quantization_error"],
        dtype=float,
    )

    validation_error = np.asarray(
        history["validation_quantization_error"],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        iterations,
        train_error,
        label="Train",
    )

    if np.any(np.isfinite(validation_error)):
        ax.plot(
            iterations,
            validation_error,
            label="Validation",
        )

    if test_metrics is not None:
        ax.scatter(
            [iterations[-1]],
            [
                test_metrics[
                    "quantization_error"
                ]
            ],
            marker="x",
            s=100,
            label="Test final",
        )

    ax.set_title(
        "Évolution de l'erreur de quantification"
    )

    ax.set_xlabel(
        "Nombre d'itérations"
    )

    ax.set_ylabel(
        "Erreur de quantification"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path

def plot_topographic_error_history(
    history,
    output_path,
    test_metrics=None,
):
    """
    Trace les erreurs topographiques train et validation.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    iterations = np.asarray(
        history["iteration"]
    )

    train_error = np.asarray(
        history["train_topographic_error"],
        dtype=float,
    )

    validation_error = np.asarray(
        history["validation_topographic_error"],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        iterations,
        train_error,
        label="Train",
    )

    if np.any(np.isfinite(validation_error)):
        ax.plot(
            iterations,
            validation_error,
            label="Validation",
        )

    if test_metrics is not None:
        ax.scatter(
            [iterations[-1]],
            [
                test_metrics[
                    "topographic_error"
                ]
            ],
            marker="x",
            s=100,
            label="Test final",
        )

    ax.set_title(
        "Évolution de l'erreur topographique"
    )

    ax.set_xlabel(
        "Nombre d'itérations"
    )

    ax.set_ylabel(
        "Erreur topographique"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path



def save_all_visualizations(
    som: SOM,
    X: np.ndarray,
    output_directory: str | Path
) -> list[Path]:
    """
    Génère toutes les visualisations principales.
    """

    output_directory = Path(output_directory)

    figures_directory = output_directory / "figures"

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        plot_preferred_directions(
            som,
            figures_directory / "preferred_directions.png",
        ),
        plot_direction_selectivity(
            som,
            figures_directory / "direction_selectivity.png",
        ),
        plot_preferred_distances(
            som,
            figures_directory / "preferred_distances.png",
        ),
        plot_activation_histogram(
            som,
            X,
            figures_directory / "activation_histogram.png",
        ),
        plot_dead_units(
            som,
            X,
            figures_directory / "dead_units.png",
        ),
        plot_umatrix(
            som,
            figures_directory / "umatrix.png",
        ),
    ]

    return paths