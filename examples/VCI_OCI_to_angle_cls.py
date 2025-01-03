import os
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.optimize import minimize

def run_angle_cls(csv_file):
    # vci_column, oci_column = "VCI", "OCI"
    vci_column, oci_column = "variability_cloudiness_index", "overall_cloudiness_index"
    human_weather_column = "humanweather"

    df = pd.read_csv(csv_file, sep=",")
    print(df.columns)

    df = df[df[human_weather_column].isin([1, 3, 7])].sample(frac=0.2)

    k = 3
    cls_simple_labels = simple_angle_cls(df, k, vci_column, oci_column)
    cls_optimize_boundaries_labels = optimize_boundaries_angle_cls(
        df, k, vci_column, oci_column, human_weather_column
    )

    df["cls_random"] = np.random.randint(0, k, df.shape[0])
    df["cls_simple_labels"] = cls_simple_labels
    df["cls_optimize_boundaries_labels"] = cls_optimize_boundaries_labels

    show_scatter_plot(df, vci_column, oci_column, human_weather_column)
    #show_scatter_plot(df, vci_column, oci_column, "cls_random")
    #show_scatter_plot(df, vci_column, oci_column, "cls_simple_labels")
    #show_scatter_plot(df, vci_column, oci_column, "cls_optimize_boundaries_labels")

    metrics = {
        "Classifier": [
            "cls_random",
            "cls_simple_labels",
            "CLS cls_optimize_boundaries_labels",
        ],
        "Accuracy": [
            accuracy_score(df[human_weather_column], df["cls_random"]),
            accuracy_score(df[human_weather_column], df["cls_simple_labels"]),
            accuracy_score(
                df[human_weather_column], df["cls_optimize_boundaries_labels"]
            ),
        ],
        "MAE": [
            mean_absolute_error(df[human_weather_column], df["cls_random"]),
            mean_absolute_error(df[human_weather_column], df["cls_simple_labels"]),
            mean_absolute_error(
                df[human_weather_column], df["cls_optimize_boundaries_labels"]
            ),
        ],
        "MSE": [
            mean_squared_error(df[human_weather_column], df["cls_random"]),
            mean_squared_error(df[human_weather_column], df["cls_simple_labels"]),
            mean_squared_error(
                df[human_weather_column], df["cls_optimize_boundaries_labels"]
            ),
        ],
        "R2": [
            r2_score(df[human_weather_column], df["cls_random"]),
            r2_score(df[human_weather_column], df["cls_simple_labels"]),
            r2_score(df[human_weather_column], df["cls_optimize_boundaries_labels"]),
        ],
    }

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)


def simple_angle_cls(df, k_classes, vci_column, oci_column):
    angles = np.arctan2(df[vci_column], df[oci_column])
    angles_normalized = (angles + np.pi) % np.pi
    
    angle_bins = np.linspace(0, np.pi, k_classes + 1)
    cls_simple_labels = np.digitize(angles_normalized, angle_bins) - 1

    return cls_simple_labels


def optimize_boundaries_angle_cls(
    df, k_classes, vci_column, oci_column, human_weather_column
):
    def optimize_boundaries(params, angles, human_labels, k):
        boundaries = np.sort(np.clip(params, 0, np.pi))
        boundaries = np.concatenate(([0], boundaries, [np.pi]))

        complex_labels = np.digitize(angles, boundaries) - 1
        return mean_absolute_error(human_labels, complex_labels)

    angles = np.arctan2(df[oci_column], df[vci_column])
    angles_normalized = (angles + np.pi) % np.pi
    initial_boundaries = np.linspace(0, np.pi, k_classes + 1)[1:-1]
    result = minimize(
        optimize_boundaries,
        initial_boundaries,
        args=(angles_normalized, df[human_weather_column], k_classes),
        method="Nelder-Mead",
    )
    optimized_boundaries = np.sort(np.clip(result.x, 0, np.pi))
    optimized_boundaries = np.concatenate(([0], optimized_boundaries, [np.pi]))

    cls_optimize_boundaries_labels = (
        np.digitize(angles_normalized, optimized_boundaries) - 1
    )
    return cls_optimize_boundaries_labels


def show_scatter_plot(df, x_column, y_column, hue_column):
    sns.set_theme(style="ticks")

    df = df[[x_column, y_column, hue_column]].dropna()

    hue_classes = sorted(df[hue_column].unique())

    n_classes = len(hue_classes)
    grid_size = int(np.ceil(np.sqrt(n_classes)))

    x_min, x_max = df[x_column].min(), df[x_column].max()
    y_min, y_max = df[y_column].min(), df[y_column].max()

    plt.figure()
    sns.kdeplot(
        data=df,
        x=x_column,
        hue=hue_column,
        fill=True,
        common_norm=True,
        palette="tab10",
        alpha=0.5,
        thresh=0.75,
        linewidth=0,
    )
    plt.ylim(0, 0.005)
    plt.show()

    plt.figure()
    sns.kdeplot(
        data=df,
        x=y_column,
        hue=hue_column,
        fill=True,
        common_norm=True,
        palette="tab10",
        alpha=0.5,
        thresh=0.75,
        linewidth=0,
    )
    plt.ylim(0, 0.075)
    plt.show()    

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i, hue_class in enumerate(hue_classes):
        ax = axes[i]
        sub_df = df[df[hue_column] == hue_class]

        sns.scatterplot(
            data=sub_df,
            x=x_column,
            y=y_column,
            alpha=0.5,
            s=10,
            legend=False,
            color=matplotlib.colormaps.get_cmap("tab10")(i/len(hue_classes)),
            ax=ax,
        )

        try:
            sns.kdeplot(
                data=sub_df,
                x=x_column,
                y=y_column,
                fill=True,
                alpha=0.75,
                levels=10,
                thresh=0.75,
                color=matplotlib.colormaps.get_cmap("tab10")(i/len(hue_classes)),
                ax=ax,
            )
        except:
            pass

        ax.set_title(f"Class: {hue_class}(n={len(sub_df)})")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file_path = "./datasets/data_used_tu_generate_images.csv"
    run_angle_cls(csv_file_path)
