import os
from typing import Tuple

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

    # show_scatter_plot(df, vci_column, oci_column, human_weather_column)
    show_kde(df, vci_column, oci_column, human_weather_column)
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
                levels=20,
                # thresh=0.75,
                color=matplotlib.colormaps.get_cmap("tab10")(i/len(hue_classes)),
                ax=ax,
                bw=2
            )
        except:
            pass

        ax.set_title(f"Class: {hue_class}(n={len(sub_df)})")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()


class AngleClassifier(BaseEstimator):
    def __init__(self, *, param=1):
        self.param = param

    def polar_angle(self, x,y):
        return np.arctan(y/x)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.centroids_ = [
            X[y == i].mean(axis=0) for i in np.unique(y)
        ]
        self.classes_ = np.unique(y)
        self.centroids_polar_angle_ = [self.polar_angle(x,y) for x,y in self.centroids_]

        temp = np.array([[c, centroid] for c, centroid in zip(self.classes_, self.centroids_polar_angle_)])
        self.centroids_polar_angle_ = temp[temp[:, 1].argsort()]
        self.boundaries_polar_angle_ = self.centroids_polar_angle_
        # self.polar_angle_boundaries = [ for c1,c2 in zip(self.centroids_polar_angle_[1:],self.centroids_polar_angle_[:-1])]
        print("centroids:", self.centroids_, "polar[angle, class]\n", self.centroids_polar_angle_)
        return self

    def predict(self, X):
        polar_angles = [self.polar_angle(x,y) for x,y in X]
        return np.full(shape=X.shape[0], fill_value=self.param)

def plot_mesh(ax, estimator: BaseEstimator, title: str, mesh_shape: Tuple[int, int], df: pd.DataFrame, x_column:str, y_column:str, hue_column:str):
    """Function plots decision boundaries for selected (created already) estimator on a selected axis (ax) """
    x_min, x_max = df[x_column].min(), df[x_column].max()
    y_min, y_max = df[y_column].min(), df[y_column].max()
    X, y = df[[x_column, y_column]].to_numpy(), df[hue_column].to_numpy()
    estimator.fit(X=X, y=y)
    x_space = np.linspace(x_min, x_max, mesh_shape[0])
    y_space = np.linspace(y_min, y_max, mesh_shape[1])
    xx, yy = np.meshgrid(x_space, y_space)
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(mesh_shape)
    ax.pcolormesh(x_space, y_space, Z)
    ax.set_title(title)

def show_kde(df, x_column, y_column, hue_column):
    """Function creates a plot that is consists of several plots:
        - 1D kde for OCI
        - 1D kde for VCI
         and 2D kde for both. """
    fig, ax = plt.subplots(6,2)
    sns.set_theme(style="ticks")
    df = df[[x_column, y_column, hue_column]].dropna()
    hue_classes = sorted(df[hue_column].unique())
    n_classes = len(hue_classes)
    grid_size = int(np.ceil(np.sqrt(n_classes)))

    x_min, x_max = df[x_column].min(), df[x_column].max()
    y_min, y_max = df[y_column].min(), df[y_column].max()

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
        ax=ax[0,0]
    )

    sns.kdeplot(
        data=df,
        y=y_column,
        hue=hue_column,
        fill=True,
        common_norm=True,
        palette="tab10",
        alpha=0.5,
        thresh=0.75,
        linewidth=0,
        ax=ax[1, 1],
    )

    for i, hue_class in enumerate(hue_classes):
        sub_df = df[df[hue_column] == hue_class]
        try:
            sns.kdeplot(
                data=sub_df,
                x=x_column,
                y=y_column,
                fill=True,
                alpha=0.75,
                levels=20,
                # palette="tab10",
                # hue=i/len(hue_classes),
                # thresh=0.75,
                color=matplotlib.colormaps.get_cmap("tab10")(i),
                ax=ax[1,0],
                bw_method=0.4
            )
        except:
            pass
    _ax = ax[1,0]
    _ax.set_xlim([x_min, x_max])
    _ax.set_ylim([y_min, y_max])

    mesh_axs = ax[2:,:].flatten()
    estimators = [
        # (KNeighborsClassifier(1, weights='distance'), "KNN(1)"),
        # (KNeighborsClassifier(4, weights='distance'), "KNN(4)"),
        # (DecisionTreeClassifier(max_depth=100), "DT(100)"),
        (RandomForestClassifier(max_depth=20), "RF(20)"), # the best model that was found during investigation for ICDM'24
        (AngleClassifier(), "A(3)"), # it requires hyper-parametrisation
        (make_pipeline(StandardScaler(), SVC(gamma='auto', C=12)), "SVM(C=12)"), # it requires hyper-parametrisation
        (make_pipeline(StandardScaler(), SVC(gamma='auto', C=1)), "SVM(C=1)"), # it requires hyper-parametrisation
        (make_pipeline(StandardScaler(), SVC(gamma='auto', C=0.1)), "SVM(C=0.1)"), # it requires hyper-parametrisation
        (make_pipeline(StandardScaler(), SVC(gamma='auto', C=0.01)), "SVM(C=0.01)"), # it requires hyper-parametrisation
    ]

    for _ax, estimator in zip(mesh_axs, estimators):
        plot_mesh(_ax, estimator[0], estimator[1], (100,100), df, x_column, y_column, hue_column)

    X, y = df[[x_column, y_column]].to_numpy(), df[hue_column].to_numpy()
    for estimator, estimator_name in estimators:
        scoring = ["balanced_accuracy", "accuracy"]
        scores = cross_validate(estimator, X,y, scoring=scoring, cv=5)
        print(estimator_name)
        for key in scores.keys():
            print(f"    {key}: {scores[key].mean():.2f} +- {scores[key].std():.3f}")


if __name__ == "__main__":
    # csv_file_path = "./datasets/data_used_tu_generate_images.csv"
    csv_file_path = "./datasets/ex_sklearn.csv" # new dataset (different model to generate OCI)
    run_angle_cls(csv_file_path)

    plt.show()
