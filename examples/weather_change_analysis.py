import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from scipy.stats import zscore, wasserstein_distance
from scipy.signal import correlate
from scipy.fft import fft
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tslearn.metrics import cdist_dtw
from fastdtw import fastdtw
from tqdm import tqdm

<<<<<<< HEAD
os.chdir("/home/kszyc/projects/OnTheWeatherBasedPVForecasting/")
=======
# os.chdir("/home/kszyc/projects/OnTheWeatherBasedPVForecasting/")
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325


def compare_signals(csv_file, column, distance_name, distance_fn):
    df = pd.read_csv(csv_file, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()

    series = df[column]
    series = series.rolling(window=12 * 3).mean()  # 3h rolling average

    date_range = pd.date_range(
        series.index.min().date(), series.index.max().date(), freq="D"
    )

    timestamps, distances = [], []
    for i, date in tqdm(enumerate(date_range[:-1]), total=len(date_range) - 1):
        start_time = pd.to_datetime(date)
        end_time = pd.to_datetime(date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)

        daily_data = series[start_time:end_time].to_numpy()
        shifted_daily_data = series[
            start_time + pd.Timedelta(days=1) : end_time + pd.Timedelta(days=1)
        ].to_numpy()

        if len(daily_data) != len(shifted_daily_data):
            continue

        try:
            distance = distance_fn(daily_data, shifted_daily_data)
            timestamps.append(start_time)
            distances.append(distance)

<<<<<<< HEAD
            if random.random() < 0.000:
=======
            if random.random() < 0.00:
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325
                plot_daily_signals(
                    distance_name, daily_data, shifted_daily_data, date, distance
                )
        except Exception as e:
            pass

    distance_threshold = (
        np.percentile(distances, 95) - np.percentile(distances, 5)
    ) * 0.25 + np.percentile(distances, 5)

    _total, _similar = 0, 0
    for distance in distances:
        _total += 1
        _similar += 1 if distance <= distance_threshold else 0

    print(f"{column} {_similar}/{_total}={_similar/_total:.2f} -- for {distance_name} with threshold {distance_threshold:.2f}")

<<<<<<< HEAD
    plot_distance_histogram(distance_name, distance_threshold, distances)
    plot_distances_over_time(distance_name, distance_threshold, timestamps, distances)
=======
    plot_distance_histogram(distance_name, distance_threshold, distances, column)
    plot_distances_over_time(distance_name, distance_threshold, timestamps, distances, column)
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325


def simple_dtw(daily_data, shifted_daily_data):
    distance, _ = fastdtw(
        daily_data, shifted_daily_data, dist=lambda x, y: np.abs(x - y)
    )

    return distance


def dtw_with_normalization(daily_data, shifted_daily_data):
    daily_data_normalized = zscore(daily_data)
    shifted_daily_data_normalized = zscore(shifted_daily_data)
    distance, _ = fastdtw(
        daily_data_normalized,
        shifted_daily_data_normalized,
        dist=lambda x, y: np.abs(x - y),
    )
    return distance

def shape_based_distance(daily_data, shifted_daily_data):
    distance = cdist_dtw([daily_data], [shifted_daily_data])[0, 0]
    return distance


def cosine_similarity_measure(daily_data, shifted_daily_data):
    daily_data_normalized = zscore(daily_data).reshape(1, -1)
    shifted_daily_data_normalized = zscore(shifted_daily_data).reshape(1, -1)
    similarity = cosine_similarity(
        daily_data_normalized, shifted_daily_data_normalized
    )[0, 0]
    return -similarity


def fourier_transform_distance(daily_data, shifted_daily_data):
    fft_daily = np.abs(fft(daily_data))
    fft_shifted_daily = np.abs(fft(shifted_daily_data))
    distance = np.linalg.norm(fft_daily - fft_shifted_daily)
    return distance


def wasserstein_distance_measure(daily_data, shifted_daily_data):
    distance = wasserstein_distance(daily_data, shifted_daily_data)
    return distance


<<<<<<< HEAD
def plot_distance_histogram(distance_name, threshold, distances):
=======
def plot_distance_histogram(distance_name, threshold, distances, column):
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325
    plt.figure(figsize=(12, 6))
    plt.hist(distances, bins=100, color="blue", alpha=0.7, label="Distances")
    plt.axvline(x=threshold, color="r", linestyle="--", label="Threshold")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title(
<<<<<<< HEAD
        f"{distance_name} -- Histogram of Distances Between Daily Data and Shifted Daily Data"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_distances_over_time(distance_name, threshold, timestamps, distances):
=======
        f"{column}, {distance_name} -- Histogram of Distances Between Daily Data and Shifted Daily Data"
    )
    plt.legend()
    plt.grid(True)



def plot_distances_over_time(distance_name, threshold, timestamps, distances, column):
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325
    plt.figure(figsize=(15, 4))

    norm = plt.Normalize(
        vmin=np.percentile(distances, 5), vmax=np.percentile(distances, 95)
    )
    cmap = matplotlib.colormaps["coolwarm"]

    scatter = plt.scatter(
        timestamps, distances, c=distances, cmap=cmap, norm=norm, s=3, label="Distance"
    )

    plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Distance")

    plt.xlabel("Timestamp")
    plt.ylabel("Distance")
    plt.title(
<<<<<<< HEAD
        f"{distance_name} -- Distance Between Daily Data and Shifted Daily Data Over Time"
    )
    plt.legend()
    plt.grid(True)
    plt.show()
=======
        f"{column}, {distance_name} -- Distance Between Daily Data and Shifted Daily Data Over Time"
    )
    plt.legend()
    plt.grid(True)

<<<<<<< HEAD
=======
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325


>>>>>>> d361e4447b35783d04fb5ebd4d2c01fac514d77a
def plot_daily_signals(distance_name, data1, data2, date, distance):
    plt.figure(figsize=(10, 4))
    plt.plot(data1, label="Day 1", color="blue")
    plt.plot(data2, label="Day 2 (Shifted)", color="red")
    plt.title(
        f'{distance_name} -- Comparison for {date.strftime("%Y-%m-%d")} - distance: {distance:.2f}'
    )
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
<<<<<<< HEAD
    plt.show()
=======
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325


if __name__ == "__main__":
    csv_file_path = "./datasets/weather_dataset_0.csv"
    distance_functions = [
        ("Simple DTW", simple_dtw),
<<<<<<< HEAD
        ("DTW with Normalization", dtw_with_normalization),
        ("Shape-Based Distance", shape_based_distance),
        ("Cosine Similarity", cosine_similarity_measure),
        ("Fourier Transform Distance", fourier_transform_distance),
        ("Wasserstein Distance", wasserstein_distance_measure),
=======
        # ("DTW with Normalization", dtw_with_normalization),
        # ("Shape-Based Distance", shape_based_distance),
        # ("Cosine Similarity", cosine_similarity_measure),
        # ("Fourier Transform Distance", fourier_transform_distance),
        # ("Wasserstein Distance", wasserstein_distance_measure),
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325
    ]

    for columnn in ["VCI", "OCI"]:
        for distance_name, distance_fn in distance_functions:
            try:
                compare_signals(csv_file_path, columnn, distance_name, distance_fn)
            except Exception as e:
                print(f"Error for {distance_name}: {e}")
<<<<<<< HEAD

    plt.show()
=======
<<<<<<< HEAD
=======
    plt.show()
>>>>>>> 56a11a252a577356322613cbed6cf636f455e325
>>>>>>> d361e4447b35783d04fb5ebd4d2c01fac514d77a
