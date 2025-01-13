import os
import sys

#os.chdir("./projects/OnTheWeatherBasedPVForecasting/")
#sys.path.append("./projects/OnTheWeatherBasedPVForecasting/")

import sklearn
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.Solar import elevation
from sktimeSEAPF.Optimized import Optimized


class ACI:
    def __init__(self, k, debug=False):
        self.debug = debug
        self.k = k
        self.df = None

    def run(
        self, timestamp, oci, vci, latitude_degrees=53.687, longitude_degrees=15.172
    ):
        self.df = pd.DataFrame(
            {"timestamp": pd.to_datetime(timestamp), "oci": oci, "vci": vci}
        )

        self.df["elevation"] = self.calculate_elevation(
            latitude_degrees, longitude_degrees
        )
        self.df["aci"] = self.calculate_aci()
        if self.debug:
            self.draw_oci_vci()

        aci_daily_mean = self.calculate_daily_mean_aci(only_day=True)
        qaci_daily_mean = (aci_daily_mean / (180 / self.k)).astype(int)

        if self.debug:
            self.calculate_roll_metrics(qaci_daily_mean)

        return qaci_daily_mean

    def calculate_elevation(self, latitude_degrees, longitude_degrees):
        timestamps = self.df["timestamp"].astype("datetime64[s]").astype("int")
        return (
            elevation(
                Optimized.from_timestamps(timestamps),
                latitude_degrees,
                longitude_degrees,
            )
            * 180
            / np.pi
        )

    def calculate_aci(self):
        oci = self.df["oci"].to_numpy()
        vci = self.df["vci"].to_numpy()

        vci = (vci - np.min(vci)) / (np.max(vci) - np.min(vci))
        oci = 2 * (oci - np.min(oci)) / (np.max(oci) - np.min(oci)) - 1

        angles = np.arctan2(oci, vci)
        angles = np.degrees(angles)

        angles = (90 - angles) % 360
        angles = np.where(angles > 180, 360 - angles, angles)

        return angles

    def calculate_daily_mean_aci(self, only_day=True):
        df = self.df[self.df["elevation"] > 0] if only_day else self.df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        daily_mean_aci = df["aci"].resample("D").mean().to_numpy()

        return daily_mean_aci

    def draw_oci_vci(self):
        oci = self.df["oci"]
        vci = self.df["vci"]
        aci = self.df["aci"]

        cmap = matplotlib.colormaps["hsv"]
        colors = [cmap(1 / self.k * (int(v / (180 / self.k)))) for v in aci]

        plt.scatter(vci, oci, c=colors, edgecolor="black", s=5, alpha=0.5)

        plt.axvline(0, color="gray", linestyle="--")
        plt.axhline(0, color="gray", linestyle="--")

        ranges = [
            (i * (180 // self.k), (i + 1) * (180 // self.k)) for i in range(self.k)
        ]
        for i, (start, end) in enumerate(ranges):
            plt.scatter(
                [], [], color=cmap(1 / self.k * i), label=f"{start}-{end}°", s=20
            )

        plt.xlabel("VCI (x-axis)")
        plt.ylabel("OCI (y-axis)")
        plt.title("OCI vs VCI with Angle Coloring")
        plt.grid(True)
        plt.legend(title="Angle Ranges")
        plt.show()

    def calculate_roll_metrics(self, qaci):

        shifted_qaci = np.roll(qaci, 1)

        qaci = qaci[1:]
        shifted_qaci = shifted_qaci[1:]

        acc = sklearn.metrics.accuracy_score(qaci, shifted_qaci)

        relaxed_correct = np.abs(qaci - shifted_qaci) <= 1
        relaxed_acc = np.sum(relaxed_correct) / len(qaci)

        mae = sklearn.metrics.mean_absolute_error(qaci, shifted_qaci)
        mse = sklearn.metrics.mean_squared_error(qaci, shifted_qaci)
        r2 = sklearn.metrics.r2_score(qaci, shifted_qaci)

        print(
            f"ACC: {acc:.2f} --- REL_ACC: {relaxed_acc:.2f} --- MAE: {mae:.2f} --- MSE: {mse:.2f} --- R2: {r2:.2f}"
        )


if __name__ == "__main__":
    csv_file_path = "./datasets/ex_sklearn.csv"

    df = pd.read_csv(csv_file_path, sep=",")
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    timestamp = df["timestamp"].to_numpy()
    oci = df["overall_cloudiness_index"].to_numpy()
    vci = df["variability_cloudiness_index"].to_numpy()

    qaci_daily_mean = ACI(k=9, debug=True).run(timestamp, oci, vci)
    print(qaci_daily_mean)
