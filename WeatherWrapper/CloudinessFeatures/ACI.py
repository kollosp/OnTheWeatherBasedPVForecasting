import os
import sys

#os.chdir("./projects/OnTheWeatherBasedPVForecasting/")
#sys.path.append("./projects/OnTheWeatherBasedPVForecasting/")
from typing import Callable, Any

import sklearn
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.Solar import elevation
from sktimeSEAPF.Optimized import Optimized
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base import BaseForecaster

class ACI:
    def __init__(self, k, debug=False):
        self.debug = debug
        self.k = k
        self.df = None

    def run(
        self, timestamp, oci, vci, latitude_degrees=53.687, longitude_degrees=15.172
    ) -> pd.Series:
        self.df = pd.DataFrame(
            {"timestamp": pd.to_datetime(timestamp), "oci": oci, "vci": vci}
        )

        self.df["elevation"] = self.calculate_elevation(
            latitude_degrees, longitude_degrees
        )

        self.df["aci"] = self.calculate_aci()
        if self.debug:
            self.draw_oci_vci()
        # print("=" * 40)
        # print("oci", oci)
        # print("vci", vci)
        # print("aci", self.df["aci"].tolist())
        # aci_daily_mean = self.calculate_daily_mean_aci(only_day=True)
        #if aci == 0 then class == 0
        #if aci == 180 then class == k
        qaci_daily_mean = (self.df["aci"]  / (180 / self.k)).astype(int)

        if self.debug:
            self.calculate_roll_metrics(qaci_daily_mean)

        self.value_ = pd.Series(qaci_daily_mean, index=self.df.index)
        return self.value_

    @property
    def elevation(self):
        return self.df["elevation"]

    def _predict_description(self):
        """
        Function needed by forecasting model
        :return:
        """
        return pd.DataFrame({
            f"{str(self)}": self.value_
        }, index = self.value_.index)

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


        # for what is that normalization?
        # vci_divider = np.max(vci) - np.min(vci)
        #
        #
        # oci_divider = np.max(oci) - np.min(oci)
        # if oci_divider:
        #     oci = 2 * (oci - np.min(oci)) / (oci_divider) - 1
        #
        # #if oci or vci contains only zeros than those conditions dont execute
        # if vci_divider != 0:
        #     vci = (vci - np.min(vci)) / vci_divider
        # else:
        #     vci[:] = 0

        angles = np.arctan2(oci, vci)
        angles = np.degrees(angles)

        angles = (90 - angles) % 360
        angles = np.where(angles > 180, 360 - angles, angles)
        #
        # if all([v == 0 for v in vci]):
        #     print("aci\n", angles.tolist())

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

    def __str__(self):
        return f"ACI({self.k})"



class Model(BaseForecaster):
    _tags = {
        "requires-fh-in-fit": False
    }
    def __init__(self, k, model_factory: Callable[[], Any], window_size:int=3) -> None:
        super().__init__()
        self.k = k
        self.window_size = window_size
        self.window_size = window_size

        # lazy initialization of the model instance
        # self.model_lazy_init: bool = True if model is None else False
        self.model_factory = model_factory


    def _fit(self, y, X=None, fh=None):
        self.model_instance_ = self.model_factory()
        self.model_instance_.fit(y)
        return self

    def transform(self, y : pd.Series) -> pd.Series:
        """
        Function calculates OCI feature on y series.
        :param y:
        :return: oci series. Has same index as y
        """
        in_sample_fh = ForecastingHorizon(y.index, is_relative=False)
        self.expected_from_model_ = self.model_instance_.in_sample_predict(fh=in_sample_fh, X=y.to_frame())
        self.expected_from_model_ = window_moving_avg(self.expected_from_model_, window_size=self.window_size, roll=True)
        sub = window_moving_avg(y, window_size=self.window_size, roll=True) - self.expected_from_model_
        self.value_ = pd.Series(data=sub, index=y.index)
        return self.value_

    def _predict_description(self) -> pd.DataFrame:
        return pd.DataFrame({
            f"{str(self)}.expected_from_model": self.expected_from_model_,
            f"{str(self)}": self.value_,
        }, index=self.value_.index)

    def __str__(self):
        return "OCI"

    def _predict(self, fh, X=None) -> pd.Series:
        """
        Function returns model_instance_ prediction on given fh
        :param fh: forecasting horizon
        :param X:
        :return: oci series. Has same index as y
        """
        #x = X.iloc[:,0] # get first column)
        self.expected_from_model_ = self.model_instance_.in_sample_predict(fh=fh, X=X)
        self.expected_from_model_ = window_moving_avg(self.expected_from_model_, window_size=self.window_size, roll=True)
        # sub = window_moving_avg(, window_size=self.window_size, roll=True) - ex
        return pd.Series(data=self.expected_from_model_, index=fh.to_pandas())





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
