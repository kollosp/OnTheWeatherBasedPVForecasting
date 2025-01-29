from math import floor
from typing import Callable

import pandas as pd
from sktime.forecasting.base import BaseForecaster
import numpy as np

from sktimeSEAPF import Optimized
from .CloudinessFeatures.OCI import Model as OCI
from .CloudinessFeatures.VCI import Model as VCI
from .CloudinessFeatures.ACI import ACI as ACI
from matplotlib import pyplot as plt
from utils import Solar
from sktimeSEAPF.Modelv2 import Model as sktimeSEAPFv2
from sktimeSEAPF.Modelv3 import Model as sktimeSEAPFv3

class Model(BaseForecaster):
    _tags = {
        "requires-fh-in-fit": False
    }
    WEATHER_MOST_FREQUENT = 1
    WEATHER_MEAN = 2
    def __init__(self,
                 latitude_degrees: float = 51,
                 longitude_degrees: float = 14,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 k:int=3, # number of weather classes
                 oci_window_size=12,
                 weather_aggregation_method:int=WEATHER_MEAN):
        super().__init__()
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.k = k
        self.oci_window_size = oci_window_size
        self.weather_aggregation_method = weather_aggregation_method


    def compute_weather_features(self, data:pd.Series) -> pd.Series:
        """
        Function computes all weather features
        :param data: timeseries to be evaluated
        :return: returns ACI assignment
        """
        self.aci_ = ACI(k=self.k)
        self.oci_ts_ = self.oci_.transform(data)
        self.vci_ts_ = self.vci_.transform(data)
        try:
            self.aci_ts_ = self.aci_.run(
                timestamp=data.index,
                oci=self.oci_ts_,
                vci=self.vci_ts_,
                latitude_degrees=self.latitude_degrees,
                longitude_degrees=self.longitude_degrees)
        except Exception as e:
            print("Exception handled data:", data.to_numpy().tolist())
            print("Exception handled vci: ", self.vci_ts_.to_numpy().tolist())
            raise e
        return self.aci_ts_

    def _fit(self, y, X=None, fh=None):
        """
        Fit function that is similar to sklearn scheme X contains features while y contains corresponding correct values
        :param X: it should be 2D pandas series [[wclass],[wclass],[wclass],[wclass],...] containing whether classes
        :param y: it should be 1D pandas series [[y1],[y2],[y3],[y4],...] containing observations made
        :return: self
        """
        self.oci_ = OCI(
            window_size=self.oci_window_size,
            model_factory = lambda : sktimeSEAPFv2(
                latitude_degrees = self.latitude_degrees,
                longitude_degrees =  self.longitude_degrees,
                x_bins=self.x_bins, y_bins=self.y_bins,
                # bandwidth=0.19597171,
                # zeros_filter_modifier=0.06627492,
                # density_filter_modifier=0.49751398,
                zeros_filter_modifier = 0,
                scale_y = 0.9,
                y_adjustment=False,
                density_filter_modifier = 0,
                bandwidth = 0.2
            ))

        self.y_ = y
        self.oci_ = self.oci_.fit(y)
        self.vci_ = VCI().fit(y)

        return self

    def _predict_description(self, y_true=None):
        d = pd.concat([
            self.oci_._predict_description(),
            self.vci_._predict_description(),
            self.aci_._predict_description()], axis=1)
        d.rename(columns={c:f"{str(self)}.{c}" for c in d.columns}, inplace=True)

        model_dict = {
            f"{str(self)}.X[0]": self.X_.iloc[:, 0],
            f"{str(self)}.decision": self.decision_,
            f"{str(self)}.decision_final": self.decision_final_,
            f"{str(self)}.elevation": self.elevation_
        }
        d = pd.concat([d, pd.DataFrame(model_dict)], axis=1)
        return d

    def _predict(self, fh, X):
        self.fh_ = fh
        self.X_ = X
        x = X.iloc[:,0]
        fh_index = fh.to_pandas()

        aci : pd.Series = self.compute_weather_features(x)
        elevation = self.aci_.elevation
        self.elevation_ = pd.Series(elevation, index=X.index)
        self.elevation_[self.elevation_ < 0] = 0
        fh_start = fh_index[0]
        fh_start.normalize()
        aci = aci[-288:] # take last day
        aci = aci[elevation > 0] # take only day data

        if self.weather_aggregation_method == Model.WEATHER_MEAN:
            if self.k > 1:
                decision_region = (self.k-1) / self.k
                decision = sum(aci) / len(aci)
                aci_most_frequent = sum(aci) / len(aci) # qACI_d
                aci_most_frequent = aci_most_frequent // decision_region
                if aci_most_frequent == self.k:
                    aci_most_frequent = self.k-1
            else:
                # what to do if class is only one?
                aci_most_frequent = self.k-1
        elif self.weather_aggregation_method == Model.WEATHER_MOST_FREQUENT:
            counts = np.bincount(aci)
            aci_most_frequent = np.argmax(counts) # most frequent value
            decision = aci_most_frequent

        self.decision_ = pd.Series(decision, index=X.index)
        self.decision_final_ = pd.Series(aci_most_frequent, index=X.index)

        self.decision_[elevation <= 0] = np.nan
        self.decision_final_[elevation <= 0] = np.nan
        self.value_ = pd.Series(aci_most_frequent, index=fh_index)
        fh_elevation = Solar.elevation(fh_index, latitude_degrees = self.latitude_degrees, longitude_degrees= self.longitude_degrees)
        self.value_[fh_elevation < 0] = np.nan
        return self.value_

    def get_params_info(self):
        return {}

    def __str__(self):
        return "ACIWF"

    def plot(self):
        fig, ax = plt.subplots(2)
        fig.suptitle(f"{str(self)}")
        model_reprs = []
        for _, m in enumerate(self.models_):
            pass
        ax[0].legend()
