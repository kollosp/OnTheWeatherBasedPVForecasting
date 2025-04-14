from typing import Callable

import pandas as pd
from baseForecaster.Model import Model as BaseForecaster
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
    def __init__(self,
                 latitude_degrees: float = 51,
                 longitude_degrees: float = 14,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 k=3, # number of weather classes
                 model_factory = Callable[[], BaseForecaster],
                 **kwargs
                 ):
        """
        Model performs PV timeseries forecasting facilitating Local Weather Properties. Local Weather Properties are
        calculated using CloudinessFeatures: OCI,VCI, ACI. Model can perform batch learning. It is done by checking
        y data passed in fit.
        :param latitude_degrees: PV latitude
        :param longitude_degrees: PV longitude
        :param x_bins: ACI model's x resolution
        :param y_bins: ACI model's y resolution
        :param k: number of ACI weather classes
        :param model_factory: function factory. Creating forecaster instance. One instance of forecaster is created and
                              fit for each ACI weather classes (K)
        :param lazy_fit: batch learning enabler. pd.DateOffset decides how often models should be fitted. If lazy_fit is
                         passed fit method checks condition: last_fitted_y + lazy_fit < y.index[-1]. Model is refitted
                         only if condition is met (the model was fitted on the data older than lazy_fit). e.g.
                         lazy_fit = pd.DateOffset(11), last_fitted_y=10/07/2024 then model is refitted if y.index[-1] >
                         21/07/2024.
        """
        super().__init__(**kwargs)
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.k = k
        self.model_factory = model_factory

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

    def batch_fit(self, y, X=None, fh=None):
        """
        Fit function that is similar to sklearn scheme X contains features while y contains corresponding correct values
        :param X: it should be 2D pandas series [[wclass],[wclass],[wclass],[wclass],...] containing whether classes
        :param y: it should be 1D pandas series [[y1],[y2],[y3],[y4],...] containing observations made
        :return: self
        """
        #
        # if not self._check_if_model_should_be_refitted(y):
        #     return self
        print("batch fit...")
        self.last_fitted_y_ = y.index[-1]
        self.oci_ = OCI(
            window_size=3,
            model_factory = lambda : sktimeSEAPFv2(
                latitude_degrees = self.latitude_degrees,
                longitude_degrees =  self.longitude_degrees,
                x_bins=self.x_bins, y_bins=self.y_bins,
                # bandwidth=0.19597171,
                # zeros_filter_modifier=0.06627492,
                # density_filter_modifier=0.49751398,
                zeros_filter_modifier = 0,
                density_filter_modifier = 0,
                bandwidth = 0.1
            ))


        self.oci_ = self.oci_.fit(y)
        self.vci_ = VCI().fit(y)
        aci = self.compute_weather_features(y)

        # print("Learning: ", len(self.oci_ts_), "len(aci)", len(aci))

        self.weather_classes_ = np.unique(aci)
        self.models_ = []
        for wc in self.weather_classes_:
            wc_y = y[aci == wc]
            m = self.model_factory()
            m.fit(wc_y)
            self.models_.append({
                "wc" : wc,
                "_model": m,
                "len": len(wc_y),
                "max": max(wc_y),
                "min": min(wc_y),
                "mean": sum(wc_y) / len(wc_y),
                "_model.repr": m.model_representation
            })

        print(self.learning_statistics_str())
        return self

    def learning_statistics_str(self):
        s = ""
        for m in self.models_:
            mrp = m['_model.repr']
            s = s + f"Weather class: {m['wc']}\n"
            s = s + f"    len: {m['len']} ~ {100 * m['len'] / len(self.y_):.1f}%\n"
            s = s + f"    max: {m['max']}\n"
            s = s + f"    min: {m['min']}\n"
            s = s + f"    mean: {m['mean']}\n"
            s = s + f"    _model.repr: {max(mrp)}, {sum(mrp)/len(mrp)}, {min(mrp)}\n"
        return s

    def _predict_description(self, y_true=None):
        d = pd.concat([
            self.oci_._predict_description(),
            self.vci_._predict_description(),
            self.aci_._predict_description()], axis=1)
        d.rename(columns={c:f"{str(self)}.{c}" for c in d.columns}, inplace=True)

        model_dict = {
            # f"{str(self)}.x": self.y_,
            # f"{str(self)}": self.prediction_,
            # f"{str(self)}.wc": pd.Series(self.weather_classes_prediction_.values, index=self.y_.index),
            f"{str(self)}.elevation": self.elevation_
        }

        # if y_true is not None:
        #     model_dict["y_true"] = y_true

        d = pd.concat([d, pd.DataFrame(model_dict)], axis=1)
        return d

    def _predict(self, fh, X):
        self.fh_ = fh
        self.X_ = X
        x = self.y #X.iloc[:,0]

        aci = self.compute_weather_features(x)
        elevation = self.aci_.elevation
        self.elevation_ = pd.Series(elevation, index=self.y.index)
        self.elevation_[self.elevation_ < 0] = 0
        aci = aci[-288:] # take last day
        aci = aci[elevation > 0] # take only day data
        counts = np.bincount(aci)
        aci_most_frequent = np.argmax(counts) # most frequent value
        # aci_most_frequent = sum(aci) / len(aci)
        # print("len(aci)", len(aci))
        # print("counts", counts, len(aci), aci_most_frequent)
        self.weather_classes_prediction_ = pd.Series(aci_most_frequent, index=fh.to_pandas())
        self.predictions_ = [np.nan] * len(self.models_)
        for i, m in enumerate(self.models_):
            self.predictions_[i] = m["_model"].predict(fh)

        # self.predictions_[aci_mean] = self.models_["_model"].predict(fh,X)

        self.prediction_ = self.predictions_[round(aci_most_frequent)]
        return self.prediction_

    def get_params_info(self):
        params = {
            "x_bins": (10, 10, 80, True),
            "y_bins": (10, 10, 80, True),
        }

        if self.transformer is not None:
            transformer_params = self.transformer.get_params_info()
            for tp in transformer_params:
                params["transformer__" + tp] = transformer_params[tp]
        return params

    def __str__(self):
        return "WSEAIPPF"

    def plot(self):
        fig, ax = plt.subplots(2)
        fig.suptitle(f"{str(self)}")
        model_reprs = []
        for _, m in enumerate(self.models_):
            m["_model"].plot(prefix=m["wc"])
            mrp = m["_model"].model_representation
            model_reprs.append(mrp)
            ax[0].plot(mrp, label=f"{m['wc']}:{str(m['_model'])}")

        ax[0].legend()
