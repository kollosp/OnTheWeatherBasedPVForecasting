import numpy as np
import pandas as pd
from typing import Callable
from sktimeSEAPF.Modelv2 import Model
from CloudinessFeatures.utils import window_moving_avg
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base import BaseForecaster

class Model(BaseForecaster):
    _tags = {
        "requires-fh-in-fit": False
    }
    def __init__(self, model_factory: Callable[[], Model], window_size:int=3) -> None:
        super().__init__()
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
        self.moving_avg_ = window_moving_avg(y, window_size=self.window_size, roll=True)
        sub = self.moving_avg_ - self.expected_from_model_
        self.value_ = pd.Series(data=sub, index=y.index)
        return self.value_

    def _predict_description(self) -> pd.DataFrame:
        return pd.DataFrame({
            f"{str(self)}.expected_from_model": self.expected_from_model_,
            f"{str(self)}.moving_avg": self.moving_avg_,
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


