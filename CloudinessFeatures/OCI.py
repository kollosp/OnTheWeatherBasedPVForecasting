import numpy as np
import pandas as pd
from typing import Callable
from sktimeSEAPF.Modelv2 import Model
from CloudinessFeatures.utils import window_moving_avg

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

    def expected_from_model(self, ts):
        return self.model_instance_.in_sample_predict(ts=ts, X=None)

    def transform(self, ts):
        model_data = self.expected_from_model(ts)
        ex = window_moving_avg(model_data, window_size=self.window_size, roll=True)
        sub = window_moving_avg(ts, window_size=self.window_size, roll=True) - ex
        return pd.Series(data=sub, index=ts.index)

    def _predict(self, fh, X):
        expected_from_model = self.model_instance_.in_sample_predict(ts=fh)
        ex = window_moving_avg(expected_from_model, window_size=self.window_size, roll=True)
        sub = window_moving_avg(X, window_size=self.window_size, roll=True) - ex
        return pd.Series(data=sub, index=fh)


