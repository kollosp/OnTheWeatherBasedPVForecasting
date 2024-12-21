import numpy as np
from CloudinessFeatures.utils import max_pool, min_pool, window_subtraction
from sktime.forecasting.base import BaseForecaster

class Model(BaseForecaster):
    _tags = {
        "requires-fh-in-fit": False
    }
    def __init__(self, window_size:int=3) -> None:
        super().__init__()
        self.window_size = window_size

    def transform(self, y):
        self._fit(y=y)
        return self._predict()

    def _fit(self, y, X=None, fh=None):
        self.y_=y.to_list()
        return self

    def _predict(self, fh=None, X=None):
        """Class transformes only the y provieded in fit function"""
        diff = np.array(np.diff(self.y_).tolist() + [0])

        mx = max_pool(diff, self.window_size)
        mi = min_pool(diff, self.window_size)
        s = window_subtraction(mx, mi, self.window_size) / self.window_size
        # s = s * factor

        s[np.isnan(s)] = 0
        s[np.isinf(s)] = 0
        return s

