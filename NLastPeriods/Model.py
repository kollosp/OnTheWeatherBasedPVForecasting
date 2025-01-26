from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster

import warnings


class Model(BaseForecaster):
    """This class implements model with different prediction method. Not iterative. """
    _tags = {
        "requires-fh-in-fit": False
    }

    def __init__(self, N=3):
        super().__init__()
        self.N = int(N)

    def _fit(self, y, X=None, fh=None):
        self.x_time_delta_ = (y.index[-1] - y.index[0]) / len(y)
        self.y_max_ = max(y.values)

    def _predict(self, fh, X):
        # if not any(type(t) == pd._libs.tslibs.timestamps.Timestamp for t in fh):
        #     ts = np.array([i*self.x_time_delta_ + self.cutoff for i in fh]).flatten()
        # else:
        #     ts = fh.to_numpy()
        return self.in_sample_predict(X,fh)

    def in_sample_predict(self, X, fh):
        x = X.iloc[:, 0]
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            prediction = np.zeros(len(fh))
            for i, timestamps in enumerate(fh):
                lower_b = timestamps - self.x_time_delta_
                upper_b = timestamps + self.x_time_delta_
                _y = np.array([x[lower_b - pd.DateOffset(n):upper_b - pd.DateOffset(n)].mean() for n in range(1, self.N+1)])
                prediction[i] = _y.mean()
            prediction[np.isnan(prediction)] = 0
            return pd.Series(data=prediction, index=fh.to_pandas())

    def __str__(self):
        return "NLastDays (" + str(self.get_params()) + ")"

    def plot(self):
        pass