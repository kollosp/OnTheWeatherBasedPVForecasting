from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
import pandas as pd
from .Model import Model as basemodel

import warnings
from sklearn.metrics import mean_absolute_error
from scipy.optimize import differential_evolution

class Model(basemodel):
    """This class implements model with different prediction method. Not iterative. """
    _tags = {
        "requires-fh-in-fit": False
    }

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        def f(x):
            self.N = x[0]
            super(Model, self)._fit(y, X, fh)
            pred = self.in_sample_predict(X=y.to_frame(), fh=y.index)
            mae = mean_absolute_error(y,pred)
            print(f"    -> Model {str(self)} differential_evolution")
            return mae

        bounds = [(0, 30)]
        integrality = [1]
        result = differential_evolution(f, bounds=bounds, integrality=integrality, maxiter=10, popsize=0.1)
        return self

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