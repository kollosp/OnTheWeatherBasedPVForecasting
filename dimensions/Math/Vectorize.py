from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension
from dimensions.Math.utils import window_moving_avg

class Vectorize(BaseDimension):
    def __init__(self, lagged:int,step_per_lag:int=1, **kwargs):
        """

        :param lagged: number of columns to be generated
        :param step_per_lag: multiplicator. number of samples per lag.  Useful for high resolution datasets
        :param kwargs:
        """
        self.lagged = lagged
        self.step_per_lag = step_per_lag
        super(Vectorize, self).__init__(dimension_name=kwargs.pop("dimension_name", "Vectorize"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series:
        x = self.extract_y_X(y,X)
        df = pd.DataFrame({}, index=y.index)
        for lag in range(1,self.lagged+1):
            tmp = np.zeros(len(y.index))
            tmp[self.step_per_lag*lag:] = x[:-self.step_per_lag*lag].values
            df[f"{x.name}_{lag}"] = tmp

        return df