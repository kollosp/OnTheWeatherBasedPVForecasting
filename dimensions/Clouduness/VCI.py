from __future__ import annotations
if __name__ == "__main__": import __config__

import numpy as np
from dimensions.BaseDimension import BaseDimension
import pandas as pd
from dimensions.Math.utils import max_pool, min_pool, window_subtraction

class VCI(BaseDimension):
    def __init__(self, window_size, **kwargs) -> None:
        self.window_size = window_size
        super(VCI, self).__init__(dimension_name=kwargs.pop("dimension_name", "VCI"), **kwargs)

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        return self

    def _transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """Class transformes only the y provieded in fit function"""
        y_np = y.values.flatten()

        diff = np.zeros(y_np.shape)
        diff[:-1] = np.diff(y_np)

        mx = max_pool(diff, self.window_size)
        mi = min_pool(diff, self.window_size)
        s = window_subtraction(mx, mi, self.window_size) / self.window_size
        # s = s * factor

        s[np.isnan(s)] = 0
        s[np.isinf(s)] = 0
        return pd.Series(data=s, index=y.index)

