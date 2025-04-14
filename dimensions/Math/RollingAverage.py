from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension
from dimensions.Math.utils import window_moving_avg

class RollingAverage(BaseDimension):
    def __init__(self, window_size, **kwargs):
        self.window_size = window_size
        super(RollingAverage, self).__init__(dimension_name=kwargs.pop("dimension_name", "RollingAverage"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series:
        x = self.extract_y_X(y,X)
        d = window_moving_avg(x.values.flatten(),window_size=self.window_size, roll=True)
        return pd.Series(data=d, index=y.index)