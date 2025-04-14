from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension

class Weekend(BaseDimension):
    def __init__(self, **kwargs):
        super(Weekend, self).__init__(dimension_name=kwargs.pop("dimension_name", "Weekend"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        month_shift = 0
        dates = y.index
        # if isinstance(dates, pd.Series):
        #     dates = dates.values
        # if isinstance(dates, list):
        #     dates = np.array(dates).astype('datetime64')

        weekend = dates.dayofweek > 4 # 4 = friday
        # season = np.round(season, decimals=0).astype(int) % 4

        return pd.Series(data=weekend, index=y.index)