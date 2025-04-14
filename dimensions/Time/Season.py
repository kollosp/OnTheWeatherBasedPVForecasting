from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension

class Season(BaseDimension):
    def __init__(self, **kwargs):
        super(Season, self).__init__(dimension_name=kwargs.pop("dimension_name", "Season"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """
        Seasons winter = 0, spring = 1, summer = 2, autumn = 3
        {0: "winter", 1:"spring", 2:"summer", 3:autumn}
        winter = Dec,Jan,Feb
        spring = Mar,Apr,May
        :param y:
        :param X:
        :return:
        """
        month_shift = 2
        dates = y.index.values
        if isinstance(dates, pd.Series):
            dates = dates.values
        if isinstance(dates, list):
            dates = np.array(dates).astype('datetime64')

        season = (dates.astype('datetime64[M]').astype(int) - month_shift) % 12 / 3
        season = np.round(season, decimals=0).astype(int) % 4

        return pd.Series(data=season, index=y.index)