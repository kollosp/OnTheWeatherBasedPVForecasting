from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension

class Weekday(BaseDimension):
    def __init__(self, **kwargs):
        super(Weekday, self).__init__(dimension_name=kwargs.pop("dimension_name", "Weekday"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        return pd.Series(data=y.index.dayofweek, index=y.index)