from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.Solar.Solar import declination_df
from dimensions.BaseDimension import BaseDimension

class Declination(BaseDimension):
    def __init__(self, **kwargs):
        super(Declination, self).__init__(dimension_name=kwargs.pop("dimension_name", "Declination"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        return declination_df(y)