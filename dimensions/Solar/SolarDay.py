from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.Solar.Solar import elevation_df
from dimensions.BaseDimension import BaseDimension

class SolarDay(BaseDimension):
    def __init__(self, latitude_degrees: float, longitude_degrees: float, **kwargs):
        super(SolarDay, self).__init__(dimension_name=kwargs.pop("dimension_name", "SolarDay"), **kwargs)
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        df = elevation_df(y, latitude_degrees = self.latitude_degrees, longitude_degrees = self.longitude_degrees)
        df[df < 0] = 0
        df[df > 0] = 1
        return df