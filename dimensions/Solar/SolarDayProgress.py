from __future__ import annotations

from dimensions.Solar import Solar

if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.Solar.Solar import day_progress_df
from dimensions.BaseDimension import BaseDimension

class SolarDayProgress(BaseDimension):
    def __init__(self, latitude_degrees: float, longitude_degrees: float, **kwargs):
        super(SolarDayProgress, self).__init__(dimension_name=kwargs.pop("dimension_name", "SolarDay%"), **kwargs)
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        return day_progress_df(y, latitude_degrees = self.latitude_degrees, longitude_degrees = self.longitude_degrees, solar_time_only=True)
