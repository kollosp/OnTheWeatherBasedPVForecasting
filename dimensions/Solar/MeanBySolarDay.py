from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.Solar.Solar import elevation_df
from dimensions.BaseDimension import BaseDimension

class MeanBySolarDay(BaseDimension):
    """
    Class calculates timeseries mean value over a solar day. For nights it returns 0
    """
    def __init__(self,latitude_degrees:float,longitude_degrees:float, **kwargs) -> None:
        self.latitude_degrees=latitude_degrees
        self.longitude_degrees=longitude_degrees
        super(MeanBySolarDay, self).__init__(dimension_name=kwargs.pop("dimension_name", "MeanBySolarDay"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        ret = pd.Series(data=0.0, index=y.index)
        df = pd.DataFrame({}, index=y.index)
        df["day"] = df.index.floor('d')
        df["Elevation"] = elevation_df(df, self.latitude_degrees, self.longitude_degrees)
        unique_days = df["day"].unique()

        x = self.extract_y_X(y, X)

        for ud in unique_days:

            calendar_day = df["day"] == ud
            solar_day = calendar_day & (df["Elevation"] > 0)
            m = x.loc[solar_day].mean()
            ret.loc[calendar_day] = m

        return ret
