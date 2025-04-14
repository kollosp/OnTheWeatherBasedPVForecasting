from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from utils.ExecutionTimer import ExecutionTimer

if __name__ == "__main__": import __config__
from dimensions.BaseDimension import BaseDimension
import pandas as pd
from dimensions.Math.utils import window_moving_avg
from dimensions.Solar.Solar import elevation_df, day_progress_df
from dimensions.Clouduness.OCIModel import OCIModel

class OCI(BaseDimension):
    def __init__(self, window_size:int, latitude_degrees:float, longitude_degrees:float, **kwargs) -> None:
        """
        :param window_size:
        :param latitude_degrees:
        :param longitude_degrees:
        :param base_column: name of column in X which contains already calculated OCI model data. X is argument of: fit, transform, fit_transform
        :param kwargs:
        """
        self.window_size = window_size
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees

        self.oci_model = OCIModel(latitude_degrees=latitude_degrees,longitude_degrees=longitude_degrees)
        super(OCI, self).__init__(dimension_name=kwargs.pop("dimension_name", "OCI"), **kwargs)

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        if self.base_dimensions is None:
            self.oci_model.fit(y)

        return self

    def _transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        y_np = y.values.flatten()

        if self.base_dimensions is None:
            # condition is same as inside extract_y_X. In that case extract_y_X returns y, no precalculated data available
            model_data = self.extract_y_X(y, X)
        else:
            model_data = self.oci_model.transform(y)

        ex = window_moving_avg(model_data, window_size=self.window_size, roll=True)

        sub = window_moving_avg(y_np, window_size=self.window_size, roll=True) - ex
        return pd.Series(data=sub, index=y.index)

    def fit_transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        if self.base_dimensions is None:
            # condition is same as inside extract_y_X. In that case extract_y_X returns y, no precalculated data available
            model_data = self.extract_y_X(y, X)
        else:
            model_data = self.oci_model.fit_transform(y)

        # print(f"OCI fit_transform time {tm.seconds_delta}s, data len: {len(y)}")

        y_np = y.values.flatten()
        ex = window_moving_avg(model_data, window_size=self.window_size, roll=True)
        sub = window_moving_avg(y_np, window_size=self.window_size, roll=True) - ex
        return pd.Series(data=sub, index=y.index)
