from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from utils.ExecutionTimer import ExecutionTimer

if __name__ == "__main__": import __config__
from dimensions.BaseDimension import BaseDimension
import pandas as pd
from dimensions.Math.utils import window_moving_avg
from dimensions.Clouduness.sktimeMLP import SktimeMLP
from dimensions.Solar.Solar import elevation_df, day_progress_df
from sklearn.neural_network import MLPRegressor

class OCIModel(BaseDimension):
    """Model used by OCI transformer to obtain PV profile"""
    def __init__(self, latitude_degrees:float, longitude_degrees:float, **kwargs) -> None:
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees

        self.model = make_pipeline(PolynomialFeatures(12), LinearRegression())
        # self.model = MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=100)

        super(OCIModel, self).__init__(dimension_name=kwargs.pop("dimension_name", "OCIModel"), **kwargs)

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        df = self.fit_data(y)
        self.model.fit(df.values, y.values.flatten())
        return self

    def fit_data(self, y: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({}, index=y.index)
        # df["elevation"] = elevation_df(y, latitude_degrees=self.latitude_degrees, longitude_degrees=self.longitude_degrees)
        df["SolarDay%"] = day_progress_df(y, latitude_degrees = self.latitude_degrees, longitude_degrees = self.longitude_degrees, solar_time_only=True)
        return df

    def _transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        df = self.fit_data(y)
        model_data = self.model.predict(df.values)

        return pd.Series(data=model_data, index=y.index)

    def fit_transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        df = self.fit_data(y)
        self.model.fit(df.values, y.values.flatten())
        model_data = self.model.predict(df.values)
        return pd.Series(data=model_data, index=y.index)
