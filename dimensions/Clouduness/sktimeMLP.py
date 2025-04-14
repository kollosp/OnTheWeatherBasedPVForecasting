import pandas as pd
import numpy as np
from sktime.forecasting.base import BaseForecaster
from dimensions.Solar.Solar import elevation_df, day_progress_df
from sklearn.neural_network import MLPRegressor

class SktimeMLP(BaseForecaster):
    """
        Helpers class. It wraps SKlearn MLP regressor into sktime-like api. Moreover, it performs data preprocessing
        including: elevation, day% calculation which are used as features for regressor
    """
    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "capability:insample": True,
    }
    def __init__(self,latitude_degrees:float, longitude_degrees:float, fit_interval=None, model=None, fit_counter=None):
        """
        :param latitude_degrees: Datasource location latitude
        :param longitude_degrees: Datasource location longitude
        :param fit_interval: interval defines how often model should be refitted. The fit procedure is performed once
                             per fit_interval iterations. If None: refit is repeated all the times. If 0 then model is
                             fitted only once at the first time.
        """
        super().__init__()
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.fit_interval = fit_interval
        self.fit_counter = fit_counter if fit_counter is not None else 0

        if model is None:
            self.model = MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=100)
            # print("Constructor")
        else:
            self.model = model
            # print("Constructor peristance model")

    def if_fit_needed(self):
        #fit all the time
        if self.fit_interval is None:
            return True
        # fit only once if self.fit_interval
        elif self.fit_interval == 0 and self.fit_counter == 0:
            return True
        # fit if o fit_interval given or the interval elapsed.
        elif self.fit_interval > 0 and self.fit_counter % self.fit_interval == 0:
            return True
        return False

    def _fit(self, y:pd.Series, X=None, fh=None):
        if not self.if_fit_needed():
            self.fit_counter += 1
            return self

        df = pd.DataFrame({}, index=y.index)
        df["Elevation"] = elevation_df(df, self.latitude_degrees, self.longitude_degrees)
        df["Day%"] = day_progress_df(df, self.latitude_degrees, self.longitude_degrees, solar_time_only=True)
        # Y = y.iloc[1:].values
        # X = df[["Day%", "Elevation"]].iloc[:-1].values
        Y = y.values
        X = df[["Day%", "Elevation"]].values
        # X = df[["Elevation"]].values

        self.model.fit(X,Y)
        self.fit_counter += 1

        return self

    def _predict(self, fh=None, X=None):
        return self.in_sample_predict(fh=fh, X=X)

    def in_sample_predict(self, fh=None, X=None, ts=None):
        if fh is not None:
            ts = fh.to_pandas()
        df = pd.DataFrame({}, index=ts)
        df["Elevation"] = elevation_df(df, self.latitude_degrees, self.longitude_degrees)
        df["Day%"] = day_progress_df(df, self.latitude_degrees, self.longitude_degrees, solar_time_only=True)
        # X = df[["Day%", "Elevation"]].iloc[:-1].values
        X = df[["Day%", "Elevation"]].values
        pred = self.model.predict(X)
        pred[df["Elevation"] < 0] = 0

        return pd.Series(data=pred, index=ts)
