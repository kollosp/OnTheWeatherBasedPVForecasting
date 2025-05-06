import numpy as np
import pandas as pd

from CloudinessFeatures.utils import max_pool, min_pool, window_subtraction
from sktime.forecasting.base import BaseForecaster

class Model(BaseForecaster):
    _tags = {
        "requires-fh-in-fit": False
    }
    def __init__(self, window_size:int=3) -> None:
        super().__init__()
        self.window_size = window_size

    def transform(self, y : pd.Series) -> pd.Series:
        """
        Function calculates VCI feature on y
        :param y:
        :return: pd.Series contains VCI. Has same index as y
        """

        diff = np.array(np.diff(y.to_numpy()).tolist() + [0])

        self.mx_ = max_pool(diff, self.window_size)
        self.mi_ = min_pool(diff, self.window_size)
        s = window_subtraction(self.mx_, self.mi_, self.window_size) / self.window_size

        s[np.isnan(s)] = 0
        s[np.isinf(s)] = 0

        self.value_ = pd.Series(data=s, index=y.index)
        return self.value_

    def _fit(self, y, X=None, fh=None):
        return self

    def _predict_description(self) -> pd.DataFrame:
        return pd.DataFrame({
            f"{str(self)}.mx": self.mx_,
            f"{str(self)}.mi": self.mi_,
            f"{str(self)}": self.value_,
        }, index=self.value_.index)

    def __str__(self) -> str:
        return "VCI"

    def _predict(self, fh=None, X=None) -> pd.Series:
        """
        Function does nothing. It exists only for consistency
        :param fh:
        :param X:
        :return:
        """
        return pd.Series(data=0, index=fh.to_pandas())

