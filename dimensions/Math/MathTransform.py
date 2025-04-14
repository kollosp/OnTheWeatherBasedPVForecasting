from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension

class MathTransform(BaseDimension):
    """
    Class digitize continues values from timeseries into K classes.
        E.g. Digitize([0,1,2,3,4,5,6,7,8,9],k=3) -> [0,0,0,0,1,1,1,2,2,2]
    """
    def __init__(self, transition:float=0, scale:float=1, **kwargs) -> None:
        """
        :param k:
        :param kwargs:
        """
        self.transition = transition
        self.scale = scale
        super(MathTransform, self).__init__(dimension_name=kwargs.pop("dimension_name", "MathTransform"), **kwargs)

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        return self

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series:
        x = self.extract_y_X(y, X)
        return  (x + self.transition) * self.scale
