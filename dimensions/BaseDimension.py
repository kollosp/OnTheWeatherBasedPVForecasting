from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

class BaseDimension:
    """
        Class designed for timeseries processing especially for feature extraction / generation
        >>> ts = pd.Series(...)
        >>> transformer = BaseDimension()
        >>> feature1 = transformer.fit(ts).predict(ts)
        >>> len(feature1) == len(ts)
        True
        >>> feature1.index == ts.index
        True

    """
    def __init__(self, dimension_name, required_dimensions:int=1, base_dimensions:List[str]=None, scale:float=1, transition:float=0):
        """
        :param required_dimensions: parameter is used to pass information how many items shoudl be passed in base_dimensions
                                    if base_dimensions is not None. If other number is passed Exception will be raised
        :param dimension_name:
        :param base_columns: if transformer needs to generate other transformation it can use already prepared.
                            function extract_y_X can be used to extract from X and y parameters described in base_columns
                            if those parameters are provided function returns dataframe limited to selected base_columns,
                            otherwise None will be returned
        :param scale: it multiplyes output data y = (output + transition) * scale
        :param transition: it is added to output data y = (output + transition) * scale
        """
        self.transition = transition
        self.scale = scale
        self.dimension_name = dimension_name
        self.required_dimensions = required_dimensions
        self._base_dimensions = base_dimensions

    @property
    def base_dimensions(self):
        return self._base_dimensions

    def extract_y_X(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """
            Function checks if 1) base_dimensions were passed 2) if all passed base_dimensions are in X. If both are met
            then the function returns either series or dataframe depends on dimension count.
        """
        if self.base_dimensions is None or not isinstance(self.base_dimensions, list) :
            return y
        if len(self.base_dimensions) != self.required_dimensions:
            raise RuntimeError(
                f"Transformer {str(self)} requires {self.required_dimensions} dimensions, however only {len(self.base_dimensions)} were specified. Dimensions: {self.base_dimensions} ")
        elif isinstance(X, pd.DataFrame) and all(c in X.columns for c in self.base_dimensions):
            if len(self.base_dimensions) == 1:
                return X[self.base_dimensions[0]] # series
            return X[self.base_dimensions] # dataframe
        else:
            raise RuntimeError(f"Transformer {str(self)} is based on dimensions '{self.base_dimensions}', however at least one of them not found in X.")

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        """"""
        return self


    def _transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """
        _transform is a default function that should be overwritten in inheriting objects
        :param y:
        :param X:
        :return:
        """
        return pd.Series(data=y.values, index=y.values)

    def transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        return self.scale_transit(self._transform(y,X))

    def fit_transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        return self.fit(y,X).transform(y,X)

    def __str__(self):
        return self.dimension_name

    def scale_transit(self,y:pd.Series | pd.DataFrame | np.array):
        return (y + self.transition) * self.scale