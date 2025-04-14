from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension

class ChainDimension(BaseDimension):
    """
        Class perfomrs transformer chain operation. Apply each transformer from trannsformers array on the provided
        timeserie y. Return pd.DataFrame cointains y, and one timeserie for each transformer
        >>> ts = pd.Series(...)
        >>> transformer = BaseTransformer()
        >>> feature1 = transformer.fit(ts).predict(ts)
        >>> len(feature1) == len(ts)
        True
        >>> feature1.index == ts.index
        True

    """
    def __init__(self, transformers:List[BaseDimension], **kwargs):
        super(ChainDimension, self).__init__(dimension_name=kwargs.get("dimension_name", "ChainTransformer"))
        self.transformers = transformers

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        for transformer in self.transformers:
            transformer.fit(y)
        return self

    def transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        if X is None:
            X = pd.DataFrame({}, index=y.index)
            X["y"] = y
        else:
            X[str(self)] = y
        for transformer in self.transformers:
            r = transformer.transform(y, X)
            if isinstance(r, pd.Series):
                X[str(transformer)] = r
            elif isinstance(r,pd.DataFrame):
                for c in r.columns:
                    X[c] = r[c]
        return X

    def fit_transform(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        if X is None:
            X = pd.DataFrame({}, index=y.index)
            X["y"] = y
        else:
            X[str(self)] = y
        for transformer in self.transformers:
            r = transformer.fit_transform(y, X)
            if isinstance(r, pd.Series):
                X[str(transformer)] = r
            elif isinstance(r,pd.DataFrame):
                for c in r.columns:
                    X[c] = r[c]
        return X