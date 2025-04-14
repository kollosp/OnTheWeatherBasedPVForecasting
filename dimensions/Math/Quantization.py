from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension

class Quantization(BaseDimension):
    """
    Class digitize continues values from timeseries into K classes.
        E.g. Quantization([0,1,2,3,4,5,6,7,8,9],k=3) -> [0,0,0,0,1,1,1,2,2,2]
    """
    def __init__(self, k:int, descriptive_params:bool=False, **kwargs) -> None:
        """
        :param k:
        :param kwargs:
        """
        self.k = k
        self.descriptive_params = descriptive_params
        super(Quantization, self).__init__(dimension_name=kwargs.pop("dimension_name", "Quantization"), **kwargs)

    def fit(self, y: pd.DataFrame | pd.Series, X:pd.DataFrame | pd.Series | None = None):
        x = self.extract_y_X(y, X)

        x = x.values
        mi = np.nanmin(x)
        mx = np.nanmax(x)
        r = mx - mi
        class_span = r / self.k
        self.decision_boundaries_ = [mi + i * class_span for i in range(0, self.k + 1)]  # decision boundaries
        print("fit::self.decision_boundaries_", self.decision_boundaries_, mx, mi)
        return self

    def digitize_number(self, y:float):
        for i, (lower, upper) in enumerate(zip(self.decision_boundaries_[0:-1], self.decision_boundaries_[1:])):
            if (lower <= y) & (y <= upper):
                return i
        if y < lower: return 0
        elif y > upper: return self.k

        return 0

    def digitize_array(self, y:np.array):
        ret = np.zeros(y.shape)
        for i, (lower, upper) in enumerate(zip(self.decision_boundaries_[0:-1], self.decision_boundaries_[1:])):
            ret[(lower <= y) & (y <= upper)] = i
        return ret

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series:
        x = self.extract_y_X(y, X)

        ret = np.zeros(x.shape)
        # print("self.decision_boundaries_", self.decision_boundaries_)
        for i, (lower, upper) in enumerate(zip(self.decision_boundaries_[0:-1], self.decision_boundaries_[1:])):
            ret[(x >= lower) & (x <= upper)] = i

        if self.descriptive_params == False:
            return pd.Series(data=ret, index=y.index)
        else:
            df = pd.DataFrame({}, index=y.index)
            df[self.dimension_name] = ret
            for i, db in enumerate(self.decision_boundaries_):
                df[f"DB{i}"] = db
            return df