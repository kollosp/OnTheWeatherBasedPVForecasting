from __future__ import annotations
if __name__ == "__main__": import __config__
import numpy as np
import pandas as pd
from dimensions.BaseDimension import BaseDimension

class DayProgress(BaseDimension):
    def __init__(self, **kwargs):
        super(DayProgress, self).__init__(dimension_name=kwargs.pop("dimension_name", "DayProgress"), **kwargs)

    def _transform(self, y: pd.DataFrame | pd.Series, X: pd.DataFrame | pd.Series | None = None) -> pd.Series | pd.DataFrame:
        seconds_per_day = 24*60*60
        seconds = pd.to_timedelta(y.index.strftime('%H:%M:%S')).total_seconds()

        return 100 * pd.Series(data=seconds / seconds_per_day, index=y.index)