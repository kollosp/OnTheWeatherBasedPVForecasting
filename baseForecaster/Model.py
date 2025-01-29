import pandas as pd
from sktime.forecasting.base import BaseForecaster

class Model(BaseForecaster):
    _tags = {
        "requires-fh-in-fit": False
    }

    def __init__(self, lazy_fit: pd.DateOffset = None):
        super(Model, self).__init__()
        self.lazy_fit = lazy_fit

    @property
    def y(self):
        """Returns last seen y (last passed to fit method)"""
        return self.y_

    @property
    def last_fitted_y(self):
        """last_fitted_y flag is used to determine if model needs to be refitted"""
        return self.last_fitted_y_

    def batch_fit(self, y,X=None,fh=None):
        """Method to fit model in batch fitting strategy. Function is executed from _fit if refitting is needed"""
        raise NotImplemented("batch_fit method has is abstract method and therefor should be implemented in extension class")

    def _fit(self,y,X=None,fh=None):
        """Function checks if model should be refitted in batch learning mode"""
        self.y_ = y # update last y. Some models like NLastPeriods require recent data

        # if not after first fitting (already fitted) or lazy fit is not defined -> fit for sure
        if self.lazy_fit is None or not hasattr(self, 'last_fitted_y_'):
            self.last_fitted_y_ = y.index[-1]
            return self.batch_fit(y,X,fh)

        # refit if condition met
        if self.lazy_fit + self.last_fitted_y < y.index[-1]:
            self.last_fitted_y_ = y.index[-1]
            return self.batch_fit(y,X,fh)

        # leave model as it is
        return self