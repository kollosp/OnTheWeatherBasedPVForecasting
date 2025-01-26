from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sklearn.metrics import r2_score
from utils import Solar
from utils.Plotter import Plotter
from matplotlib import pyplot as plt
from datetime import datetime as dt
from typing import List
# package imports
from .Optimized import Optimized

from .Overlay import Overlay
from .Modelv2 import Model as basemodel
from sklearn.metrics import mean_absolute_error
from scipy.optimize import differential_evolution

class Model(basemodel):
    """This class implements model with different prediction method. Not iterative. """
    def __init__(self,
                 latitude_degrees: float = 51,
                 longitude_degrees: float = 14,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 # bandwidth: float = 0.4,
                 window_size: int = None,
                 y_adjustment=False,
                 enable_debug_params: bool = True,

                 # zeros_filter_modifier:float=0, # switched to hyperparametrisation in fit!
                 # density_filter_modifier:float=0,# switched to hyperparametrisation in fit!


                 str_representation_limit = -1,
                 ):

        super().__init__(
            latitude_degrees =latitude_degrees,
            longitude_degrees = longitude_degrees,
            x_bins = x_bins,
            y_bins = y_bins,
            bandwidth = 0.2,
            window_size = window_size,
            enable_debug_params = enable_debug_params,
            zeros_filter_modifier = 0.1,
            density_filter_modifier = .5,
            interpolation = True,
            return_sequences = False,
            str_representation_limit = str_representation_limit,
            y_adjustment = y_adjustment
        )
        self.model_name = "SEAPFv3"

    def _fit(self, y, X=None, fh=None):
        """
        Function performs iterative differential evolution hyperparametrisation to minimize in-sample prediction MAE
        >>> best_params = self.params()
        >>> self.fit(y,X)
        >>> pred = self.in_sample_predict(y,X)
        >>> mae = mae(pred,y)
        >>> self.params += derivative(mae) * alpha

        :param y:
        :param X:
        :param fh:
        :return:
        """
        def f(x):
            self.bandwidth = x[0]
            self.zeros_filter_modifier = x[1]
            self.density_filter_modifier = x[2]
            super(Model, self)._fit(y, X, fh)
            pred = self.in_sample_predict(X, ts=y.index)
            mae = mean_absolute_error(y,pred)
            # print(":",x, mae)

            return mae

        init_mae = f([self.bandwidth, self.zeros_filter_modifier, self.density_filter_modifier])
        bounds=[(0,0.3),(0,1),(0,1)]
        integrality=[0,0,0]
        # 3 *2
        result = differential_evolution(f, bounds=bounds, integrality=integrality, maxiter=10, popsize=0.1)
        # print("results", result.x, result.fun)
        end_mae = f(result.x)

        print("verification", {
            "Init":init_mae,
            "dif ret:": result.fun,
            "in sample:": end_mae,
            "params: ": result.x
        })
        return self