from typing import List, Callable, Iterable, Tuple
if __name__ == "__main__": import __config__

import pandas as pd
import numpy as np

from .SlidingWindowExperimentBase import SlidingWindowExperimentBase

class SlidingWindowExperimentTrajectoryBase(SlidingWindowExperimentBase):
    def __init__(self, **kwargs):
        super(SlidingWindowExperimentTrajectoryBase, self).__init__(swe_model_prefix="traj", **kwargs)

    def call_select_model_dataset(self, train_ds_y : pd.Series, train_ds_x : pd.DataFrame, dims: List[str]) -> Tuple[np.array, np.array]:
        """
        Function limits already prepared learning dataset for each model separately using defined dimensions set (list)
        which is provided for each model in self.register_model. Moreover, function transforms pd.DataFrames into numpy arrays
        that can be consumed by sklearn native models. Method reshapes arrays into trajectory forecasting. it means that
        e.g. if one day needs to be forecasted it it sorecasted at once each observation will be returned in 2D array instead
        of 1D array.
        :param train_ds_y: call_fit_dataset first returned tuple element fitting labels set (Y)
        :param train_ds_x: call_fit_dataset second returned tuple element fitting features set (X)
        :param dims: list of columns in X (aka list of features) used by particlular model
        :return: Tuple of train_ds_y, train_ds_x containing limited number of columns (by model's dimensions)- prepared for current test iteration
        """
        if dims is None: # check if for selected model dimensions are defined. If so then adjust learning set
            train_ds_y_np, train_ds_x_np = train_ds_y.values, train_ds_x.values
        else:
            train_ds_y_np, train_ds_x_np = train_ds_y.values, train_ds_x[dims].values

        return train_ds_y_np.reshape(-1, 1, self.forecasting_horizon_), train_ds_x_np.reshape(-1, 1, len(dims)*self.forecasting_horizon_)