from typing import List, Callable, Iterable, Tuple
if __name__ == "__main__": import __config__

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from SlidingWindowExperimentBase import SlidingWindowExperimentBase

class SWE(SlidingWindowExperimentBase):
    def __init__(self, **kwargs):
        super(SWE, self).__init__(**kwargs)

        y = pd.DataFrame({
            "Sin": np.sin(np.arange(360 * 10) / (4 * np.pi))
        })

        self.register_metric(mean_absolute_error, "MAE")
        self.register_metric(mean_squared_error, "MSE")
        self.register_dataset(y)
        self.register_model(MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=100_000, random_state=0), "MLP")
        self.register_model(make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR")

        #create extra dimensions

    # method to be overwritten
    def fit_generate_dimensions_and_make_dataset(self,train_ts:pd.DataFrame) -> Tuple[np.array, np.array]:
        return super(SWE, self).fit_generate_dimensions_and_make_dataset(train_ts)

    # method to be overwritten
    def predict_generate_dimensions_and_make_dataset(self, predict_ts:pd.DataFrame, test_df:pd.DataFrame) -> Tuple[np.array, np.array]:
        return super(SWE, self).predict_generate_dimensions_and_make_dataset(predict_ts,test_df)

if __name__ == "__main__":
    swe = SWE()

    # perform one day ahead forecasting using 30 days as input data
    output_ts, metrics_df, learning_window_length = swe(
        learning_window_length = 30*4,
        forecasting_horizon = 10,
        predict_window_length = 30,
        train_pause = 30
    )
    swe.show_results()
    plt.show()