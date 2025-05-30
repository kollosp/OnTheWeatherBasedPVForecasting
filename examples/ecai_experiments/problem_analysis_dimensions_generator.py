from typing import Tuple
if __name__ == "__main__": import __config__

import numpy as np
from dimensions import ChainDimension
from dimensions import Elevation
from dimensions import DayProgress
from dimensions import OCI, VCI, ACI
from dimensions import OCIModel
from dimensions import SolarDayProgress
from dimensions import Quantization
from dimensions import MeanBySolarDay
from dimensions import MathTransform
from dimensions import RollingAverage
from dimensions import Vectorize
from dimensions import Season

from itertools import product

import pandas as pd
import os

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error, max_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,LSTM
from tensorflow.keras import Input

# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.utils import np_utils


from SlidingWindowExperiment import SlidingWindowExperimentBase
from SlidingWindowExperiment import SlidingWindowExperimentTrajectoryBase

class SWE(SlidingWindowExperimentBase):
    def __init__(self, k=4, **kwargs):
        super(SWE, self).__init__(**kwargs)
        latitude_degrees = kwargs.get("latitude_degrees")
        longitude_degrees = kwargs.get("longitude_degrees")
        #create extra dimension generators
        self.chain = ChainDimension(transformers=[
            DayProgress(dimension_name="Day%"),
            Season(dimension_name="Season"),
            Elevation(
                dimension_name="Elevation",
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            Vectorize(lagged=10, step_per_lag=10, base_dimensions=["y"]),
            SolarDayProgress(
                scale=0.01,
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            VCI(
                window_size=12,
                dimension_name="VCI"),
            OCIModel(
                dimension_name="OCIModel",
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            OCI(
                window_size=12,
                dimension_name="OCI",
                base_dimensions=["OCIModel"],
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            ACI(window_size=12,
                dimension_name="ACI",
                base_dimensions=["OCI", "VCI"],
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            MeanBySolarDay(dimension_name="ACId", base_dimensions=["ACI"],
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            Quantization(k=k, dimension_name=f"qACId({k})", base_dimensions=["ACId"]),
            MeanBySolarDay(dimension_name="OCId", base_dimensions=["OCI"],
                latitude_degrees=latitude_degrees,
                longitude_degrees=longitude_degrees),
            Quantization(k=k, dimension_name=f"qOCId({k})", base_dimensions=["OCId"])
        ])

    # method to be overwritten
    def fit_generate_dimensions_and_make_dataset(self,train_ts:pd.DataFrame, fh:int, predict_window_length:int) -> Tuple[pd.Series, pd.DataFrame]:
        train_ts_ex = self.chain.fit_transform(train_ts)
        #[fh:] and [:-fh] is needed to avoid look ahead. However, it is needed only in fit_(...)_dataset beacuse this
        # function gets train_ts and transforms it into features and labels. predict_(...)_dataset gets predict_ts(X) and
        # test_df (Y) separately. SWE ensres that those two sets do not overlap in time domain
        train_ds_y = train_ts_ex.iloc[fh:]["y"] # y contains ground truth
        train_ds_x = train_ts_ex.iloc[:-fh] # it contains features generated by chain transform and y observations
        return train_ds_y, train_ds_x

    # method to be overwritten
    def predict_generate_dimensions_and_make_dataset(self, predict_ts:pd.DataFrame, test_ts:pd.DataFrame, fh:int) -> Tuple[pd.Series, pd.DataFrame]:
        #print("predict_ts.shape", predict_ts.shape, "self.predict_window_length_", self.predict_window_length_)
        test_ts_ex = self.chain.transform(predict_ts)

        test_ds_y = test_ts # test_df is a series containing ground truth
        test_ds_x = test_ts_ex # it contains features generated by chain transform and y observations

        return test_ds_y, test_ds_x

def create_dense_keras_model(input_shape, output_shape, hidded_layers=(20,20,20), optimizer='adam',
                 kernel_initializer='glorot_uniform'):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    for hidded_layer in hidded_layers:
        model.add(Dense(hidded_layer,activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dense(output_shape,kernel_initializer=kernel_initializer))
    model.compile(loss='mean_absolute_error',optimizer=optimizer, metrics=['mean_squared_error'], steps_per_execution=10)
    return model

def create_lstm_keras_model(input_shape, output_shape, hidded_layers=(20,), optimizer='adam',
                 kernel_initializer='glorot_uniform'):
    model = Sequential()
    model.add(Input(shape=input_shape))
    for hidded_layer in hidded_layers:
        model.add(LSTM(hidded_layer,return_sequences=True, kernel_initializer=kernel_initializer))
    model.add(Flatten())
    model.add(Dense(output_shape,kernel_initializer=kernel_initializer))
    model.compile(loss='mean_absolute_error',optimizer=optimizer, metrics=['mean_squared_error'], steps_per_execution=10)
    return model

if __name__ == "__main__":
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["../datasets/dataset.csv"])
    dataset = pd.read_csv(file_path, low_memory=False)
    # self.full_data = self.full_data[30:]
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.index = dataset['timestamp']
    dataset.drop(columns=["timestamp"], inplace=True)
    dataset = dataset[:2*360*288].loc["2020-04-18":]

    print(dataset.columns)

    ks=[2,3,4,5,6,7,8]
    # ks=[7,8]*2
    selectors = [
        #*[[f"qOCId({k})"] for k in ks],
        *[[f"qACId({k})"] for k in ks]]

    configs = list(zip(ks, selectors))
    configs.insert(-1, (1,[]))
    print(configs)

    instance = 0
    metrics_dfs = []
    for i, config in enumerate(configs[:1]):
        k=config[0]
        selector=config[1]
        latitude_degrees = dataset[f"{instance}_Latitude"][0]
        longitude_degrees = dataset[f"{instance}_Longitude"][0]

        df = pd.DataFrame({}, index=dataset.index)
        df["power"] = dataset[f"{instance}_Power"]
        # df["power"] = RollingAverage(window_size=6).fit_transform(df["power"])

        swe = SWE(k=k, latitude_degrees=latitude_degrees, longitude_degrees=longitude_degrees)
        swe.register_dataset(df)
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["SolarDay%", "qACId"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["SolarDay%", "qOCId"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["y", "SolarDay%", "qACId"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["y", "SolarDay%", "qOCId"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["y", "SolarDay%"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["y"])
        # swe.register_model(create_dense_keras_model(input_shape=(10,2), output_shape=1), "tf::MLP", ["SolarDay%", "qACId"], n=10, n_step=2)
        # swe.register_model(create_lstm_keras_model(hidded_layers=(20,20), input_shape=(10,2), output_shape=1), "tf::LSTM(20,20)", ["y", "qACId"], n=10, n_step=28)
        # swe.register_model(create_dense_keras_model(input_shape=11, output_shape=1), "tf::MLP", [f"y_{i}" for i in range(1,11)] + ["qACId"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 10), max_iter=500_000, random_state=0), "MLP", ["y", "SolarDay%", "qACId"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 10), max_iter=500_000, random_state=0), "MLP", ["y", "SolarDay%"])
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 10), max_iter=500_000, random_state=0), "MLP", ["y"])

        # if  1 == 0:
        # swe.register_model(MLPRegressor(hidden_layer_sizes=(2,2,2), max_iter=10, random_state=0), "MLP", ["y", "SolarDay%"]+ selector)
        swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["y", "SolarDay%", "Season"]+ selector)
        swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["SolarDay%"]+ selector)
        swe.register_model(MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=500, random_state=0), "MLP", ["y"] + selector)

        swe.register_model(make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR(12)", ["SolarDay%"] + selector)
        swe.register_model(make_pipeline(PolynomialFeatures(6), LinearRegression()), "LR(6)", ["SolarDay%"] + selector)

        swe.register_model(DecisionTreeRegressor(), "DT", ["SolarDay%"] + selector)
        # swe.register_model(DecisionTreeRegressor(), "DT", ["y", "SolarDay%"] + selector)
        # swe.register_model(DecisionTreeRegressor(), "DT", ["y"] + selector)
        swe.register_model(RandomForestRegressor(), "RF", ["SolarDay%"] + selector)
        # swe.register_model(RandomForestRegressor(), "RF", ["y", "SolarDay%"] + selector)
        # swe.register_model(RandomForestRegressor(), "RF", ["y"] + selector)

        #
        dims = ["y"] + selector
        swe.register_model(create_lstm_keras_model(hidded_layers=(10, 10), input_shape=(10, len(dims)), output_shape=1),"tf::LSTM(10,10)", dims, n=10, n_step=2)
        # swe.register_model(create_lstm_keras_model(hidded_layers=(10, 10), input_shape=(10, len(dims)), output_shape=1),"tf::LSTM(10,10)", dims, n=10, n_step=28)
        # swe.register_model(create_lstm_keras_model(hidded_layers=(20, 10), input_shape=(20, len(dims)), output_shape=1),"tf::LSTM(20,10)", dims, n=20, n_step=28)


        predictions_df = pd.DataFrame()
        # metrics_df = pd.DataFrame()

        swe.register_metric(mean_absolute_error, "MAE")
        swe.register_metric(mean_squared_error, "MSE")
        swe.register_metric(mean_absolute_percentage_error, "MAPE")
        swe.register_metric(max_error, "ME")

        swe.summary()
        _,metrics_df,_ = swe()
        metrics_df["i"] = i

        plotter = swe.show_results()
        metrics_dfs.append(metrics_df)
        plotter2 = swe.show_fit_dimensions()
        # print(metrics_df)

        concat_df = pd.concat(metrics_dfs)
        concat_df.to_csv(f"cm/concat_{i}.csv")
        print(concat_df)

    print("Final")
    print(pd.concat(metrics_dfs))

    plt.show()
