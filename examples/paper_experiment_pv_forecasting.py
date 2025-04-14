import numpy as np

if __name__ == "__main__": import __config__
from utils.Experimental import Experimental
from WeatherWrapper.Model import Model as WeatherWrapper
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from SEAIPPF.Transformers.TransformerTest import TransformerTest
from SEAIPPF.Model import Model as SEAIPPF
from sktimeSEAPF.Modelv2 import Model as sktimeSEAPFv2
from sktimeSEAPF.Modelv3 import Model as sktimeSEAPFv3
from sklearn.neighbors import KNeighborsRegressor
from RollingAverage.Model import Model as NLastPeriods
from WeatherWrapper.CloudinessFeatures.OCI import Model as OCI
import os
from sktime.forecasting.neuralforecast import NeuralForecastLSTM
from utils.ArticleUtils import print_or_save_figures, df_2_latex_str, join_dataframes
from matplotlib import pyplot as plt
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sklearn.neural_network import MLPRegressor
from utils.Plotter import Plotter

from sktime.forecasting.naive import NaiveForecaster

def main_instance(pv_instance):
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/dataset.csv"])
    df = pd.read_csv(file_path, low_memory=False)
    df.rename(columns={'Unnamed: 0':'timestamp'}, inplace=True)
    print(df.columns)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    df.index = df['timestamp']



    df.fillna(0, inplace=True)
    latitude_degrees = df[f"{pv_instance}_Latitude"][0]
    longitude_degrees =  df[f"{pv_instance}_Longitude"][0]

    if pv_instance == 1:
        df = df["2022/10/01":] # remove nans from data

    # ex = Experimental(storage_file=f"cm/paper_experiment_pv_{pv_instance}.csv")
    ex = Experimental()
    time_series = df[f"{pv_instance}_Power"]
    time_series = time_series[time_series.first_valid_index():]
    time_series.index = pd.Index(data=np.arange(len(time_series)))

    ex.register_dataset(time_series)
    ex.register_metric(mean_absolute_error, "MAE")
    ex.register_metric(mean_squared_error, "MSE")
    # ex.register_metric(mean_absolute_percentage_error, "MAPE")
    # ex.register_metric(r2_score, "R2")

    m1 = sktimeSEAPFv3(
        latitude_degrees= latitude_degrees,
        longitude_degrees= longitude_degrees,
        x_bins = 34,
        y_bins = 34,
        y_adjustment=True
    )

    m2 = sktimeSEAPFv2(
        latitude_degrees= latitude_degrees,
        longitude_degrees= longitude_degrees,
        x_bins = 34,
        y_bins = 34,
        y_adjustment=False,
        bandwidth =0.4122019567175368,
        zeros_filter_modifier = 0.1,
        density_filter_modifier = 0.4,
        interpolation = True,
        enable_debug_params = True,
        scale_y = 1
    )

    model = WeatherWrapper(
        lazy_fit = pd.DateOffset(300),
        latitude_degrees= latitude_degrees,
        longitude_degrees= longitude_degrees,
        x_bins = 34,
        y_bins = 34,
        model_factory= lambda: sktimeSEAPFv2(
            latitude_degrees= latitude_degrees,
            longitude_degrees= longitude_degrees,
            x_bins = 34,
            y_bins = 34,
            y_adjustment=False,
            bandwidth =0.4122019567175368,
            zeros_filter_modifier = 0,
            density_filter_modifier = 0.3,
            interpolation = True,
            enable_debug_params = True,
            window_size=12,
            scale_y=1
        ))

    # model = NeuralForecastLSTM(
    #     "A-DEC", max_steps=5
    # )
    # ex.register_model(make_reduction(MLPRegressor(hidden_layer_sizes=(100,100,100), max_iter=5000)))
    ex.register_model(make_reduction(KNeighborsRegressor(3)))

    # ex.register_model(NLastPeriods(N=1))
    # ex.register_model(NLastPeriods(N=3))
    # ex.register_model(model)
    # ex.register_model(model)
    # ex.register_model(NaiveForecaster(sp=288))
    # ex.register_model(model)

    #What with batch ?
    print([str(m) for m in ex.models])
    pred, metrics_results, forecast_start_point = ex.predict_or_load(
        forecast_horizon=288,
        learning_window_length=288*1,
        window_size=288,
        batch=1,
        # early_stop=2000,
        enable_description=True
    )

    # print(metrics_results)
    # # desc = model._predict_description()
    #
    #
    # desc = ex.prediction_descriptions
    # desc = desc[["WSEAIPPF.ACI(3)"]]
    # pred = ex.predictions
    # print(desc.index)
    # print(pred.index)
    # pred = pd.concat([desc, pred], axis=1)
    # ex.models[0].plot()

    debug_columns = ["data", str(model), f"{model}.ACI(3)",f"{model}.wc",f"{model}.OCI.expected_from_model",f"{model}.OCI",f"{model}.VCI"]
    test_columns = ["data", str(model), str(m2), f"{model}.wc"]
    # plotter = ex.plot(include_columns=test_columns)


    plotter = Plotter(pred.index, [*[pred[column] for column in pred.columns]],
                      pred.columns, debug=False)
    plotter.show()
    # print(ex.statistics())
    plt.show()

    return [metrics_results]


if __name__ == "__main__":
    pv_instances = [1]
    all_dfs =[]
    indexes = []
    for pv_instance in pv_instances:
        dfs = main_instance(pv_instance=pv_instance)
        all_dfs.append(dfs)
        indexes.append(f"PV {pv_instance}")

    pred = join_dataframes(all_dfs, indexes, ["\ds{}", "Model"])
    pred[0].rename(index={'WSEAIPPF':'Proposed'}, inplace=True)
    print(pred[0])
    txt = df_2_latex_str(pred[0], caption=f"...",
                         command_name="PVPrediction", float_format="{:0.3f}".format)

    with open("cm/paper_experiment_pv_forecaster.tex", "w") as f:
        f.write(txt)
