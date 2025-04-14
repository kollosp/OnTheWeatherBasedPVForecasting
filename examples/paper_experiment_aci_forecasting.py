import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__": import __config__
import os
from utils.Experimental import Experimental
from WeatherWrapper.ACIWF import Model as ACIWF
import pandas as pd
from matplotlib import pyplot as plt
from utils.ArticleUtils import print_or_save_figures, df_2_latex_str, join_dataframes
from paper_experiment_weather_statistics import compute_aci_aggregated_df, generate_or_load_aci
from RollingAverage.Model import Model as RollingAverage
WEATHER_CLASS = "\wc{}"
DATASET = "\ds{}"
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def ACICLASS_2_TEXT(c):
    # classes = [
    #     "Good weather / sunny",
    #     "Fair weather",
    #     "Bad weather / overcast",
    #     "Snowy weather / no power"]

    classes = [
        "Good w.",
        "Fair w.",
        "Bad w.",
        "Snowy w."]
    return classes[c]

def main_instance(pv_instance, y,X):
    fig, ax = plt.subplots(2)
    # fig.suptitle(f"WC forecasting for PV {pv_instance}")
    ax[0].plot(y.index, y.values)
    ax[1].plot(X.index, X.values)

    ex = Experimental()
    ex.register_dataset(X)
    ex.register_model(NaiveForecaster(strategy="last"))
    ex.register_model(RollingAverage(N=3))
    ex.register_model(RollingAverage(N=30))
    ex.register_model(ExponentialSmoothing())
    # ex.register_model(make_reduction(MLPRegressor(hidden_layer_sizes=(10,10,10))))
    # ex.register_model(make_reduction(MLPRegressor(hidden_layer_sizes=(10,5,3))))
    k = 3
    pred, metrics, forecast_start_point = ex.predict(
        forecast_horizon = 1,
        batch = 1,
        learning_window_length = 360,
        window_size=30,
        # early_stop=30,
        enable_description=False)
    prediction_columns = ex.model_names
    pred = pred[prediction_columns]

    for c in pred.columns:
        decision_boundary = (k-1) / k
        pred["WC_" + c] = pred[c] // decision_boundary
        pred["WC_" + c][pred["WC_" + c] == k] = k-1

        ax[0].plot(y.index, [np.nan] * forecast_start_point + pred["WC_" + c].tolist(), label=c)
        ax[1].plot(X.index, [np.nan] * forecast_start_point + pred[c].tolist())
    _ax = ax[0]
    _ax.legend()
    _ax.set_xticklabels([])
    _ax.set_ylabel("ACI (class)")

    _ax = ax[1]
    _ax.tick_params(axis='x', labelrotation=30)
    _ax.set_ylabel("ACI")

    wc_columns = [c for c in pred.columns if "WC_" in c]
    y = y[forecast_start_point:]

    ret = {}
    for wc_column in wc_columns:
        metrics = []
        metrics.append(accuracy_score(y,pred[wc_column]))
        # metrics.append(balanced_accuracy_score(y,pred[wc_column]))
        ret[wc_column[3:]] = metrics

    df = pd.DataFrame.from_dict(ret, orient='index', columns=["accuracy"])
    df.index.rename("Model", inplace=True)
    df.columns.rename("Metric", inplace=True)
    return [df], [fig]

def main(pv_instances):
    all_dfs = []
    indexes = []
    all_figures = []
    for pv_instance in pv_instances:
        pred, latitude_degrees, longitude_degrees  = generate_or_load_aci(pv_instance=pv_instance, enable_load=True)
        pred = compute_aci_aggregated_df(pred)
        y = pred["ACIWF.decision_final"]
        X = pred["ACIWF.decision"]

        dfs, figs = main_instance(pv_instance, y, X)
        all_figures = all_figures + figs
        all_dfs.append(dfs)
        indexes.append(f"PV {pv_instance}")


    pred = join_dataframes(all_dfs, indexes, [DATASET, WEATHER_CLASS])
    txt = df_2_latex_str(pred[0], caption=f"The table presents the accuracy of various forecasting methods for predicting quantized qACId across datasets (PV 0, PV 1, and PV 2). It demonstrates the effectiveness of alternative prediction strategies compared to the naive forecast baseline, effectively capturing both short-term fluctuations and longer-term trends. The results highlight that RollingAverage with a 3-day window consistently achieved the best or near-best performance across the datasets, confirming that a short-term smoothing approach effectively captures weather-driven variations in qACId.",
                         command_name="WCPrediction", float_format="{:0.3f}".format)

    with open("cm/paper_experiment_aci_forecaster.tex", "w") as f:
        f.write(txt)
    print_or_save_figures(figures=all_figures, path="cm/paper_experiment_aci_forecaster.pdf")


if __name__ == "__main__":
    pv_instances = [0,1,2]
    main(pv_instances)
    plt.show()