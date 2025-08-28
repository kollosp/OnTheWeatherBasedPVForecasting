import os

if __name__ == "__main__": import __config__
import pandas as pd
import matplotlib.dates as mdates
import numpy as np
from ANN.model_wrappers import *
from utils.Plotter import Plotter
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import argparse
from problem_analysis_run_experiment import test


def load_dataset():
    file_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2] + [f"..{os.sep}datasets{os.sep}dataset.csv"])
    dataset = pd.read_csv(file_path, low_memory=False)
    # self.full_data = self.full_data[30:]
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.index = dataset['timestamp']
    dataset.drop(columns=["timestamp"], inplace=True)
    dataset = dataset[:2 * 360 * 288].loc["2020-04-18":]
    # print(dataset.columns)
    return dataset[["0_Power"]]

def name_change(name:str):
    return name.replace("_0", "").replace("0_Power", "PV 0").replace("Declination", "D").replace("Elevation", "E").replace("SolarDay", "S")

def draw_ax(axis, d):
    for c in d.columns:
        axis.plot(d.index, d[c], label=name_change(c))
        axis.grid(True)

    axis.xaxis.set_major_locator(mdates.HourLocator(12))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))

    axis.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 24, 2)))
    axis.xaxis.set_minor_formatter(mdates.DateFormatter(""))

def draw_chart(df, top_chart_start="16.06.21", top_chart_end="18.06.21", bottom_chart_start="21.08.21", bottom_chart_end="23.08.21"):

    fig, ax = plt.subplots(2)
    draw_ax(ax[0], df.loc[top_chart_start:top_chart_end])
    draw_ax(ax[1], df.loc[bottom_chart_start:bottom_chart_end])
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("Power [kW]")
    ax[0].legend()
    ax[1].set_ylabel("Power [kW]")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Predict script. It is used to generate forecast of selected model.',
        description='')
    args = parser.parse_args()
    parser.print_help()

    # test model
    models = [
        # (lambda *args: make_pipeline(PolynomialFeatures(16), LinearRegression()), "LR(16)", None, None, ["Declination", "Elevation", "y"]),
        # (lambda *args: make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR(12)", None, None, ["SolarDay%", "qACId(8)"]),
        # (lambda *args: make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR(12)", None, None, ["SolarDay%", "qACId(6)"]),
        # (lambda *args: make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR(12)", None, None, ["SolarDay%", "qACId(4)"]),
        # (lambda *args: make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR(12)", None, None, ["SolarDay%"]),
        # (lambda n, dims,: ModelCNN(input_shape=(n, dims), output_shape=1), "CNN", 10, 28, []),
        # (lambda n, dims,: ModelLSTM(input_shape=(n, dims), output_shape=1), "LSTM", 10, 28, ["qACId(4)", "SolarDay%"]),
        # (lambda n, dims,: ModelLSTM(input_shape=(n, dims), output_shape=1), "LSTM", 10, 28, ["SolarDay%"]),
        # (lambda n, dims,: ModelMLP(input_shape=(n, dims), output_shape=1), "MLP", 10, 28, ["Declination", "Elevation", "y"]),
        (lambda n, dims,: ModelMLP(input_shape=(n, dims), output_shape=1), "MLP", 10, 28, ["SolarDay%", "qACId(8)"]),
        (lambda n, dims,: ModelMLP(input_shape=(n, dims), output_shape=1), "MLP", 10, 28, ["SolarDay%", "qACId(6)"]),
        (lambda n, dims,: ModelMLP(input_shape=(n, dims), output_shape=1), "MLP", 10, 28, ["SolarDay%", "qACId(4)"]),
        (lambda n, dims,: ModelMLP(input_shape=(n, dims), output_shape=1), "MLP", 10, 28, ["SolarDay%"]),
        (lambda n, dims,: ModelMLP(input_shape=(n, dims), output_shape=1), "MLP", 10, 28, ["y"]),

    ]
    forecasting_horizon = 288
    prediction_step_len =  288
    datasets = load_dataset()
    metrics_df = pd.DataFrame()
    for model in models:
        p_df, m_df = test(
            forecasting_horizon=forecasting_horizon,
            prediction_step_len=prediction_step_len,
            n=model[2],
            models=[model[0]], # LR16
            models_names=[model[1]], #LR16
            n_steps=model[3],
            instance=0,
            included_dims=model[4],
        )

        for column in m_df.index.to_list():
            datasets[column] = p_df[column]

        metrics_df = pd.concat([metrics_df, m_df])
        #reference model
        # pr_df, mr_df = test(
        #     forecasting_horizon=forecasting_horizon,
        #     prediction_step_len=prediction_step_len,
        #     n=model[2],
        #     models=[model[0]], # LR16
        #     models_names=[model[1]], #LR16
        #     n_steps=model[3],
        #     instance=0,
        #     excluded_dims=["Elevation", "Declination", "SolarDay%"],
        # )
        #
        # for column in mr_df.index.to_list():
        #     datasets[column] = pr_df[column]
        # metrics_df = pd.concat([metrics_df, mr_df])

    #remove learning part of timeseries
    datasets = datasets.loc[datasets[datasets.columns[-1]].first_valid_index():]

    for df in [datasets, metrics_df]:
        print(df.columns)
        print(df)
        print("========")

    # plotter = Plotter(prediction_df.index, [prediction_df[c] for c in prediction_df.columns if not ":" in c], debug=False)
    line_name_list = [name_change(c) for c in datasets.columns]
    plotter = Plotter(datasets.index, [datasets[c] for c in datasets.columns], line_name_list, debug=False)
    plotter.show()

    # figures = []
    # for model in models:
    #     model_prefix = model[1]
    #     model_columns = ["0_Power"] + [c for c in datasets if model_prefix in c]
    #     fig = draw_chart(datasets[model_columns])
    #     figures.append(fig)

    plt.show()

    print("Script done.")

