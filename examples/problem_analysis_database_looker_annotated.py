if __name__ == "__main__": import __config__
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt

from utils.Plotter import Plotter
from utils import Solar

from sktimeSEAPF.Optimized import Optimized

def print_full(x):
    pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 2000)
    # pd.set_option('display.float_format', '{:20,.2f}'.format)
    # pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

if __name__ == "__main__":
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/pv_weather_human_and_auto.csv"])
    df = pd.read_csv(file_path, low_memory=False, sep=";")
    print(df.columns)
    # self.full_data = self.full_data[30:]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.index = df['Timestamp']
    df.drop(columns=["Timestamp"], inplace=True)
    df.drop(columns=[f"HumanWeather.{i}" for i in range(1,8)], inplace=True)

    print(df)
    print(df.columns)
    #
    # # print_full(df[["0_Power"]].head(288))
    # latitude_degrees = df["0_Latitude"][0]
    # longitude_degrees = df["0_Longitude"][0]
    # # print_full(df.index.astype('datetime64[s]'))
    # timestamps = df.index.astype('datetime64[s]').astype('int')
    #
    # elevation = Solar.elevation(Optimized.from_timestamps(timestamps), latitude_degrees,
    #                             longitude_degrees) * 180 / np.pi
    # d = pd.DataFrame({
    #     "power": df["0_Power"],
    #     "elevation": elevation
    # }, index = df.index)
    # # print_full(df[["0_Power"]][288:].head(288))
    # # print("First 288 rows")
    #
    plotter = Plotter(df.index, [df[i] for i in ['Production', 'Elevation']], debug=False)
    plotter.show()
    plt.show()


