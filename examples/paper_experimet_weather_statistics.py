import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__": import __config__
import os
from utils.Experimental import Experimental
from WeatherWrapper.ACIWF import Model as ACIWF
import pandas as pd
from matplotlib import pyplot as plt
WEATHER_CLASS = "\wc{}"
DATASET = "\ds{}"

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

def df_2_latex_str(df: pd.DataFrame, caption, command_name):
    txt = "\\newcommand*\\" + command_name + "{" + df.to_latex(
        caption=caption,
        label=f"tab:{command_name}",
        float_format="{:0.1f}".format,
        na_rep="-"
    ) + "}"
    txt = txt.replace("\\begin{table}", "\\begin{table}\\centering\\small")
    # return str(df)
    return txt

def plot_weather_classes_vs_energy_per_day(df):
    fig, ax = plt.subplots(2)
    fig.suptitle(f"Weather classes vs. mean power by day")
    _ax = ax[1]
    df = df.groupby(pd.Grouper(freq='d')).agg({
        "data": "sum",
        "ACIWF.decision_final": "mean",
        "ACIWF.decision": "mean"
    })
    df["ACIWF.decision_final"] = df["ACIWF.decision_final"].astype(int)
    df["data"] = df["data"] / 12
    _ax.bar(df.index, df["data"], label="Production by day")
    _ax.set_ylabel("Production [kWh]")
    _ax.set_xlabel("Time")
    _ax.legend()

    bar_labels_clear = [ACICLASS_2_TEXT(k) for k in df["ACIWF.decision_final"]]
    bar_labels = [ACICLASS_2_TEXT(k) for k in df["ACIWF.decision_final"]]
    found = []
    for i,k in enumerate(bar_labels):
        if bar_labels[i] in found:
            bar_labels[i] = "_" + bar_labels[i]
        else:
            found.append(bar_labels[i])

    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    bar_color = [colors[int(k % len(colors))] for k in df["ACIWF.decision_final"]]

    _ax = ax[0]
    _ax.set_ylabel("ACI")
    _ax.bar(df.index, df["ACIWF.decision_final"], label = bar_labels, color=bar_color)
    _ax.legend(title='ACI class label')
    fig.show()
    longest_uninterrupted = {}
    next_classes = {}
    unique = np.unique(df["ACIWF.decision_final"].values)
    for wc in unique:
        # d = df["ACIWF"].values
        d = np.where(df["ACIWF.decision_final"].values == wc, 1, 0)
        k = np.zeros(d.shape)
        for i in range(1, len(d)):
            if not d[i]:
                k[i] = 0
            else:
                k[i] = d[i] + k[i-1]
        diff = np.diff(k)
        lengths = k[:-1][diff < 0]
        nexts = df["ACIWF.decision_final"][1:][diff < 0].values
        next_classes[ACICLASS_2_TEXT(wc)] = 100 * np.bincount(nexts, minlength=len(unique)) / len(nexts)

        lu = [max(lengths), min(lengths), sum(lengths) / len(lengths), np.median(lengths)]
        longest_uninterrupted[ACICLASS_2_TEXT(wc)] = lu

    for i, row in enumerate(next_classes.keys()):
        next_classes[row][i] = np.nan

    stats_nexts = pd.DataFrame.from_dict(next_classes, orient='index',
                                               columns=next_classes.keys())
    stats_nexts.columns = pd.MultiIndex.from_product([["Row Src. Column Dst."], stats_nexts.columns], names=["", WEATHER_CLASS])
    # stats_nexts.index = pd.MultiIndex.from_product([["Src."], stats_nexts.index ], names=["", WEATHER_CLASS])

    stats_longest_uni = pd.DataFrame.from_dict(longest_uninterrupted, orient='index', columns=["max", "min", "mean", "median"])
    stats_longest_uni.index.rename(WEATHER_CLASS, inplace=True)

    df["ACIWF.decision_final"] = bar_labels_clear
    stats = df.groupby("ACIWF.decision_final").agg({
        "data": ["max", "mean", "min", "std", "count"]
    })
    # print(stats.columns)
    stats.index.rename(WEATHER_CLASS, inplace=True)
    stats.columns = stats.columns.droplevel()
    stats["share [\%]"] = 100 * stats["count"] // sum(stats["count"])

    # print("=============================================")

    #
    return stats_nexts, stats_longest_uni, stats

def generate_data(pv_instance):
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/dataset.csv"])
    df = pd.read_csv(file_path, low_memory=False)
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    print(df.columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    df.drop(columns=["timestamp"], inplace=True)
    latitude_degrees = df[f"{pv_instance}_Latitude"][0]
    longitude_degrees = df[f"{pv_instance}_Longitude"][0]
    df = df[[f"{pv_instance}_Power"]].dropna()

    ex = Experimental()
    ex.register_dataset(df[f"{pv_instance}_Power"])

    aciwf = ACIWF(
        latitude_degrees=latitude_degrees,
        longitude_degrees=longitude_degrees,
        x_bins=34,
        y_bins=34,
        k=3)

    ex.register_model(aciwf)
    _predictions, metrics_results, forecast_start_point = ex.predict(
        learning_window_length=288 * 180,
        window_size=288,
        batch=288 * 360,
        # early_stop=288 * 20,
        enable_description=True
    )
    pred = ex.predictions[["data", "ACIWF.decision_final", "ACIWF.decision", "ACIWF.ACI(k=3)"]]

    pred = pred.dropna()
    # plotter = ex.plot(include_columns=["data", "ACIWF", "ACIWF.decision_final", "ACIWF.ACI(3)", "ACIWF.OCI.expected_from_model", "ACIWF.decision"])
    # plotter.show()
    # plt.show()
    return pred

def main(pv_instance=0, persist_file="cm/paper_experiment_aci"):
    file_path = persist_file + f".{pv_instance}.csv"
    if os.path.exists(file_path):
        pred = pd.read_csv(file_path, header=0)
        pred['timestamp'] = pd.to_datetime(pred['timestamp'])
        pred.index = pred['timestamp']
        pred.drop(columns=["timestamp"], inplace=True)
    else:
        pred = generate_data(pv_instance)
        pred.to_csv(file_path)

    return plot_weather_classes_vs_energy_per_day(pred)

def join_dataframes(dfs, indexes=None):
    ret = [pd.DataFrame()] * len(dfs[0])

    for df, indx in zip(dfs, indexes):
        for i, d in enumerate(df):
            d.index = pd.MultiIndex.from_product([[indx], d.index], names=[DATASET, WEATHER_CLASS])
            ret[i] = pd.concat([ret[i],d])

    return ret

if __name__ == "__main__":
    pv_instances = [0,1,2]
    dfs = []
    indexes = []
    for pv_instance in pv_instances:
        ret = main(pv_instance=pv_instance)
        indexes.append(f"PV {pv_instance}")
        dfs.append(ret)

    stats_nexts, stats_longest_uni, stats = join_dataframes(dfs,indexes)

    print(df_2_latex_str(stats_nexts, caption=f"{WEATHER_CLASS} changes structure in percent [\%]", command_name="WeatherChangesStructure"))
    print(df_2_latex_str(stats_longest_uni,  caption=f"Uninterrupted {WEATHER_CLASS} chain length statistics by {WEATHER_CLASS}", command_name="LongestUninterrupted"))
    print(df_2_latex_str(stats, caption=f"Power production statistics by {WEATHER_CLASS} ", command_name="PowerProductionStats"))

    plt.show()
