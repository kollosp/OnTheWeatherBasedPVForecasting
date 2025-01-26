import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__": import __config__
import os
from utils.Experimental import Experimental
from WeatherWrapper.ACIWF import Model as ACIWF
import pandas as pd
from matplotlib import pyplot as plt
from utils.ArticleUtils import print_or_save_figures, df_2_latex_str, join_dataframes
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




def compute_aci_aggregated_df(df):
    df = df.groupby(pd.Grouper(freq='d')).agg({
        "data": "sum",
        "ACIWF.decision_final": "mean",
        "ACIWF.decision": "mean"
    })

    # print(df["ACIWF.decision_final"]["2021-05-26":"2021-06-03"].tolist())
    df["ACIWF.decision_final"] = df["ACIWF.decision_final"].round(0).astype(int)
    df["data"] = df["data"] / 12
    return df

def plot_weather_classes_vs_energy_per_day(df, pv_instance):

    fig, ax = plt.subplots(3)
    fig.suptitle(f"Weather classes and mean daily power by day for PV {pv_instance}")

    df = compute_aci_aggregated_df(df)
    _ax = ax[-1]
    _ax.tick_params(axis='x', labelrotation=30)

    _ax.bar(df.index, df["data"], label="Production by day")
    _ax.set_ylabel("Production [kWh]")
    _ax.set_xlabel("Time")
    _ax.legend()
    unique = np.unique(df["ACIWF.decision_final"].values)
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


    for i in range(0, len(unique)-1):
        ax[0].plot(df.index, [i+0.5] * len(df), "k--")
        ax[1].plot(df.index, [i+0.5] * len(df), "k--")

    _ax = ax[0]
    _ax.set_xticklabels([])
    _ax.set_ylabel("ACI")
    _ax.bar(df.index, df["ACIWF.decision"], label = bar_labels, color=bar_color)
    #

    _ax = ax[1]
    _ax.set_xticklabels([])
    _ax.set_ylabel("ACI")
    for i, wc in enumerate(unique):
        d = np.where(df["ACIWF.decision_final"] != wc, np.nan, df["ACIWF.decision"].values)
        _ax.plot(df.index, d, label=ACICLASS_2_TEXT(wc), color=colors[i])
        _ax.scatter(df.index, d, color=colors[i], marker=".")

    _ax.legend(title='ACI class label')
    longest_uninterrupted = {}
    next_classes = {}

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

    figs = [fig]
    dfs = [stats_nexts, stats_longest_uni, stats]
    return dfs, figs

def generate_data(pv_instance) -> pd.DataFrame:
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
        # early_stop=288 * 360,
        enable_description=True
    )
    print("columns: ", ex.predictions.columns)
    pred = ex.predictions[["data", "ACIWF.decision_final", "ACIWF.decision", "ACIWF.ACI(3)"]]

    pred = pred.dropna()

    # plotter = ex.plot(include_columns=["data", "ACIWF", "ACIWF.decision_final", "ACIWF.ACI(3)", "ACIWF.OCI.expected_from_model", "ACIWF.decision"])
    # plotter.show()
    # plt.show()
    return pred

def generate_or_load_aci(pv_instance=0, persist_file="cm/paper_experiment_aci", enable_load=True) -> pd.DataFrame:
    plotter = None
    file_path = persist_file + f".{pv_instance}.csv"
    if os.path.exists(file_path) and enable_load:
        pred = pd.read_csv(file_path, header=0)
        pred['timestamp'] = pd.to_datetime(pred['timestamp'])
        pred.index = pred['timestamp']
        pred.drop(columns=["timestamp"], inplace=True)
    else:
        pred = generate_data(pv_instance)
        pred.to_csv(file_path)

    return pred

def main(pv_instances):
    all_dfs = []
    indexes = []
    all_figures = []
    for pv_instance in pv_instances:
        pred = generate_or_load_aci(pv_instance=pv_instance, enable_load=True)
        dfs, figs_all_data  = plot_weather_classes_vs_energy_per_day(pred, pv_instance)
        _, figs_one_year = plot_weather_classes_vs_energy_per_day(pred[:pred.index[0] + pd.DateOffset(365)], pv_instance)
        all_figures = all_figures + figs_all_data + figs_one_year

        indexes.append(f"PV {pv_instance}")
        all_dfs.append(dfs)

    stats_nexts, stats_longest_uni, stats = join_dataframes(all_dfs, indexes, [DATASET, WEATHER_CLASS])
    txt = df_2_latex_str(stats_nexts, caption=f"{WEATHER_CLASS} changes structure in percent [\%]",
                         command_name="WeatherChangesStructure")
    txt += df_2_latex_str(stats_longest_uni,
                         caption=f"Uninterrupted {WEATHER_CLASS} chain length statistics by {WEATHER_CLASS}",
                         command_name="LongestUninterrupted")
    txt += df_2_latex_str(stats, caption=f"Power production statistics by {WEATHER_CLASS} ",
                         command_name="PowerProductionStats")
    with open("cm/paper_experiment_aci.tex", "w") as f:
        f.write(txt)
    print_or_save_figures(figures=all_figures, path="cm/paper_experiment_aci.pdf")


if __name__ == "__main__":
    pv_instances = [0,1,2]
    main(pv_instances)
    plt.show()