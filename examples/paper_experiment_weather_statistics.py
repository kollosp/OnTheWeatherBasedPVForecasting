import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__": import __config__
import matplotlib.dates as md
import os
from utils.Experimental import Experimental
from WeatherWrapper.ACIWF import Model as ACIWF
import pandas as pd
from matplotlib import pyplot as plt
from WeatherWrapper.CloudinessFeatures.OCI import Model as OCI
from WeatherWrapper.CloudinessFeatures.VCI import Model as VCI
from sktimeSEAPF.Modelv2 import Model as sktimeSEAPFv2
from utils.ArticleUtils import print_or_save_figures, df_2_latex_str, join_dataframes, concatenate_df
import scipy
from sktimeSEAPF.Optimized import Optimized
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

def plot_weather_classes_vs_energy_per_day(df, pv_instance, latitude_degrees, longitude_degrees):

    fig, ax = plt.subplots(2)
    # fig.suptitle(f"Weather classes and mean daily power by day for PV {pv_instance}")

    dataset_statistics_df = pd.DataFrame(data = {
        "Begin Date": [df.index[0].strftime("%m/%d/%Y")],
        "End Date": [df.index[-1].strftime("%m/%d/%Y")],
        "Days": [len(df) // 288],
        "Max [kW]": [max(df["data"])],
        "Lat.": [latitude_degrees],
        "Long.": [longitude_degrees],
    }, index=pd.Index([f"PV {pv_instance}"], name=f"{DATASET}"))

    df = compute_aci_aggregated_df(df)

    #correlation computation
    s, p = scipy.stats.pearsonr(df["ACIWF.decision_final"].to_numpy(),df["data"].to_numpy())
    correlation_df = pd.DataFrame(data = {"Statistic": [s], "p-value": [p]}, index=[f"PV {pv_instance}"])
    correlation_df.index.rename(f"{DATASET}", inplace=True)
    #dataset statistics

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

    colors = ['tab:green', 'tab:orange' , 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    bar_color = [colors[int(k % len(colors))] for k in df["ACIWF.decision_final"]]

    # len(unique) => weather class
    decision_region = (len(unique)-1) / len(unique)
    for i in range(1, len(unique)):
        ax[0].plot(df.index, [i*decision_region] * len(df), "k--", label=("_" if i != 1 else "") + "Decision boundaries")
        # ax[1].plot(df.index, [i*decision_region] * len(df), "k--")

    _ax = ax[0]
    _ax.set_xticklabels([])
    _ax.set_ylabel("qACI_d")
    _ax.bar(df.index, df["ACIWF.decision"], label = bar_labels, color=bar_color)
    #

    # _ax = ax[1]
    # _ax.set_xticklabels([])
    # _ax.set_ylabel("qACI_d")
    # for i, wc in enumerate(unique):
    #     d = np.where(df["ACIWF.decision_final"] != wc, np.nan, df["ACIWF.decision"].values)
    #     _ax.plot(df.index, d, label=ACICLASS_2_TEXT(wc), color=colors[i])
    #     _ax.scatter(df.index, d, color=colors[i], marker=".")
    #
    _ax.legend()
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
        # nexts = df["ACIWF.decision_final"][1:][diff < 0].values # WC change after chain
        nexts = [n for p,n  in zip(df["ACIWF.decision_final"][:-1], df["ACIWF.decision_final"][1:]) if p==wc] #[1:].values # Wc change
        next_classes[ACICLASS_2_TEXT(wc)] = 100 * np.bincount(nexts, minlength=len(unique)) / len(nexts)

        lu = [max(lengths), sum(lengths) / len(lengths),100 * sum(d) / len(df["ACIWF.decision_final"])]
        longest_uninterrupted[ACICLASS_2_TEXT(wc)] = lu

    # for i, row in enumerate(next_classes.keys()):
    #     next_classes[row][i] = np.nan

    stats_nexts = pd.DataFrame.from_dict(next_classes, orient='index',
                                               columns=next_classes.keys())
    # stats_nexts.columns = pd.MultiIndex.from_product([["Row Src. Column Dst."], stats_nexts.columns], names=["", WEATHER_CLASS])
    # stats_nexts.index = pd.MultiIndex.from_product([["Src."], stats_nexts.index ], names=["", WEATHER_CLASS])

    stats_longest_uni = pd.DataFrame.from_dict(longest_uninterrupted, orient='index', columns=["max", "mean", "share [\%]"])
    stats_longest_uni.index.rename(WEATHER_CLASS, inplace=True)

    df["ACIWF.decision_final"] = bar_labels_clear
    stats = df.groupby("ACIWF.decision_final").agg({
        "data": ["max", "mean", "min", "std", "count"]
    })
    # print(stats.columns)
    stats.index.rename(WEATHER_CLASS, inplace=True)
    stats.columns = stats.columns.droplevel()
    # stats["share [\%]"] = 100 * stats["count"] // sum(stats["count"])
    stats.drop(columns="count", inplace=True)

    figs = [fig]
    dfs = [stats_nexts, stats_longest_uni, stats, correlation_df, dataset_statistics_df]
    return dfs, figs

def generate_data(pv_instance) -> pd.DataFrame:
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/dataset.csv"])
    df = pd.read_csv(file_path, low_memory=False)
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
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
        learning_window_length=288 * 360,
        window_size=288,
        batch=288 * 360,
        # early_stop=288 * 365,
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

    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/dataset.csv"])
    df = pd.read_csv(file_path, low_memory=False)
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    df.drop(columns=["timestamp"], inplace=True)
    latitude_degrees = df[f"{pv_instance}_Latitude"][0]
    longitude_degrees = df[f"{pv_instance}_Longitude"][0]

    return pred, latitude_degrees, longitude_degrees


def generate_pv_prod_figure(df, latitude_degrees_list, longitude_degrees_list):
    file_path = "/".join(os.path.abspath(__file__).split("/")[:-2] + ["datasets/dataset.csv"])
    df = pd.read_csv(file_path, low_memory=False)
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    df.drop(columns=["timestamp"], inplace=True)

    START_DATE_GOOD = '2023-05-08'
    END_DATE_GOOD = '2023-05-10'

    START_DATE_BAD = '2020-04-29'
    END_DATE_BAD = '2020-05-01'

    f1 = draw_power_figure(df, pv_instances, "Power [kW]", START_DATE_GOOD, END_DATE_GOOD, START_DATE_BAD, END_DATE_BAD)
    f2 = draw_oci_vci_figure(df, 0, "Power [kW]", START_DATE_GOOD, END_DATE_GOOD, latitude_degrees_list[0], longitude_degrees_list[0])
    f3 = draw_oci_vci_figure(df, 0, "Power [kW]", START_DATE_BAD, END_DATE_BAD, latitude_degrees_list[0], longitude_degrees_list[0])

    return [f1,f2,f3]

def draw_oci_vci_figure(df, pv_instance, y_label, START_DATE, END_DATE,  latitude_degrees, longitude_degrees):
    fig, ax = plt.subplots(2)
    _ax = ax[0]
    lines = []

    # for pv_instance in pv_instances:
    df = df[f"{pv_instance}_Power"]
    data = df[START_DATE:END_DATE]

    d = Optimized.window_moving_avg(data, window_size=3, roll=True)
    l, = _ax.plot(data.index, d, label=f"PV {pv_instance}")
    lines.append(l)
    for _ax in ax:
        _ax.plot(data.index, np.zeros(len(data)), label="_zeros", c="black")

    oci_ = OCI(
    window_size=3,
    model_factory=lambda: sktimeSEAPFv2(
        latitude_degrees=latitude_degrees,
        longitude_degrees=longitude_degrees,
        x_bins=34, y_bins=34,
        zeros_filter_modifier=0,
        density_filter_modifier=0,
        bandwidth=0.1
    ))
    oci_ = oci_.fit(df)
    oci_value = oci_.transform(data)
    vci_ = VCI(window_size=6).fit(df)
    vci_value = vci_.transform(data)
    _ax = ax[0]
    # l, = _ax.plot(data.index, d, label=f"PV {pv_instance}")
    l, = _ax.plot(data.index,oci_value, label=f"OCI", color="gray")
    lines.append(l)
    l, = _ax.plot(data.index,oci_._predict_description()["OCI.expected_from_model"], label=f"OCI", color="red")
    lines.append(l)

    _ax = ax[1]
    _ax.set_ylabel("VCI")
    # l, = _ax.plot(data.index,vci_value, label=f"VCI")
    vci_description = vci_._predict_description()

    # Optimized.window_moving_avg(vci_description["VCI.mx"], window_size=12, roll=True)
    mx = Optimized.window_moving_avg(vci_description["VCI.mx"], window_size=3, roll=True)
    l, = _ax.plot(data.index, mx, label=f"VCI.dx/dt.max", color="red")
    lines.append(l)
    mi = Optimized.window_moving_avg(vci_description["VCI.mi"], window_size=3, roll=True)
    l, = _ax.plot(data.index,mi, label=f"VCI.dx/dt.min", color="red")
    lines.append(l)
    l = _ax.fill_between(data.index, mi, mx, label=f"VCI", color="gray")

    lines.append(l)
    _ax = ax[0]
    _ax.set_ylabel("OCI components [kW]")
    # _ax.legend(lines, [l.get_label() for l in lines])
    _ax.legend()
    _ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    _ax = ax[1]
    _ax.legend()
    # _ax.set_ylim(-5,5)
    _ax.set_ylabel("VCI components")
    _ax.set_xlabel("Time")
    _ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    fig.tight_layout()
    return fig


def draw_power_figure(df, pv_instances, y_label, START_DATE_GOOD, END_DATE_GOOD, START_DATE_BAD, END_DATE_BAD):
    fig, ax = plt.subplots(2)
    _ax = ax[0]
    lines = []
    data = df[START_DATE_GOOD:END_DATE_GOOD]
    for pv_instance in pv_instances:
        d = Optimized.window_moving_avg(data[f"{pv_instance}_Power"].to_numpy(), window_size=12, roll=True)
        l, = _ax.plot(data.index, d, label=f"PV {pv_instance}")
        lines.append(l)
    _ax.plot(data.index, np.zeros(len(data)), label="_zeros", c="black")

    _ax = ax[1]
    data = df[START_DATE_BAD:END_DATE_BAD]
    for pv_instance in pv_instances:
        d = Optimized.window_moving_avg(data[f"{pv_instance}_Power"].to_numpy(), window_size=12, roll=True)
        l, = _ax.plot(data.index, d, label=f"PV {pv_instance}")
    _ax.plot(data.index, np.zeros(len(data)), label="_zeros", c="black")

    _ax = ax[0]
    _ax.set_ylabel(y_label)
    _ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))

    _ax = ax[1]
    _ax.set_ylabel(y_label)
    _ax.set_xlabel("Time")
    _ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    _ax.legend(lines, [l.get_label() for l in lines])

    fig.tight_layout()
    return fig

def main(pv_instances):
    all_dfs = []
    indexes = []
    all_figures = []
    correlation_dfs = []
    dataset_statistics_df = []
    latitude_degrees_list = []
    longitude_degrees_list = []
    for pv_instance in pv_instances:
        pred, latitude_degrees, longitude_degrees = generate_or_load_aci(pv_instance=pv_instance, enable_load=True)
        latitude_degrees_list.append(latitude_degrees)
        longitude_degrees_list.append(longitude_degrees)
        if pv_instance == 1:
            pred = pred["2022/10/01":]
        dfs, figs_all_data  = plot_weather_classes_vs_energy_per_day(pred, pv_instance, latitude_degrees, longitude_degrees)
        _, figs_one_year = plot_weather_classes_vs_energy_per_day(pred[:pred.index[0] + pd.DateOffset(365)], pv_instance, latitude_degrees, longitude_degrees)
        all_figures = all_figures + figs_all_data + figs_one_year

        indexes.append(f"PV {pv_instance}")
        all_dfs.append(dfs[0:3])
        correlation_dfs.append(dfs[3])
        dataset_statistics_df.append(dfs[4])

    correlation_dfs = concatenate_df(correlation_dfs)
    txt = df_2_latex_str(correlation_dfs, caption="Correlation pearsonr coeficient.",
                         command_name="Correlation", float_format="{:1.3f}".format)

    dataset_statistics_df = concatenate_df(dataset_statistics_df)
    txt += df_2_latex_str(dataset_statistics_df.T, caption="Dataset statistics",
                         command_name="DatasetStatistics", float_format="{:.1f}".format)

    stats_nexts, stats_longest_uni, stats = join_dataframes(all_dfs, indexes, [DATASET, WEATHER_CLASS])
    txt += df_2_latex_str(stats_nexts, caption="Structure of \wc{} changes represented as percentages. Each row (\wc{} source) indicates the weather condition on a given day, while each column (\wc{} destination) shows the probability of transitioning to a different weather condition on the following day. The data suggests that weather conditions tend to persist, as the diagonal entries (where the source and destination \wc{} are the same) have the highest percentages. Exceptions, such as the high likelihood of transitioning from 'Good weather' to 'Fair weather' in PV 2, highlight some variations in weather patterns across installations.",
                         command_name="WeatherChangesStructure")
    txt += df_2_latex_str(stats_longest_uni,
                         caption="Statistics of uninterrupted weather conditions \wc{} chain lengths across different photovoltaic installations (\ds{}). For each combination of \ds{} and \wc{}, the mean chain length is typically just a few days, indicating that the most common pattern involves maintaining a consistent weather type for a short duration. Moreover, the maximum chain lengths, which exceed 10 days in several cases, suggest that extended periods of consistent weather patterns are possible, especially for 'Bad weather' conditions. The share percentage highlights how frequently each weather condition occurs.",
                         command_name="LongestUninterrupted")
    txt += df_2_latex_str(stats, caption="Power production statistics by weather conditions (\wc{}) for different photovoltaic installations (\ds{}). The data demonstrates a clear relationship between weather conditions and power production, with better weather conditions ('Good weather') resulting in higher mean and maximum power outputs. Conversely, 'Bad weather' conditions show significantly lower power production on average, though variability (as indicated by the standard deviation) is often higher. ",
                         command_name="PowerProductionStats")
    with open("cm/paper_experiment_aci.tex", "w") as f:
        f.write(txt)

    all_figures = all_figures + generate_pv_prod_figure(pv_instances, latitude_degrees_list, longitude_degrees_list)
    print_or_save_figures(figures=all_figures, path="cm/paper_experiment_aci.pdf")


if __name__ == "__main__":
    pv_instances = [0,1,2]
    main(pv_instances)
    plt.show()