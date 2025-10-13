import os

import numpy as np
import pandas as pd


def clear2_df(df_):
    df__ = df_.replace({
        "SolarDay%": "$S\%$",
        "Elevation": "$E$",
        "qACId\(4\)": "$qA(4)$",
        "qACId\(6\)": "$qA(6)$",
        "qACId\(8\)": "$qA(8)$",
        "_0": "",
        "_1": "",
        "_2": "",
        "y": "$y$",
    }, regex=True)
    df__.rename(columns={
        "BestModel": "Model configuration",
        "BestModelMean": "Metric",
        "BestRefModel": "R. Model configuration",
        "BestRefModelMean": "R. Metric",
        "IR2": "IR",
    }, inplace=True)
    df__.index = df__.index.rename({
        "ML_Model": "\ml{}",
        "instance": "\ds{}",
    })

    return df__

def clear_df(df_):
    df__ = df_.replace({
        "SolarDay%": "$S\%$",
        "Elevation": "$E$",
        "qACId\(4\)": "$qA(4)$",
        "qACId\(6\)": "$qA(6)$",
        "qACId\(8\)": "$qA(8)$",
        "CNN": "",
        "MLP": "",
        "LSTM": "",
        "KNN\(10\)": "",
        "RF": "",
        "LR\(12\)": "",
        "_0": "",
        "_1": "",
        "_2": "",
        "y": "$y$",
    }, regex=True)
    df__.rename(columns={
        "BestModel": "Model configuration",
        "BestModelMean": "Metric",
        "BestRefModel": "R. Model configuration",
        "BestRefModelMean": "R. Metric",
        "IR2": "IR",
    }, inplace=True)
    df__.index = df__.index.rename({
        "ML_Model": "\ml{}",
        "instance": "\ds{}",
    })
    return df__

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "experiments_all.csv"))

df.loc[:, "instance"] = df.loc[:, "instance"].replace({
    0: "\pv0{}",
    1: "\pv1{}",
    2: "\pv2{}",
}, regex=True)
df.loc[:, "Metric"] = df.loc[:, "Metric"].replace({
    "MAE": "\\\mae{}",
    "MSE": "\\\mse{}",
    "ME": "\\\me{}",
}, regex=True)
metrics = np.unique(df["Metric"])
df = df.set_index(['instance', 'ML_Model', 'Metric']).sort_index()
df = df[['BestModel', 'BestModelMean', 'BestRefModel', 'BestRefModelMean', 'IR2']]


print(clear_df(df).to_latex(float_format="{:0.2f}".format).replace("NaN", ""))

idx = pd.IndexSlice
print(" ==== AVG ==== ")
for metric in metrics:
    print(f"metric: {metric}")
    print(pd.pivot_table(df.loc[idx[:,:,metric], :].reset_index(), values="IR2", index="instance"))
    print(df.loc[idx[:,:,metric], :].reset_index()["IR2"].mean())


print(" ================ MIN ================")
best_models = pd.DataFrame()
for metric in metrics:
    # print(f"metric: {metric}")
    l = df.loc[idx[:,:,metric], :].reset_index()
    t = pd.pivot_table(l, values="BestModelMean", index="instance", aggfunc="min")
    merged = pd.merge(t, l[["BestModelMean", "BestModel"]], left_on="BestModelMean", right_on="BestModelMean")
    merged.index = merged.index.set_names("instance")
    merged["Metric"] = metric
    best_models = pd.concat([best_models, merged])
best_models = best_models.reset_index()
best_models.loc[:, "instance"] = best_models.loc[:, "instance"].replace({
    0: "\pv0{}",
    1: "\pv1{}",
    2: "\pv2{}",
}, regex=True)
best_models.loc[:, "Metric"] = best_models.loc[:, "Metric"].replace({
    "MAE": "\\\mae{}",
    "MSE": "\\\mse{}",
    "ME": "\\\me{}",
}, regex=True)

best_models = best_models.set_index(["instance", "Metric"])
print(clear2_df(best_models.sort_index()).to_latex(float_format="{:0.2f}".format).replace("NaN", ""))
