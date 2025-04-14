if __name__ == "__main__": import __config__

import pandas as pd

from utils.ArticleUtils import print_or_save_figures, df_2_latex_str, join_dataframes

dfs = []
# for pv 0
df = pd.DataFrame.from_dict(data={
    "NaiveForecaster": [0.1,0.1]
}, columns=["MAE", "MSE"],  orient='index')
dfs.append(df)

# for pv 1
df = pd.DataFrame.from_dict(data={}, columns=["MAE", "MSE"],  orient='index')
dfs.append(df)

# for pv 2
df = pd.DataFrame.from_dict(data={}, columns=["MAE", "MSE"],  orient='index')
dfs.append(df)

pred = join_dataframes([dfs], ["PV 0", "PV 1", "PV 2"], ["\ds{}", "Model"])
print(pred[0])

# PV 0 & NaiveForecaster(sp=288) & 0.591 & 2.049 \\
# \cline{1-4}
# PV 1 & NaiveForecaster(sp=288) & 0.397 & 0.738 \\
# \cline{1-4}
# PV 2 & NaiveForecaster(sp=288) & 1.583 & 12.446 \\
#
# batch = 1
# PV 0  RollingAverage({'N': 1})  0.617366   2.164520
# PV 1  RollingAverage({'N': 1})  0.339723   0.601886
# PV 2  RollingAverage({'N': 1})  1.649935  13.221165

####################################
# pierwszy eksperyment bez przeuczania dal takie rezultaty
# PV 0 SEAPFv3 0.696 2.417
# PV 1 SEAPFv3 0.740 2.168
# PV 2 SEAPFv3 1.584 9.801
####################################

####################################
# PV 0  SEAPFv2  0.959462   3.990337
# PV 1  SEAPFv2  0.571230   1.237783
# PV 2  SEAPFv2  2.244323  20.179139
# batch = 30 dni
# PV 0  SEAPFv2  0.979168   4.144284
# PV 1  SEAPFv2  0.596136   1.347692
# PV 2  SEAPFv2  2.291026  21.107262
# batch 365 learning widnow = 360
# PV 0  SEAPFv2  0.926882   3.748630
# PV 1  SEAPFv2  0.600842   1.446014
# PV 2  SEAPFv2  2.259698  20.687115
