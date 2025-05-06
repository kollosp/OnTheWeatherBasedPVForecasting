# One-Day-Ahead Photovoltaic Power Prediction Based on
Local Weather Properties Analysis
Repository contains code and dataset supporting research "One-Day-Ahead Photovoltaic Power Prediction Based on
Local Weather Properties Analysis" ECAI'2025

## Initialization
At the very beginning the python venv must be initialized and packages listed in requirements.txt needs to be installed.


## Dataset preparation

First of all the dataset file `datasets/orgdataset.tar.xz` need to be unpacked to `datasets/orgdataset.csv`. This file is an original dataset that contains raw data. Those data needs to be cleared and 
transform before processing. To achieve that use the command below.  
```
    (.venv):~/$python datasets/process_data.py
```
It will transform original dataset into its clear version. New file will appear inside `datasets/dataset.csv`

## Database display

To display dataset as interactive chart. Use arrows to move over the data.
```
(.venv):~/$python  examples/problem_analysis_database_looker.py
```

To display and create files containing images and statistics used in article
```
(.venv):~/$python  examples/paper_experiment_weather_statistics.py
```

## Experiments

ACI forecasting experiment. Script displays charts and stores statistics in a file `cm/paper_experiment_aci_forecaster.tex`
```
    (.venv):~/$python  examples/paper_experiment_aci_forecasting.py
```

Polinomial Regression, KNN and Random Forest experiment:
```
    (.venv):~/$python  examples/ecai_experiments/problem_analysis_classic_min.py
```

Deep learning experiments:
```
    (.venv):~/$python  examples/ecai_experiments/problem_analysis_deep_min.py
```
For more information about used ANN architectures refer to `examples/ecai_experiments/model_wrappers.py`.

