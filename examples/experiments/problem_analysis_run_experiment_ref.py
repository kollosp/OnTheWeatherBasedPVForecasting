import copy
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from typing import Tuple
if __name__ == "__main__": import __config__

from dimensions import ChainDimension, Elevation, DayProgress, OCI, VCI, ACI
from dimensions import OCIModel, SolarDayProgress, Quantization, MeanBySolarDay
from dimensions import Declination
import numpy as np
from ANN.model_wrappers import *
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from SlidingWindowExperiment import SlidingWindowExperimentBase
import argparse
from examples.experiments.problem_analysis_run_experiment import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Experiment Ref',
        description='Program performs experiments. It iterate over all provided combinations of forecasting horizons, '
                    'different models, data sets, n-values, n_steps and dimensions (fetures). This script does not use '
                    'any arguments. This message is only for information. The results of this script execution are saved '
                    'in \'cm\' directory. Output consists of two types files the \'concat\' files contains aggregated '
                    'model results, while \'prediction\' files contain prediction results. The results can be processed '
                    'by Results Analysis script (problem_analysis_results).')
    args = parser.parse_args()
    parser.print_help()

    main(
        forecasting_horizons=[288],  # 24h
        prediction_step_len=288,  # it must be equal to the shortest forecasting horizon
        models = [
            (lambda n,dims,: ModelCNN(input_shape=(n,dims), output_shape=1), "CNN"),
            (lambda n,dims,: ModelLSTM(input_shape=(n,dims), output_shape=1), "LSTM"),
            (lambda n,dims,: ModelMLP(input_shape=(n,dims), output_shape=1), "MLP"),
        ],
        instance_values = [
            0, 1, 2
        ],
        n_values = [
            10, 28, 56, 84
        ], n_steps_values = [10],
        dims = [
            # for plain models
            ["y", "SolarDay%", "Elevation"],
            ["y", "SolarDay%"],
            ["y", "Elevation"],
            ["SolarDay%", "Elevation"],
            ["SolarDay%"],
            ["Elevation"],
            ["y"],
            # ["y", "SolarDay%", "Elevation", "qACId(4)"],
            # ["y", "SolarDay%", "qACId(4)"],
            # ["y", "Elevation", "qACId(4)"],
            # ["SolarDay%", "Elevation", "qACId(4)"],
            # ["SolarDay%", "qACId(4)"],
            # ["Elevation", "qACId(4)"],
            # ["y", "qACId(4)"],
            # ["y", "SolarDay%", "Elevation", "qACId(6)"],
            # ["y", "SolarDay%", "qACId(6)"],
            # ["y", "Elevation", "qACId(6)"],
            # ["SolarDay%", "Elevation", "qACId(6)"],
            # ["SolarDay%", "qACId(6)"],
            # ["Elevation", "qACId(6)"],
            # ["y", "qACId(6)"],
            # ["y", "SolarDay%", "Elevation", "qACId(8)"],
            # ["y", "SolarDay%", "qACId(8)"],
            # ["y", "Elevation", "qACId(8)"],
            # ["SolarDay%", "Elevation", "qACId(8)"],
            # ["SolarDay%", "qACId(8)"],
            # ["Elevation", "qACId(8)"],
            # ["y", "qACId(8)"],

        ]
    )

    # for qacid in [[], ["qACId(4)"], ["qACId(6)"], ["qACId(8)"]]:
    main(
        forecasting_horizons=[288],  # 24h
        prediction_step_len=288,  # it must be equal to the shortest forecasting horizon
        models=[
            (lambda *args: make_pipeline(PolynomialFeatures(12), LinearRegression()), "LR(12)"),
            (lambda *args: KNeighborsRegressor(10), "KNN(10)"),
            # (lambda *args: make_pipeline(PolynomialFeatures(16), LinearRegression()), "LR(16)"),
            (lambda *args: RandomForestRegressor(), "RF")
        ],
        instance_values=[
            0, 1, 2
        ],
        n_values=[
            None  # 1  #, 56, 84
        ], n_steps_values=[None],
        # covered_dimensions={"y","SolarDay%", "Declination", "Elevation"},
        # excluded_dimensions=qacid
        dims=[
            # for plain models
            ["y", "SolarDay%", "Elevation"],
            ["y", "SolarDay%"],
            ["y", "Elevation"],
            ["SolarDay%", "Elevation"],
            ["SolarDay%"],
            ["Elevation"],
            ["y"],
            # ["y", "SolarDay%", "Elevation", "qACId(4)"],
            # ["y", "SolarDay%", "qACId(4)"],
            # ["y", "Elevation", "qACId(4)"],
            # ["SolarDay%", "Elevation", "qACId(4)"],
            # ["SolarDay%", "qACId(4)"],
            # ["Elevation", "qACId(4)"],
            # ["y", "qACId(4)"],
            # ["y", "SolarDay%", "Elevation", "qACId(6)"],
            # ["y", "SolarDay%", "qACId(6)"],
            # ["y", "Elevation", "qACId(6)"],
            # ["SolarDay%", "Elevation", "qACId(6)"],
            # ["SolarDay%", "qACId(6)"],
            # ["Elevation", "qACId(6)"],
            # ["y", "qACId(6)"],
            # ["y", "SolarDay%", "Elevation", "qACId(8)"],
            # ["y", "SolarDay%", "qACId(8)"],
            # ["y", "Elevation", "qACId(8)"],
            # ["SolarDay%", "Elevation", "qACId(8)"],
            # ["SolarDay%", "qACId(8)"],
            # ["Elevation", "qACId(8)"],
            # ["y", "qACId(8)"],
        ]
    )

    # there is a list of other deep learning models that can be evaluated
    # main(
    #     forecasting_horizon=288,
    #     models=[
    #         #(lambda n, dims,: DeepResidualCNN_ComplexV2(input_shape=(n, dims), output_shape=1), "DeepResidualCNN_ComplexV2"),
    #         #(lambda n, dims,: DeepResidualCNN_Regularized(input_shape=(n, dims), output_shape=1), "DeepResidualCNN_Regularized"),
    #         #(lambda n, dims,: DeepMLP_ComplexV1(input_shape=(n, dims), output_shape=1), "DeepMLP_ComplexV1"),
    #         #(lambda n, dims,: DeepMLP_ComplexV2(input_shape=(n, dims), output_shape=1), "DeepMLP_ComplexV2"),
    #         (lambda n, dims,: EnsembleNetwork(input_shape=(n, dims), output_shape=1), "EnsembleNetwork"),
    #         (lambda n, dims,: TransformerEncoder(input_shape=(n, dims), output_shape=1), "TransformerEncoder"),
    #         (lambda n, dims,: DenseResNet(input_shape=(n, dims), output_shape=1), "DenseResNet"),
    #         (lambda n, dims,: AttentionLSTM(input_shape=(n, dims), output_shape=1), "AttentionLSTM"),
    #         (lambda n, dims,: AdvancedHybrid(input_shape=(n, dims), output_shape=1), "AdvancedHybrid"),
    #         (lambda n, dims,: BidirectionalLSTM(input_shape=(n, dims), output_shape=1), "BidirectionalLSTM"),
    #         (lambda n, dims,: DeepMLP(input_shape=(n, dims), output_shape=1), "DeepMLP"),
    #         (lambda n, dims,: HybridCNNLSTM(input_shape=(n, dims), output_shape=1), "HybridCNNLSTM"),
    #         (lambda n, dims,: BasicGRU(input_shape=(n, dims), output_shape=1), "BasicGRU")
    #     ],
    #     instance_values=[
    #         0
    #     ],
    #     n_values=[
    #         10
    #     ], n_steps_values=[28],
    #     # covered_dimensions={},
    #     covered_dimensions={"y", "Declination", "SolarDay%", "Elevation"},
    #     excluded_dimensions=[]
    # )

