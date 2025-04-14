from typing import List, Callable, Iterable, Tuple
if __name__ == "__main__": import __config__

import pandas as pd
import numpy as np


from utils.Plotter import Plotter
from utils.ExecutionTimer import ExecutionTimer

from sktime.split import SlidingWindowSplitter

class NamedObject:
    def __init__(self, obj, name:str):
        self._obj = obj
        self._name = name
    def __call__(self, *args, **kwargs):
        if self._obj is not None:
            return self._obj(*args, **kwargs)
        else:
            return None

    def __str__(self):
        return self._name

class SlidingWindowExperimentBase:
    def __init__(self, swe_model_prefix:str=None,**kwargs):
        """

        :param swe_model_prefix: prefix that is added to each model tested inside this particular SlidingWindowExperiment.
                                 during results comparition it allows to distuingish models. Model name is constructed as
                                 follows: {str(model)}{::swe_model_prefix if not none}[{d for d in dims}]
        :param kwargs:
        """
        self._set_params(**kwargs)
        self._models = []
        self.swe_model_prefix = swe_model_prefix
        self._models_name = []
        self._metrics = [NamedObject(None, "FT"), NamedObject(None, "PT")] #fit time and predict time. underscore excludes it from automatic execution in call
        self._models_dimensions = [] #list od dimensions included for selected model
        self._y = None
        self._y_all_dims = [] # field contains list of all available models' features (empty until register_dataset)
        self._models_n = []
        self._models_n_step = []

    @property
    def max_predict_window_offset(self):
        l = [n*n_step for n,n_step in zip(self._models_n, self._models_n_step) if n is not None and n_step is not None]
        if len(l) > 0:
            return max(l)
        return 0

    @property
    def models(self):
        return self._models
    @property
    def model_names(self):
        return [str(model) for model in self._models]
    @property
    def metrics(self):
        return self._metrics
    @property
    def all_dims(self):
        return self._y_all_dims
    def register_model(self, model, name:str="", dims: List[str]=None, n:int=None, n_step:int=1):
        """
        :param model: sklearn-like model (fit, predict) functions
        :param name: name of the model. It will be used as keys in dictionaries
        :param dims: names of dimensions that model consumes during fitting and prediction
        :param n: number of days that shoudl be included in the model's dataset. Refer to 'call_select_model_dataset'
        :param n_step: len of step if n is provided Refer to 'call_select_model_dataset'
        :return:
        """
        if self._y is None:
            raise RuntimeError("You can register model only if the dataset is already registered! Use register_dataset first.")

        self._models.append(model)

        if dims is not None and not all(c in self._y_all_dims for c in dims):
            raise RuntimeError(
                f"All model dimensions {dims} must be included in dataset features {self._y_all_dims}.")

        self._models_dimensions.append(dims)
        self._models_n.append(n)
        self._models_n_step.append(n_step)

        dims_str = ""
        if dims is not None:
            dims_str = "[" + ",".join(dims) + "]"
        if name == "":
            name = str(model)

        if self.swe_model_prefix is not None:
            name += f"::{self.swe_model_prefix}"

        steps_postfix = ""
        if n is not None:
            steps_postfix = f"*n={n},step={n_step}"

        self._models_name.append(name+dims_str+steps_postfix)

    def register_models(self, models : List):
        for model in models:
            self.register_model(model)

    def register_metric(self, metric: Callable[[Iterable, Iterable], float], name: str) -> None:
        self._metrics.append(NamedObject(metric, name))
    def register_metrics(self, metrics: List):
        """
        Register multiple metrics
        :param metrics: a list of tuples containing [(callable, name), (callable, name), ...]
        """
        for metric in metrics:
            self._metrics.append(NamedObject(metric[0], metric[1]))
    def register_dataset(self, dataset: pd.DataFrame) -> None:
        """
        Function registers dataset to the model. It will be used to iterate over. Function does not distinguish
        features and labels. It simply passes it to transformer functions:
        fit_generate_dimensions_and_make_dataset, predict_generate_dimensions_and_make_dataset.


        :param dataset: pd.DataFrame indexed by time range
        :return: None
        """
        self._y = dataset
        print(f"Dataset Registered. Shape: y={self._y.shape}. Columns = {self._y.columns.to_list()}, Time range = {self._y.index[[0,-1]]}")
        _, _, self._y_all_dims = self.fit_shape_dimensions()



    def _init_metrics_ts(self, splits):
        """Function prepares metrics timeseries dataframe"""
        # metrics shape: columns = len(metrics)*len(models) rows: len(splits)
        return pd.DataFrame(
            {},
            index=[i for i in range(splits)],
            columns=
                [str(model) + ":" + str(metric) for model in self.models for metric in self.metrics]
                #[str(model) + ":" + str(metric) for model in self.models for metric in ["FT", "PT"]] #Fit time, Predict time
        )
    def _init_metrics_df(self):
        """Function prepares metrics summary dataframe"""
        return pd.DataFrame({}, index=self._models_name, columns=pd.MultiIndex.from_tuples(
            [(str(metric), t) for metric in self.metrics for t in ["Mean", "Std"]], names=['Name', 'Param']))

    def fit_generate_dimensions_and_make_dataset(self,train_ts:pd.DataFrame, fh:int, predict_window_length:int) -> Tuple[pd.Series, pd.DataFrame]:
        """
            Method can be overwritten:  Function fits dimensions generator, then generate extra dimensions. Finally,
            transforms them into features (X) and labels (Y).
            :param train_ts: part of dataset Timeseries that is available during this training. contains only past data
            :return: pd.Series train_ds_y containing labels, pd.Dataframe containing features train_ds_x
        """
        train_ds_y = train_ts[train_ts.columns[0]].iloc[fh:]
        train_ds_x = train_ts[train_ts.columns[1:]].iloc[:-fh]
        return train_ds_y, train_ds_x

    def predict_generate_dimensions_and_make_dataset(self, predict_ts:pd.DataFrame, test_ts:pd.DataFrame, fh:int) -> Tuple[pd.Series, pd.DataFrame]:
        """
            Method can be overwritten: Function uses already fitted dimension generators, to generate features in predict_ts which is a window of
            past observations that are available for prediction. Then function can use those past data (and extra dimensions)
            to describe future data stored in test_df. Finally, transforms test_df into features (X) and labels (Y).
            In this function no fit method can be used!
            :param predict_ts: part of dataset Timeseries that is available during this prediction. contains only past data
            :param test_df: future data ordered by time. This set must be transform into into features (X) and labels (Y).
            :return: pd.Series test_ds_y containing labels, pd.DataFrame containing features test_ds_x
        """
        test_ds_y = test_ts[test_ts.columns[0]]
        test_ds_x = predict_ts[test_ts.columns[1:]].iloc[-fh:]
        # test_ds_x = pt[pt.columns[0:1]].values
        return test_ds_y, test_ds_x

    def _set_params(self, *args, **kwargs):
        self.learning_window_length_ = kwargs.get("learning_window_length", 360 * 288)
        self.forecasting_horizon_ = kwargs.get("forecasting_horizon", 1 * 288)
        self.predict_window_length_ = kwargs.get("predict_window_length", 1 * 288)
        self.train_pause_ = kwargs.get("train_pause", 360)

    def _init_fh(self):
        fh = np.arange(self.forecasting_horizon_) + 1
        return fh

    def _init_splitter(self):
        fh = self._init_fh()
        ts_cv = SlidingWindowSplitter(window_length=self.learning_window_length_, fh=fh,
                                      step_length=self.forecasting_horizon_)
        return ts_cv

    def fit_shape_dimensions(self, *args, **kwargs):
        """Function returns shape of fitting data"""
        self._set_params(*args, **kwargs)
        ts_cv = self._init_splitter()
        split = next(ts_cv.split(self._y))
        train_ds_y, train_ds_x = self.fit_generate_dimensions_and_make_dataset(
            train_ts=self._y.iloc[:2*self.forecasting_horizon_],
            fh=self.forecasting_horizon_,
            predict_window_length=self.predict_window_length_
        )
        return train_ds_y, train_ds_x, train_ds_x.columns.to_list()

    def predict_shape_dimensions(self, *args, **kwargs):
        """Function returns shape of fitting data"""
        self._set_params(*args, **kwargs)
        ts_cv = self._init_splitter()
        split = next(ts_cv.split(self._y))
        test_ds_y, test_ds_x = self.predict_generate_dimensions_and_make_dataset(
            predict_ts=self._y.iloc[:2*self.forecasting_horizon_].iloc[-self.predict_window_length_-self.max_predict_window_offset:],
            test_ts=self._y.iloc[split[1]],
            fh=self.forecasting_horizon_
        )
        return test_ds_y, test_ds_x, test_ds_x.columns.to_list()

    def summary(self, *args, **kwargs):
        """Prints SWE parameters descriptions. Function is useful during debugging"""
        self._set_params(*args, **kwargs)
        ts_cv = self._init_splitter()
        test_cases = ts_cv.get_n_splits(self._y)
        train_ds_y, train_ds_x, train_ds_x_dims  = self.fit_shape_dimensions()
        test_ds_y, test_ds_x, test_ds_x_dims = self.predict_shape_dimensions()
        print("===========================================")
        np_printoptions = np.get_printoptions()
        np.set_printoptions(threshold=10)
        print(f"{self.__class__.__name__}.summary:\n"
              f" - forecasting_horizon: {self.forecasting_horizon_} = {self._init_fh()}\n"
              f" - learning_window_length: {self.learning_window_length_ }\n"
              f" - predict_window_length: {self.predict_window_length_}\n"
              f" - train_pause: {self.train_pause_}\n"
              f" - test_cases: {test_cases}\n"
              f" - fit_x.shape: {train_ds_x.shape}, fit_y.shape: {train_ds_y.shape}\n"
              f" - predict_x.shape: {test_ds_x.shape}, predict_y.shape: {test_ds_y.shape}"
              f" - fit_x_dims: {train_ds_x_dims}"
              f" - predict_x_dims: {test_ds_x_dims}"
        )
        np.set_printoptions(**np_printoptions)
        print("===========================================")

        print("Registered models:")
        for model_indx, (name, m, dims, n, n_step) in enumerate(
                zip(self._models_name, self.models, self._models_dimensions, self._models_n, self._models_n_step)):
            train_ds_y_np, train_ds_x_np = self.call_select_model_dataset(train_ds_y, train_ds_x, dims, n, n_step)
            test_ds_y_np, test_ds_x_np = self.call_select_model_dataset(test_ds_y, test_ds_x, dims, n, n_step)
            print(f" - {name} -> {dims}. n={n} n_step={n_step}\n"
                  f"     fit_x.shape: {train_ds_x_np.shape}, fit_y.shape: {train_ds_y_np.shape},\n"
                  f"     predict_x.shape: {test_ds_x_np.shape}, predict_y.shape: {test_ds_y_np.shape},")
            if "tf" in name:
                m.summary()

        print("===========================================")

    def call_fit_dataset(self, train_ts: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
            Sub function of call method. It executes fit_(...)_dataset therefore generates fit dataset.
            :param train_ts: subset of self._y (dataset) selected during this particular test iteration. Based on this
                             data learning set will be prepared.
            :return: Two elements tuple: train_ds_y, train_ds_x created by fit_generate_dimensions_and_make_dataset.
        """
        train_ds_y, train_ds_x = self.fit_generate_dimensions_and_make_dataset(
            train_ts=train_ts,
            fh=self.forecasting_horizon_,
            predict_window_length=self.predict_window_length_
        )
        return train_ds_y, train_ds_x

    def call_predict_dataset(self, train_ts:pd.Series, test_ts:pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        test_ds_y, test_ds_x = self.predict_generate_dimensions_and_make_dataset(
            predict_ts=train_ts.iloc[-self.predict_window_length_-self.max_predict_window_offset:],
            test_ts=test_ts,
            fh=self.forecasting_horizon_
        )
        return test_ds_y, test_ds_x

    def call_select_model_dataset(self, train_ds_y : pd.Series, train_ds_x : pd.DataFrame, dims: List[str], n:int, n_step:int) -> Tuple[np.array, np.array]:
        """
        Function limits already prepared learning dataset for each model separately using defined dimensions set (list)
        which is provided for each model in self.register_model. Moreover, function transforms pd.DataFrames into numpy arrays
        that can be consumed by sklearn native models. Method may be overwritten in case of trajectory forecasting.
        :param train_ds_y: call_fit_dataset first returned tuple element fitting labels set (Y)
        :param train_ds_x: call_fit_dataset second returned tuple element fitting features set (X)
        :param dims: list of columns in X (aka list of features) used by particlular model
        :param n: n tells the model how many days (window-size) shoudl describes test sets. If n is not none shape of returned
                  data changes into (samples, time, features) instead of (samples, features) which is a default behavior.
                  n is equals to shape[1] (len(time)) it can be either 1 or grater. If 1 it only reshapes dataset. n_step
                  parameter is closly connected with n. If n is > 1 then each time sample is lagged by n_steps e.g. n_steps=1 =>
                  time(n-1) = time(n) - 1, if n_steps=10 => time(n-10) = time(n) - 10 (time is counted in samples not time units(seconds, minutes etc.)
        :param n_step: see n argument description
        :return: Tuple of train_ds_y, train_ds_x containing limited number of columns (by model's dimensions)- prepared for current test iteration
        """
        # reduce dataset
        if dims is None: # check if for selected model dimensions are defined. If so then adjust learning set
            ds_y, ds_x = train_ds_y.values, train_ds_x.values # 2D (samples, 1) or 1D (samples), 2D (samples, features)
        else:
            ds_y, ds_x = train_ds_y.values, train_ds_x[dims].values # 2D (samples, 1) or 1D (samples), 2D (samples, features)

        if len(ds_y.shape) == 1:
            ds_y = ds_y.reshape(-1, 1) # ensure it is 2D

        if n is not None:
            # ds_y = np.stack([ds_y[i:(-n+i),:] for i in range(n)], axis=1) #3D (samples, n, 1) n=>time
            ds_x = np.stack([ds_x[i:(-(n*n_step)+i),:] for i in range(n)], axis=1) #3D (samples, n, features) n=>time


        min_shape = min([ds_x.shape[0], ds_y.shape[0]])
        ds_y = ds_y[-min_shape:]
        ds_x = ds_x[-min_shape:]
        return ds_y, ds_x

    def call_check_prediction_performance(self, ground_truth:np.array, prediction:np.array, i:int, model_indx:int, metrics_ts:pd.DataFrame) -> None:
        """
        Function evaluate prediction over ground_truth for each defined metric and stores results in metrics_ts
        :param ground_truth: ground_truth data from dataset Should be 1D array
        :param prediction: prediction generated by model (at model_indx index in self.models). Should be 1D array
        :param i: current experiment iteration
        :param model_indx: model index in self.models list
        :param metrics_ts: pd.DataFrame used to store computed metrics
        :return: None
        """
        for metric_indx, metric in enumerate(self.metrics):
            prediction[np.isnan(prediction)] = 0
            val = metric(ground_truth, prediction)
            if val is not None:  # checks if metric.call is proper function. It is only if returns list of values
                metrics_ts.iloc[i, model_indx * len(self.metrics) + metric_indx] = val

    def call_fit(self, train_ds_y:pd.Series, train_ds_x:pd.DataFrame, metrics_ts:pd.DataFrame, i:int)->None:
        """
        Function train all models with already prepared train_ds_y and train_ds_x by call_fit_dataset. It stores fitting
         time in metrics_ts
        :param train_ds_y: from call_fit_dataset
        :param train_ds_x: from call_fit_dataset
        :param metrics_ts: store for metrics
        :param i: current test iteration
        :return:
        """
        for model_indx, (name, m, dims, n, n_step) in enumerate(zip(self._models_name, self.models, self._models_dimensions, self._models_n, self._models_n_step)):
            train_ds_y_np, train_ds_x_np = self.call_select_model_dataset(train_ds_y, train_ds_x, dims, n, n_step)

            if dims is None:
                print(f"fitting {m} on full dataset dimensions={self._y_all_dims}...")
            else:
                print(f"fitting {m} on limited dataset dimensions={dims}...")

            ft = ExecutionTimer()
            with ft:  # measure execution time
                m.fit(train_ds_x_np, train_ds_y_np)

            print(f"done in {ft.seconds_elapsed}s")
            metrics_ts.iloc[i][f"{str(m)}:FT"] = ft.seconds_elapsed

    def call_predict(self, test_ds_y, test_ds_x, metrics_ts:pd.DataFrame, prediction_ts:pd.DataFrame, prediction_features_ts:pd.DataFrame, i:int, test:np.array)->None:
        """
        Function tests all models with already prepared test_ds_y and test_ds_x by call_predict_dataset method.
        It stores predictions in prediction_ts and prepares metrics for each prediction which are stored in metrics_ts
        :param test_ds_y: from call_predict_dataset
        :param test_ds_x: from call_predict_dataset
        :param metrics_ts: store for metrics
        :param prediction_ts: store for predictions
        :param prediction_features_ts: store for predictions features
        :param i: current test iteration
        :param test: np.array containint indicies of current selection of test_ts. It is obtained from sliding window
                     splitter
        :return:
        """
        prediction_features_ts.iloc[test] = test_ds_x[-len(test):] #agree array sizes, since test_ds_x amy be longer if n and n_steps are defined for any model
        for model_indx, (name, m, dims, n, n_step) in enumerate(
            zip(self._models_name, self.models, self._models_dimensions, self._models_n, self._models_n_step)):

            pt = ExecutionTimer()

            test_ds_y_np, test_ds_x_np = self.call_select_model_dataset(test_ds_y, test_ds_x, dims, n, n_step)
            with pt:  # measure execution time
                prediction_y = m.predict(test_ds_x_np)#.reshape(self.forecasting_horizon_)
                cumprod = np.cumprod(prediction_y.shape)[-1]
                if cumprod != self.forecasting_horizon_:
                    if "tf" in name:
                        m.summary()
                    raise RuntimeError(f"SWE.call_predict: Model {name} must return {self.forecasting_horizon_} number of samples while array of shape\n"
                                       f"{prediction_y.shape} = {cumprod}, was returned. Product is not equal to {self.forecasting_horizon_}. If you used\n"
                                       f"Tensorflow ensure that last layer has proper size. In case of sklearn models it\n"
                                       f"should be done automatically.")

                # print(f"prediction: {name} y.shape:",prediction_y.shape)

            metrics_ts.iloc[i][f"{str(m)}:PT"] = pt.seconds_elapsed
            # in case of trajectory forecasting self.predict_gene(...)_dataset returns 2D array that shape[0] = 1 and
            # shape[1] = fh. It is for consistency (same shape as fitting). It requires flattening
            test_ds_y_np = test_ds_y_np.flatten()
            prediction_y = prediction_y.flatten()
            prediction_ts.iloc[test, model_indx] = prediction_y

            # check prediction performance
            self.call_check_prediction_performance(
                ground_truth=test_ds_y_np.flatten(),
                prediction=prediction_y.flatten(),
                i=i, model_indx=model_indx, metrics_ts=metrics_ts)


    def __call__(self, *args, **kwargs):
        """Performs experiment using sliding window splitter approach. It stores results and computes metrics."""
        self._set_params(*args, **kwargs)
        ts = self._y

        # prepare splitter
        # fh = [i+1 for i in range(forecasting_horizon+1)] # list of predicted samples

        # prepare output structures
        prediction_ts = pd.DataFrame({}, index=ts.index, columns=[m for m in self._models_name])
        prediction_features_ts = pd.DataFrame({}, index=ts.index, columns=[m for m in self._y_all_dims])

        ts_cv = self._init_splitter()
        test_cases = ts_cv.get_n_splits(ts)
        metrics_ts = self._init_metrics_ts(test_cases)

        # self.summary()
        execution = ExecutionTimer()
        with execution:
            for i, (train, test) in enumerate(ts_cv.split(ts)):
                # init test iteration: select test iteration train and test datasets
                train_ts = ts.iloc[train]
                test_ts = ts.iloc[test]

                print(f"it {i}. Td: {execution.seconds_delta}s, Tt: {execution.seconds}. {train[[0, -1]]}, x: {train_ts.shape}, y: {test_ts.shape}")

                # train once per train_pause iterations
                if i % self.train_pause_ == 0:
                    # init train: prepare train dataset
                    train_ds_y, train_ds_x = self.call_fit_dataset(train_ts=train_ts)
                    #train (fit)
                    self.call_fit(train_ds_y=train_ds_y,train_ds_x=train_ds_x,metrics_ts=metrics_ts,i=i)

                # init test: prepare test dataset
                test_ds_y, test_ds_x = self.call_predict_dataset(train_ts, test_ts)

                # test (predict)
                self.call_predict(
                    test_ds_y=test_ds_y,
                    test_ds_x=test_ds_x,
                    metrics_ts=metrics_ts,
                    prediction_ts=prediction_ts,
                    prediction_features_ts=prediction_features_ts,
                    i=i,
                    test=test)
        # sum up
        self.metrics_df_ = self._init_metrics_df()
        for model_indx, model in enumerate(self.models):
            for metric_indx, metric in enumerate(self.metrics):
                self.metrics_df_.loc[self._models_name[model_indx], (str(metric), "Mean")] = np.nanmean(metrics_ts.iloc[:, model_indx*len(self.metrics) + metric_indx])
                self.metrics_df_.loc[self._models_name[model_indx], (str(metric), "Std")] = np.nanstd(metrics_ts.iloc[:, model_indx*len(self.metrics) + metric_indx])

        # prediction_features_ts.drop(columns="y", inplace=True)
        self.output_ts_ = pd.concat([ts[train_ts.columns[0]], prediction_ts, prediction_features_ts], axis=1)
        return self.output_ts_, self.metrics_df_, self.learning_window_length_

    def show_fit_dimensions(self, dimensions:List[str] = None):
        pdc = self.get_fit_dimensions(dimensions)

        plotter = Plotter.from_df(pdc)
        plotter.show()
        return plotter

    def get_fit_dimensions(self, dimensions:List[str] = None):
        _, train_ds_x = self.call_fit_dataset(train_ts=self._y)
        pdc = pd.concat([self._y, train_ds_x], axis=1)
        if not dimensions is None:
            pdc = pdc[dimensions]

        return pdc

    def show_results(self, dimensions:List[str] = None):

        used_dims = set()
        for dims in self._models_dimensions:
            used_dims.update(dims)

        if isinstance(dimensions, list):
            dimensions = self._y.columns.to_list() + self._models_name + dimensions + list(used_dims)
        else:
            dimensions = self._y.columns.to_list() + self._models_name + list(used_dims)

        print("=== Results ===")
        print(self.metrics_df_)
        print("list of available dimensions:", self.output_ts_.columns.to_list())
        plotter = Plotter.from_df(self.output_ts_[dimensions][self.learning_window_length_:])
        plotter.show()
        return plotter
