import os.path

import pandas as pd
from typing import List, Callable, Iterable, Any, Tuple
from sktime.forecasting.model_selection import SlidingWindowSplitter


# from utils.Plotterv2 import Plotterv2 as Plotter
from utils.Plotter import Plotter as Plotter

def series_to_Xy(data: pd.Series):
    return data.index.astype('datetime64[s]').astype('int'), data.to_numpy()
    # return (data.index.to_numpy().astype(int)/9).astype(int).reshape(-1,1), data.to_numpy()

class Metric:
    def __init__(self, metric, name):
        self._metric = metric
        self._name = name
    def __call__(self, *args, **kwargs):
        return self._metric(*args, **kwargs)
    def __str__(self):
        return self._name

class Experimental:
    def __init__(self, storage_file):
        self._storage_file = storage_file
        self._predictions = None
        self._dataset = None
        self._ts, self._y = None, None
        self._models = []
        self._metrics = []

    @property
    def predictions(self):
        return self._predictions

    @property
    def dataset(self):
        return self._dataset

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
    def prediction_descriptions(self):
        return self._prediction_descriptions

    def register_dataset(self, dataset: pd.DataFrame):
        self._dataset = dataset
        self._ts, self._y = series_to_Xy(self._dataset)
        print(f"Dataset Registered. Shape: X={self._ts.shape}, y={self._y.shape}")

    def register_model(self, model):
        self._models.append(model)
    def register_models(self, models : List):
        for model in models:
            self._models.append(model)

    def register_metric(self, metric: Callable[[Iterable, Iterable], float], name: str) -> None:
        self._metrics.append(Metric(metric, name))

    def register_metrics(self, metrics: List):
        """
        Register multiple metrics
        :param metrics: a list of tuples containing [(callable, name), (callable, name), ...]
        """
        for metric in metrics:
            self._metrics.append(Metric(metric[0], metric[1]))

    def predict_or_load(self, **kwargs) -> tuple:
        forecast_start_point = kwargs.get("learning_window_length") - 1
        self._ts = self._dataset.index.astype('datetime64[s]')[forecast_start_point:]

        if not self.storage_load():
            return self.predict(**kwargs)


        predictions = [[0] * forecast_start_point + self._predictions[str(m)].to_list() for m in self._models]
        # print(predictions)

        # create metrics dataframe
        metrics_results = {}
        for metric in self._metrics:
            metrics_results[f"{metric}"] = []
            for model, prediction in zip(self._models, predictions):
                # print("+++++++++",prediction[forecast_start_point:], "+++++++++", self._dataset[forecast_start_point:])
                result = metric(prediction[forecast_start_point:], self._dataset[forecast_start_point:])
                metrics_results[f"{metric}"].append(result)

        metrics_results["Models"] = [str(m) for m in self._models]
        metrics_results = pd.DataFrame(metrics_results)
        metrics_results.set_index("Models", inplace=True)
        metrics_results.columns.rename("Metric", inplace=True)

        self.metric_results = metrics_results
        return  self._predictions, metrics_results, forecast_start_point

    def predict(self,
                forecast_horizon = 288,
                batch = 288,
                learning_window_length = 288 * 30,
                window_size=288,
                early_stop=None,
                enable_description=False) -> Tuple:
        """
        Function performs train/test loop.
        :param forecast_horizon: number of samples in prediction. Relative to end of the learning data.
        :param batch: number of predictions made between subsequent calls to the fit method.
        :param learning_window_length: length of data passed to the model during fit.
        :param window_size: length of data passed to the model during prediction as X parameter. Relative to start of current
                            forecast_horizon.
        :param early_stop: number of iteration to perform ('none' means to use all dataset).
        :param enable_description: flag enabling predict descriptors curves. Those are parameters provided by
                                   model._predict_description function. Each model returns predict_description for X
                                   given to the model during predict(.., X). Those parametes are stored in form of
                                   List[pd.DataFrame] for each registered model in self._prediction_descriptions and can be read using property
                                   self.prediction_descriptions
        :return: predictions, metrics_results, forecast_start_point
        """
        fh = list(range(forecast_horizon))
        cv = SlidingWindowSplitter(window_length=learning_window_length, fh=fh, step_length=len(fh))

        # initialize prediction results
        forecast_start_point = learning_window_length - 1
        predictions = [[0] * len(self._dataset.values) for _ in self._models]
        splits = cv.get_n_splits(self._dataset)
        self._prediction_descriptions = pd.DataFrame({})
        # make cross validation using SlidingWindowSplitter
        for i, (train, test) in enumerate(cv.split(self._dataset)):
            st = self._dataset.index[test[0]]
            en = self._dataset.index[test[-1]]
            txt = f"\rBatch learning. Iter: {i} ~ {int(100 * i/splits)}%. Period: <{st}, {en}>"
            print(txt)

            # fit models once per batch
            if i % batch == 0:
                for model in self._models:
                    model.fit(y=self._dataset[train][:-1])

            # print("Train & Test", train, len(test), test[0], test[-1])

            # current position in return data is starting point + all already made predictions
            s = forecast_start_point + i * forecast_horizon

            for j, model in enumerate(self._models):
                # self._dataset[test].index -> prediction timestamps len(test) == forecast_horizon
                # self._dataset[train[-window_size:] -> last window_size samples self._dataset[-1].index == self._dataset[test][0]-1
                # those two sets are adjusted but do not overlap
                if batch != 1: # if batch is disabled model should remember input from fit
                    x= self._dataset[train[-window_size:]]
                    prediction = model.predict(fh=self._dataset[test].index, X=x)
                else:
                    prediction = model.predict(fh=self._dataset[test].index)
                # prediction is a pandas.series contains len(test) instances
                predictions[j][s:s + len(prediction)] = prediction

            if enable_description:
                wide_df = pd.DataFrame()
                for j, model in enumerate(self._models):
                    if hasattr(model, '_predict_description') and callable(model._predict_description):
                        m_df = model._predict_description(y_true=self._dataset[test])
                        wide_df = pd.concat([wide_df, m_df], axis=1)
                self._prediction_descriptions = pd.concat([self._prediction_descriptions, wide_df])

            if early_stop is not None and i * forecast_horizon > early_stop:
                print(f"Early stop on i={i} > {early_stop}")
                break

        # create metrics dataframe
        metrics_results = {}
        for metric in self._metrics:
            metrics_results[f"{metric}"] = []
            for model, prediction in zip(self._models, predictions):
                result = metric(prediction[forecast_start_point:], self._dataset[forecast_start_point:])
                metrics_results[f"{metric}"].append(result)

        metrics_results["Models"] = [str(m) for m in self._models]
        metrics_results = pd.DataFrame(metrics_results)
        metrics_results.set_index("Models", inplace=True)

        metrics_results.columns.rename("Metric", inplace=True)
        # log = {str(model): f"l: {len(prediction)}, max:{max(prediction)}, min: {min(prediction)}" for model, prediction in zip(self._models, predictions)}
        # print(log)
        d = {str(model): prediction for model, prediction in zip(self._models, predictions)}

        d["data"] = self._dataset.values

        self._predictions = pd.DataFrame.from_dict(d)
        self._predictions.index = self._dataset.index


        if enable_description:
            self._predictions = pd.concat([self.prediction_descriptions, self._predictions], axis=1)

        self._predictions = self._predictions[forecast_start_point:]

        self._ts = self._dataset.index.astype('datetime64[s]')[forecast_start_point:]
        self.metric_results = metrics_results
        self.storage()
        return self._predictions, metrics_results, forecast_start_point

    def statistics(self):
        return self.metric_results

    def plot(self, include_columns=None, exclude_columns=None):
        # print(len(self._dataset), len(self._dataset.values), [len(self._predictions[column].to_numpy() for column in self._predictions.columns])
        columns = self._predictions.columns.tolist()
        if include_columns is not None:
            columns = include_columns

        if exclude_columns is not None:
            for ex in exclude_columns:
                columns.remove(ex)

        data_columns = [*[self._predictions[column] for column in columns]]
        plotter = Plotter(x_axis=self._ts,
                          list_of_data_or_plotter_object=data_columns,
                          list_of_line_names=columns)

        return plotter.show()

    def storage_load(self):
        """
        Function loads csv from selected storage_file directory if it is available. If data is
        loaded correctly function returns True.
        """
        if self._storage_file is not None:
            if os.path.exists(self._storage_file):
                df = pd.read_csv(self._storage_file, low_memory=False)
                # self.full_data = self.full_data[30:]
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.index = df['timestamp']
                df.drop(columns=["timestamp"], inplace=True)
                self._predictions = df
                return True
        return False

    def storage(self):
        if self._storage_file is not None:
            self._predictions.to_csv(self._storage_file)
