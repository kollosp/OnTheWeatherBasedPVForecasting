from __future__ import annotations  # type or "|" operator is available since python 3.10 for lower python used this line
# lib imports
import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import EngFormatter, StrMethodFormatter
from sklearn.metrics import r2_score
from utils import Solar
from utils.Plotter import Plotter
from matplotlib import pyplot as plt
from datetime import datetime as dt
from typing import List
from collections.abc import Iterable
# package imports
# import statsmodels.api as sm
from .Optimized import Optimized
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from .Overlay import Overlay

class Model(BaseForecaster):
    _tags = {
        "requires-fh-in-fit": False
    }

    def __init__(self,
                 latitude_degrees: float = 51,
                 longitude_degrees: float = 14,
                 x_bins: int = 10,
                 y_bins: int = 10,
                 bandwidth: float = 0.4,
                 window_size: int = None,
                 enable_debug_params: bool = True,
                 zeros_filter_modifier:float=0,
                 density_filter_modifier:float=0,
                 interpolation=False,
                 return_sequences = False,
                 str_representation_limit = -1,
                 str_repr_enable_params=False
                 ):

        super().__init__()
        self.model_name = "SEAPF"
        self.str_repr_enable_params = str_repr_enable_params
        self.zeros_filter_modifier = zeros_filter_modifier
        self.density_filter_modifier = density_filter_modifier
        self.x_bins = x_bins
        self.return_sequences = return_sequences
        self.bandwidth = bandwidth
        self.y_bins = y_bins
        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.interpolation = interpolation
        # self.model_representation_ = None
        # self.elevation_bins_ = None
        # self.overlay_ = None
        # self.heatmap_ = None
        # self.kde_ = None
        self.enable_debug_params = enable_debug_params
        self.window_size = window_size  # if set then fit function performs moving avreage on the input data
        self.str_representation_limit = str_representation_limit


    @property
    def name(self):
        return self.model_name

    @property
    def model_representation(self):

        return self.model_representation_

    @property
    def str_limit(self):
        return self.str_representation_limit

    @property
    def overlay(self):
        return self.overlay_
    @property
    def debug_data(self):
        return self.debug_data_ if self.debug_data_ is not None else []

    @property
    def max_seen(self):
        return self.max_seen_

    def _fit(self, y, X=None, fh=None):
        """
        Fit function that is similar to sklearn scheme X contains features while y contains corresponding correct values
        :param X: it should be 2D array [[ts1],[ts2],[ts3],[ts4],...] containing timestamps
        :param y: it should be 2D array [[y1],[y2],[y3],[y4],...] containing observations made at the corresponding timestamps
        :param zeros_filter_modifier:
        :param density_filter_modifier:
        :return: self
        """
        self._fit_compute_statistics(y, X, fh)
        self.overlay_, self.y_adjustment_obs = self._fit_generate_overlay(y, X, fh)
        self._fit_y_adjustment(y, X, fh, self.y_adjustment_obs)


        if self.zeros_filter_modifier > 0:
            self.overlay_ = self.overlay_.apply_zeros_filter(modifier=self.zeros_filter_modifier)
        if self.density_filter_modifier > 0:
            self.overlay_ = self.overlay_.apply_density_based_filter(modifier=self.density_filter_modifier)

        self.model_representation_ = np.apply_along_axis(lambda a: self.overlay_.bins[np.argmax(a)], 0, self.overlay_.kde).flatten()
        self.model_representation_ = Optimized.window_moving_avg(self.model_representation_ , window_size=3, roll=True)
        # print(self.model _representation_)
        return self

    def _fit_y_adjustment(self, y, X=None, fh=None, y_adjustment_obs=np.zeros((1,1))):

        # polynomial regression
        self._y_adjustment_reg_ = make_pipeline(
            PolynomialFeatures(2),
            LinearRegression()
        ).fit(X=y_adjustment_obs[:,1:], y=y_adjustment_obs[:,0])
        self._y_adjustment_coef_ = 0# self._y_adjustment_reg_.coef_[0]
        self._y_adjustment_intercept_ = 0#self._y_adjustment_reg_.intercept_

    def _fit_compute_statistics(self, y, X=None, fh=None):
        self.max_seen_ = max(y)


    def _fit_generate_overlay(self, y, X=None, fh=None):

        # model is prepared to work with only one param in X
        # pandas._libs.tslibs.timedeltas.Timedelta
        self.x_time_delta_ = (y.index[-1] - y.index[0]) / len(y)
        self.y_max_ = max(y.values)

        # ts = ts + self.step_count_ * self.x_time_delta_ # if more data is in Y then those data corresponds
        # with future. So if the model uses only one (last) element
        # in y[:, ] then ts has to be adjusted
        data = y.values
        if self.window_size is not None and self.window_size >= 1:
            data = Optimized.window_moving_avg(y.values, window_size=self.window_size, roll=True)

        timestamps = y.index.astype(int) / 10 ** 9
        ts = Optimized.from_timestamps(timestamps)
        # print("elevation", self.latitude_degrees,self.longitude_degrees )
        elevation = Solar.elevation(ts, self.latitude_degrees,
                                    self.longitude_degrees) * 180 / np.pi

        # elevation[elevation <= 0] = 0
        # create assignment series, which will be used in heatmap processing
        days_assignment = Optimized.date_day_bins(timestamps)
        #print(days_assignment)
        elevation_assignment, self.elevation_bins_ = \
            Optimized.digitize(elevation, self.x_bins, mi=0,
                               mx=Solar.sun_maximum_positive_elevation(self.latitude_degrees))

        y_adjustment_obs = pd.DataFrame({
            "y": y,
            "elevation": elevation,
            "days_assignment": days_assignment
        })[elevation >= 0].groupby("days_assignment").agg({"y": "max", "elevation":"max"}).to_numpy()

        indicies = elevation > 0
        # print("_fit_generate_overlay", len(elevation), len(y), len(np.unique(days_assignment[indicies])))
        overlay = Optimized.overlay(data[indicies],
                                    elevation_assignment[indicies],
                                    days_assignment[indicies],
                                    x_bins=self.x_bins,
                                    y_bins= len(np.unique(days_assignment[indicies])))
        # print("overlay", pd.DataFrame({
        #     "y": data,
        #     "elevation": elevation,
        # }))
        # plt.imshow(data)
        # plt.show()
        # print("data[indicies]",  indicies, elevation)
        return Overlay(overlay, self.y_bins, self.bandwidth), y_adjustment_obs

    @staticmethod
    def _shape_check(shape1, shape2):
        # print("_shape_check", shape1, shape2)
        if len(shape1) != len(shape2):
            return False
        for s1,s2 in zip(shape1, shape2):
            if s1 != s2:
                return False
        return True

    def set_model_representation(self, model_representation):
        """
        For overlay insection
        """

        if not Model._shape_check(model_representation.shape,  self.model_representation_.shape):
            raise ValueError(f"SEAPF.set_model_representation: cannot set model representation. It has incorrect shape:" +
                             f" {model_representation.shape}, expected: {self.model_representation_.shape}")

        self.model_representation_ = np.concatenate([self.overlay_.bins[i] for _,i in enumerate(model_representation)])

    def plot_model(self, fig=None, ax=None, return_plot_items=False):
        if ax is None or fig in None:
            fig, ax = plt.subplots(1)
        ov = self.overlay_.overlay
        lines = []
        if return_plot_items:
            fig, ax, l = Plotter.plot_overlay(ov, fig=fig, ax=ax, return_plot_items=return_plot_items)
            lines.append(l)
        else:
            fig, ax = Plotter.plot_overlay(ov, fig=fig, ax=ax, return_plot_items=return_plot_items)

        x = list(range(ov.shape[1]))
        mean = np.nanmean(ov, axis=0)
        # mx = np.nanmax(ov, axis=0)
        # mi = np.nanmin(ov, axis=0)

        l, = ax.plot(x, self.model_representation_, color="r", label="Model representation")
        #print(l)
        lines.append(l)
        l, = ax.plot(x, mean, color="orange", label="Mean")
        lines.append(l)
        # ax.plot(x, mx, color="orange")
        # ax.plot(x, mi, color="orange")

        if return_plot_items:
            return fig, ax, lines
        return fig,ax

    def plot_y_adjustment(self, fig=None, ax=None):
        if ax is None or fig is None:
            fig, ax = plt.subplots(1)

        # alpha = 0.05  # 95% confidence interval
        # lr = sm.OLS(self.y_adjustment_obs[:, 0], sm.add_constant(self.y_adjustment_obs[:, 1])).fit()
        # conf_interval = lr.conf_int(alpha)
        # ci_p = self.y_adjustment_obs[:, 1] * conf_interval[1,0] + conf_interval[0,0]
        # ci_m = self.y_adjustment_obs[:, 1] * conf_interval[1,1] + conf_interval[0,1]
        # l, = ax.plot(self.y_adjustment_obs[:, 1], ci_p, color="red", alpha=0.4, label="CI")
        # ax.plot(self.y_adjustment_obs[:, 1], ci_m, color="red", alpha=0.4)

        # ax.fill_between(self.y_adjustment_obs[:, 1], ci_m, ci_p, alpha=0.1, color="blue")

        ax.scatter(x=self.y_adjustment_obs[:, 1], y=self.y_adjustment_obs[:, 0], label="Power in zenith")
        xs = self.y_adjustment_obs[:, 1]
        ys = self._y_adjustment_reg_.predict(self.y_adjustment_obs[:, 1].reshape(-1,1))
        xs, ys = zip(*sorted(zip(xs, ys)))
        ax.plot(xs,ys, c="red", label="Y-adjustment")

        x_max = max(self.y_adjustment_obs[:, 1])
        y_max = max(self.y_adjustment_obs[:, 0])
        #
        # pred = self._y_adjustment_reg_.predict(self.y_adjustment_obs[:,1:])
        # r2 = r2_score(self.y_adjustment_obs[:,0], pred)
        # ax.text(0.4*x_max, 0.8*y_max, f"y={self._y_adjustment_coef_:.2f}x + { self._y_adjustment_intercept_:0.2f}")
        # ax.text(0.2*x_max, 0.7*y_max, f"$R^2 = {r2:.2f}$")
        # ax.text(0.4*x_max, 0.7*y_max, f"$CI = {1-alpha}%$")

        ax.set_xlabel("Solar elevation angle")
        ax.set_ylabel("Normalized power at zenith")
        ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
        ax.legend()

        return fig, ax

    def plot(self, plots=["overlay", "model", "y_adjustment"], prefix=""):
        plots = [p.lower() for p in plots]
        fig, ax = plt.subplots(4)
        if prefix != "":
            fig.suptitle(f"{prefix}:{self.model_name}")
        else:
            fig.suptitle(f"{self.model_name}")
        ov = self.overlay_.overlay

        Plotter.plot_overlay(ov, fig=fig, ax=ax[0])
        x = list(range(ov.shape[1]))
        ax[0].plot(x, self.model_representation_, color="r")

        # compute mean values
        # mean = np.apply_along_axis(lambda a: np.nanmean(), 0, self.overlay_)
        mean = np.nanmean(ov, axis=0)
        mx = np.nanmax(ov, axis=0)
        mi = np.nanmin(ov, axis=0)

        if plots is None or "model" in plots:
            ax[0].plot(x, mean, color="orange")
            ax[0].plot(x, mx, color="orange")
            ax[0].plot(x, mi, color="orange")

            ax[1].imshow(self.overlay_.heatmap, cmap='Reds', origin='lower')
            ax[2].imshow(self.overlay_.kde, cmap='Blues', origin='lower')

        # Plotter.plot_2D_histograms(self.overlay_.heatmap, self.overlay_.kde)
        # if plots is None or "overlay" in plots:
        #     self.overlay_.plot()

        if plots is None or "y_adjustment" in plots:
            self.plot_y_adjustment(fig=fig, ax=ax[3])
            # ax[3].scatter(x=self.y_adjustment_obs[:,1], y=self.y_adjustment_obs[:,0])
            # ax[3].plot(self.y_adjustment_obs[:,1], self._y_adjustment_reg_.predict(self.y_adjustment_obs[:, 1].reshape(-1,1)), c="orange")
        return fig, ax

    def _predict(self, fh, X):
        ts = np.array([i*self.x_time_delta_ + self.cutoff for i in fh]).flatten()
        # function uses delegated stuff (common for normal (n-step-ahead) prediction and in sample prediction)
        return self.in_sample_predict(X,ts)

    def _predict_step(self, elevation):
        return Optimized.model_assign(self.model_representation_,
                                      self.elevation_bins_,
                                      elevation,
                                      ret_bins=True,
                                      interpolation=self.interpolation)

    def in_sample_predict(self, X, ts):
        """
        This is a special predict which ommits cutoff supported by sktime. The function consumes timestamps instead of fh
        like _predict function
        """
        # print(timestamps)
        # if sequence expected then return 2D array that contains prediction for each step.

        # iterative approach - make trajectory forecasting
        timestamps = (ts.astype(int) // 10 ** 9).astype(int)
        elevation = Solar.elevation(Optimized.from_timestamps(timestamps), self.latitude_degrees,
                                    self.longitude_degrees) * 180 / np.pi

        data, bins = self._predict_step(elevation)

        debug_datas = pd.DataFrame(data={"Bins": bins, "Elevation": elevation}, index=pd.DatetimeIndex(ts), dtype=float)
        debug_datas.name = "Debug Data"
        debug_datas.index.name = "Debug"
        self.debug_data_ = debug_datas

        pred = pd.Series(index=pd.DatetimeIndex(ts), data=data, dtype=float)

        pred.name = "Prediction"
        pred.index.name = "Datetime"
        return pred


    # def score(self, X, y):
    #     # poor values - function crated only for api consistence
    #     #
    #     # X, y = check_X_y(X, y)
    #     pred = self.predict(X,y)
    #     return r2_score(pred, y)
    #     # return 0.6

    def generate_params_info_string(self):
        params = self.get_params()
        for key in params:
            b_min, b_max, b_integral = 0, 0, False

            if isinstance(params[key], float):
                b_min, b_max, b_integral = None, None, False
            if isinstance(params[key], int):
                b_min, b_max, b_integral = None, None, True
            if isinstance(params[key], bool):
                b_min, b_max, b_integral = 0, 1, True

            params[key] = (params[key], b_min, b_max, b_integral)
        return params

    def get_params_info(self):
        # return only those parameters that can be fitted during optimisation / grid search
        params = {
            'bandwidth': (0.4,  0, 1, False),
            'density_filter_modifier': (0, -1, 1, True),
            # 'enable_debug_params': (False, 0, 1, True),
            # 'interpolation': (False, 0, 1, True),
            # 'latitude_degrees': (51, 0, 180, True),
            # 'longitude_degrees': (14, 0, 180, True),
            # 'return_sequences': (False, 0, 1, True),
            'window_size': (None, 0, 1000, True),
            'x_bins': (90, 5, 250, True),
            'y_bins': (90, 5, 250, True),
            'zeros_filter_modifier': (0, -1, 1, True)
        }

        return params


    def __str__(self):
        # return "Model representation: " + str(self.model_representation_) + \
        #     " len(" + str(len(self.model_representation_)) + ")" + \
        #     "\nBins: " + str(self.elevation_bins_) + " len(" + str(len(self.elevation_bins_)) + ")"
        limit = self.str_limit
        s = self.model_name
        if self.str_repr_enable_params:
            s = s + " (" + str(self.get_params()) + ")"
        if len(s) > limit and limit != -1:
            return s[0:limit//2] + "..." + s[-limit//2:]
        else:
            return s