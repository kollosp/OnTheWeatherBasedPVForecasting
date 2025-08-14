import os
from typing import List

import numpy as np
import pandas as pd
import scipy

class StatisticalAnalysis:
    """
        Class performs statistical analysis of results obtained by SlidingWindowExperimentsBase-based experiments classes
    """
    def __init__(self, concat_files_directories:List[str], **kwargs):
        self.modelCnf = "modelCnf"
        self._prediction_path_column = kwargs.pop('prediction_path_column', "prediction_path")
        self._model_name_column = kwargs.pop('model_name_column', "_model_name")
        self._model_dimensions_column = kwargs.pop('model_dimensions_column', "_model_dimensions")
        self._ref_model_dimensions_column = kwargs.pop('ref_model_dimensions_column', "_ref_model_dimensions")

        self._model_column = kwargs.pop('model_column', "model") # transformed model configuration column with ordered dimensions
        self._ref_model_column = kwargs.pop('ref_model_column', "ref_model") # transformed ref model column with order dimensions

        self._tested_dimensions = kwargs.pop('tested_dimensions', [])
        self._concat_df = self.load_concat_files(self.find_concat_files_in_directories(concat_files_directories))
        self._group_by_columns = [] # a list of columns used to group results.
        self.process_df()

    @property
    def concat_df(self) -> pd.DataFrame:
        return self._concat_df

    @staticmethod
    def find_concat_files_in_directories(concat_files_directories:List[str]):
        """
        function finds all 'concat' files located in given directory. Concat files should be created
        from SlidingWindowExperimentsBase metric_df results (may be concatenated)
        """
        r = []
        for concat_files_directory in concat_files_directories:
            for file in os.listdir(concat_files_directory):
                if "concat" in file:
                    r.append(os.path.join(concat_files_directory, file))
        return r


    def load_concat_files(self, files:List[str]):
        """
        Function loads all files and concatenates them into a single dataframe. After the concatenation is done, function
        checks types of columns that contains 'Mean' or 'Std' substrings in their names. Those columns must be floats.
        They represent metrics summary.
        """
        df = None
        for file in files:
            read = pd.read_csv(file, delimiter=",", header=0, decimal=".")
            read["concat_file_path"] = file

            metrics = self._get_metric_columns(read)
            for column in metrics:
                read.loc[:, column] = pd.to_numeric(read.loc[:, column], errors='coerce')
                # cast Mean and std columns to float. If experiment returned exception those column
                # will contain its textual representation. Those values will be replaced with nans
                read = read[~read.loc[:, column].isna()] # remove NaN column
            read[self._prediction_path_column] = os.path.dirname(file) + os.sep + \
                                                 read[self._prediction_path_column].str.split("/").str[-1]

            if df is None:
                df = read
            else:
                df = pd.concat([df, read])
        return df

    def get_modelCnf_model_name_conversion_dict(self):
        """
        Function returns a dictionary that can be used to map modelCnf with model name e.g.:
        {
        'KNN(10)_0[y,Declination,Elevation,SolarDay%]': 'KNN(10)_0[Declination,Elevation,SolarDay%,y]',
         'LR(12)_0[y,Declination,Elevation,SolarDay%]': 'LR(12)_0[Declination,Elevation,SolarDay%,y]', ...
        }
        """
        return self.concat_df.set_index(self.modelCnf)[self._model_column].to_dict()

    def get_group_concat_report(self):
        df = self._concat_df.groupby(self._group_by_columns)
        groups = None# pd.DataFrame(columns=self._group_by_columns + ["df"])
        # print("Total len of dataframe", len(self._concat_df))
        for i, groupKey in enumerate(df.groups.keys()):
            # ind = df.groups[groupKey]
            # group_dict = self._concat_df.iloc[ind]#.drop(columns=self._group_by_columns)

            ind_arrays = [self._concat_df[c] == key for c,key in zip(self._group_by_columns, groupKey)]
            ind = ind_arrays[0]
            for i in range(1, len(ind_arrays)):
                ind = ind & ind_arrays[i]
            group_ind = self._concat_df[ind]#.drop(columns=self._group_by_columns)

            print(f"group {i} len of ind: {len(df.groups[groupKey])}. len of group: {len(group_ind)}, keys: {groupKey}")
            # print("Making stats for group from dict:", {c: pd.unique(group_dict[c]) for c in self._group_by_columns})
            print("Making stats for group from inds:", {c: pd.unique(group_ind[c]) for c in self._group_by_columns})
            # print(ind)
            new_data = pd.DataFrame({
                **{k:[v] for k,v in zip(self._group_by_columns, groupKey)},
                "df": [group_ind],
            }, index=[i])

            if group_ind is not None:
                groups = pd.concat([groups, new_data])
            else:
                groups = new_data
        print("groups", groups)
        return groups

    def get_predictions_directory(self, df=None):
        """
        Function returns dataframe where predictions numpy arrays are stored. Dataframe is indexed by model and metric
        """
        if df is None:
            raise ValueError("get_model_improvement_ratio_df: df must be provided (usually concat_df)")
        predictions_to_load = df[self._prediction_path_column]
        predictions_to_load = np.unique(predictions_to_load)
        dataframes = []
        model_name_conversion_dict = self.get_modelCnf_model_name_conversion_dict()

        # load all necessary prediction files to json
        for prediction in predictions_to_load:
            df = pd.read_csv(prediction, delimiter=",", header=0, decimal=".")
            # get only prediction error metrics
            metric_columns = [c for c in df.columns if ":" in c and not "FT" in c] # metrics are marked as <model>:<metric>
            if "Unnamed: 0" in metric_columns:
                metric_columns.remove("Unnamed: 0")
            df = df[metric_columns]
            df.dropna(inplace=True) # drop empty rows (not all rows contains metrics -- only one per day)
            #split  <model>:<metric> into multiindex
            mi = [tuple(c.split(":")) for c in df.columns]
            mi = [(model_name_conversion_dict[c[0]], c[1]) for c in mi] #change modelCnf to model
            df.columns = pd.MultiIndex.from_tuples(mi)
            for model, metric in mi:
                dataframes.append({
                    self._model_column: model,
                    "metric": metric,
                    "data": df.loc[:, (model, metric)].to_numpy()
                })

        m = pd.DataFrame(dataframes)
        m.set_index(["model", "metric"], inplace=True)
        m.sort_index(inplace=True)
        return m

    def get_ttest_df(self, df=None, model_join_df=None, alpha=0.05):
        """
            perform ttest
            alpha: reject H0 if pvalue < alpha
        """
        #check arguments
        if model_join_df is None:
            raise ValueError("get_model_improvement_ratio_df: model_join_df must be provided")
        if df is None:
            raise ValueError("get_model_improvement_ratio_df: df must be provided (usually concat_df)")

        #select primary key column
        model_join_df.reset_index(inplace=True)
        model_join_df = model_join_df.loc[:, ["model", "ref_model", "metric"]]
        model_join_df.set_index([self._model_column, self._ref_model_column, "metric"], inplace=True)
        model_join_df.sort_index(inplace=True)
        #get predictions lists
        predictions_df = self.get_predictions_directory(df)


        #add columns
        stat_columns = ["p", "statistic", "rejectH0"]
        for c in stat_columns:
            model_join_df.loc[:, f"{c}"] = 0.0
        model_join_df.sort_index(inplace=True)

        #iterate and perform function
        i,n= 0, len(model_join_df)
        for index, row in model_join_df.iterrows():
            i += 1
            model = index[0]
            ref_model = index[1]
            metric = index[2]
            model_predictions = predictions_df.loc[(model, metric), "data"]

            ref_model_predictions = predictions_df.loc[(ref_model, metric), "data"]
            # print("#", model_predictions[[0,-1]], ref_model_predictions[[0,-1]], "?")
            # print(model, ref_model, "\n", row.index)
            ttest = scipy.stats.ttest_ind(model_predictions, ref_model_predictions, equal_var=False)
            rejH0 = int(ttest.pvalue < alpha)
            model_join_df.loc[index, "p"] = ttest.pvalue
            model_join_df.loc[index, "statistic"] = ttest.statistic
            model_join_df.loc[index, "rejectH0"] = rejH0
            if i % 1000 == 0:
                print(f"{i} / {n}: {model} vs. {ref_model} -> rejectH0: {rejH0}")

        #finalize table
        model_join_df.columns = model_join_df.columns.droplevel(1) # multilevel (P, ""), (stat, ""), ...
        model_join_df.columns = [f"ttest_{c}" for c in model_join_df.columns]
        return model_join_df

    def get_model_improvement_ratio_df(self, df, model_join_df=None):
        """
        Function performs simple mean-based models' experimental results between models and their reference. It calculates
        improvement ratio IR: 100% * (<ref metric> - <model metric>) / <ref metric>. Positive IR means that model is
        better than its reference.
        metric: a filter for metric it is passed to get_results_df
        model_fiter_str: a filter for model name (not used yet)
        """
        # check arguments
        if model_join_df is None:
            raise ValueError("get_model_improvement_ratio_df: model_join_df must be provided")

        #initilize processing table.
        mean_df = self.get_results_df(df)
        mean_df.columns = mean_df.columns.levels[1]
        # mean_df.reset_index("instance", inplace=True)
        # mean_df = mean_df.unstack()

        if model_join_df is None:
            raise ValueError("get_model_improvement_ratio_df: model_join_df must be provided")
        model_join_df.reset_index(inplace=True)
        model_join_df = model_join_df[["model", "ref_model", "metric"]]

        model_join_df.set_index([self._model_column, self._ref_model_column, "metric"], inplace=True)
        stat_columns = ["IR"]
        for c in stat_columns:
            model_join_df[f"{c}"] = 0.0
        model_join_df.sort_index(inplace=True)
        i, n = 0, len(model_join_df)
        for index, row in model_join_df.iterrows():
            i += 1
            model = index[0]
            ref_model = index[1]
            metric = index[2]
            model_metric = mean_df.loc[(model, metric), "Mean"]
            ref_model_metric = mean_df.loc[(ref_model, metric), "Mean"]

            ir =  (ref_model_metric - model_metric) / ref_model_metric
            model_join_df.loc[index, "IR"] = ir
            if i % 1000 == 0:
                print(f"{i} / {n}: {model} vs. {ref_model} {metric} -> IR: {ir}")

        model_join_df.columns = ["_".join([x for x in list(c) if x != ""]) for c in
                            model_join_df.columns.to_flat_index()]  # make index flat
        return model_join_df

    def _get_available_metric_columns(self):
        return np.unique([x.split("_")[0] for x in self._get_metric_columns()])

    def _get_metric_columns(self, df=None):
        """
        Function return list of metrics available in dataset
        df: Optional dataframe. If None self.concat_df is used.
        """
        suffixes = ["Mean", "Std", "Samples"]
        if df is None:
            df = self.concat_df
        return [x for x in df.columns if any(suffix in x for suffix in suffixes) and not "FT" in x]

    def _get_model_cartesian_join(self, df:None):
        """
        function performs cartesian join of models (creates model_Ref) however does in respect of instance. Does not
        mix model between instance (dataset)
        """
        results_df = self.get_results_df(df)
        results_df.reset_index(inplace=True)
        results_df.sort_index(axis=1, inplace=True)
        m = pd.merge(results_df, results_df, how="cross", suffixes=("", "_ref")) # compare each pair of

        # filter for columns instance, model, ref model and metric -> those columns will became indexes of dataframe
        m=m[
            # (m["instance"] == m["instance_ref"]) & # filter if instance is consistent
            (m["model"] != m["model_ref"]) & # filter self comparison
            (m["metric"] == m["metric_ref"])  # filter different metric comparision
        ]

        # clear dataframe

        m.rename(columns={"model_ref": "ref_model"}, inplace=True)
        m.sort_index(axis=1, inplace=True)
        m.drop(columns=["metric_ref"], inplace=True)
        m.set_index(["model", "ref_model", "metric"], inplace=True)

        return m

    def get_report_ir_ttest_metric_df_v2(self):
        groups = self.get_group_concat_report()
        merged = pd.DataFrame()
        for index, value in groups.iterrows():
            concat_df = value["df"]
            print("Making stats for group:", {c:pd.unique(concat_df[c]) for c in self._group_by_columns})
            model_join = self._get_model_cartesian_join(concat_df).reset_index()

            ttest = self.get_ttest_df(concat_df, model_join).reset_index()
            ir = self.get_model_improvement_ratio_df(concat_df, model_join).reset_index()
            #add instance to dataframe (remove indexes, mergem add indexes)
            model_join.columns = ["_".join([x for x in list(c) if x != ""]) for c in
                                model_join.columns.to_flat_index()]  # make index flat

            merge_df = pd.merge(model_join, ttest, how="left",on=["model", "ref_model", "metric"])
            merge_df = pd.merge(merge_df, ir, how="left", on=["model", "ref_model", "metric"])
            group_columns = [x for x in value.index if x != "df"]
            for c in group_columns:
                merge_df[c] = value[c]
            merged = pd.concat([merged, merge_df])
        print("mergerd==========================================================\n",merged)
        return merged

    # def get_report_ir_ttest_metric_df(self):
    #     """
    #     Function computes improvement ratio and performs ttest over model and reference. Returns results as a dataframe
    #     """
    #     results_df = self.get_results_df()
    #     ir_df = self.get_model_improvement_ratio_df()
    #     ttest_df = self.get_ttest_df()
    #
    #     all_together = pd.merge(ttest_df, ir_df, on=["model", "ref_model", "metric"])
    #     all_together["ref_model"] = all_together.index.get_level_values("ref_model")
    #     # very bad way to copy indexing column to columns
    #     results_df["instance"] = results_df.reset_index()[["instance"]].to_numpy()
    #     all_together = pd.merge(all_together, results_df, on=["model", "metric"])
    #     all_together.reset_index(inplace=True)
    #     all_together.set_index(["model", "ref_model", "metric"], inplace=True)
    #     return all_together

    def get_results_df(self, df=None):
        """
        function returns view from concat_df. It contains real model name self.modelCnf, model name with ordered dimensions
        model and metric results. Metric are find by "Mean" or "Std" suffixes
        metric: selected metric to be included in table (element from self._get_available_metric_columns()) if metric_name is None, return all metrics
        """
        if df is None:
            df = self.concat_df

        metric_columns = self._get_metric_columns(df)

        ret = df[metric_columns + [self._model_column]]
        ret.set_index([self._model_column], inplace=True)
        ret.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in ret.columns])
        # if metric is not None:
        #     print(metric)
        #     ret = ret[metric]
        ret.columns = ret.columns.reorder_levels(order=[1, 0]) #reorder for unstack
        ret.columns = pd.MultiIndex.from_tuples([("MetricStat", *c) for c in ret.columns.to_list()], names=["", "", "metric"])

        ret = ret.stack(future_stack=True)

        return ret

    # def get_model_and_its_reference_df(self, tested_dimensions=None, default_reference_dims="y"):
    #     """
    #     Function transforms self.modelCnf from concat_df table. It finds reference model for each model that was evaluated
    #     during tests and is included in concat_df table. Function returns two-column table containing model and model_ref
    #     columns. The returned table can be used to join result tables.
    #
    #     tested_dimensions:
    #         list of tested dimensions that must be removed from Model column to find reference model.
    #     default_reference_dims:
    #         string containing default reference model dimensions. It is used when after dimensions removal dimension list
    #         is empty.
    #     """
    #     if tested_dimensions is None:
    #         tested_dimensions = self._tested_dimensions
    #     concat_df = self.concat_df[[self._model_column, "instance"]]
    #     concat_df[self._model_name_column] = concat_df[self._model_column].str.split("_").str[0]
    #     # in case of deep model after star "*" there are n and n_steps params:
    #     # LSTM_0[Elevation,SolarDay%]*n=1,step=28
    #     # KNN_0[Elevation,SolarDay%]
    #     self._concat_df[self._model_name_column + "postfix"] = self._concat_df[self.modelCnf].str.extract(
    #         "\](.*)$").astype(str).replace("nan", "")
    #
    #     # extract group inside brackets [...]
    #     concat_df[self._model_dimensions_column] = concat_df[self._model_column].str.extract("\[(.+)\]")
    #     # removes all tested_dimensions from
    #     concat_df[self._ref_model_dimensions_column] = concat_df[self._model_dimensions_column].apply(
    #         lambda x: ",".join([xx for xx in x.split(",") if not xx in tested_dimensions])
    #         # lambda x: type(x)
    #     )
    #     # in case of this experiment default ref model is that one which used "y" as dimension
    #     concat_df[self._ref_model_dimensions_column] = concat_df[self._ref_model_dimensions_column].replace("",default_reference_dims)
    #     concat_df[self._ref_model_column] = \
    #         concat_df[self._model_name_column] + "_" + \
    #         concat_df["instance"].astype(str) + "[" + concat_df[self._ref_model_dimensions_column] + "]" + \
    #         self._concat_df[self._model_name_column + "postfix"]
    #
    #     return concat_df[[self._model_column, self._ref_model_column]]

    def process_df(self):
        
        self._concat_df.rename(columns={"Unnamed: 0": self.modelCnf}, inplace=True)
        # self._concat_df[self._prediction_path_column] = self._concat_df[self._prediction_path_column].str.split("/").str[
        #     -1]
        self._concat_df[self._model_name_column] = self._concat_df[self.modelCnf].str.split("_").str[0]
        # in case of deep model after star "*" there are n and n_steps params:
        # LSTM_0[Elevation,SolarDay%]*n=1,step=28
        # KNN_0[Elevation,SolarDay%]
        self._concat_df[self._model_name_column + "postfix"] = self._concat_df[self.modelCnf].str.extract("\](.*)$").astype(str).replace("nan", "")
        # extract group inside brackets [...]
        self._concat_df[self._model_dimensions_column] = self._concat_df[self.modelCnf].str.extract("\[(.+)\]")
        # removes test instances which has no dimensions
        self._concat_df = self._concat_df[self._concat_df[self._model_dimensions_column].notna()]
        # sort dimensions alphabetically to avoid mixing and enabling string comparison

        self._concat_df[self._model_dimensions_column] = self._concat_df[self._model_dimensions_column].apply(
            lambda x: ",".join(sorted(x.split(",")))
        )
        self._concat_df[self._model_column] = \
            self._concat_df[self._model_name_column] + "_" + \
            self._concat_df["instance"].astype(str) + "[" + self._concat_df[self._model_dimensions_column] + "]" + \
            self._concat_df[self._model_name_column + "postfix"]

        self._concat_df.drop(columns=[self._model_dimensions_column, self._model_name_column + "postfix", 'n', 'n_steps', 'excluded_dims'],inplace=True)
        # find all columns that are needed for calculations. Other columns are used to gropu calculations
        metrics = [x for x in self._concat_df.columns if "Mean" in x or "Std" in x or "Samples" in x]
        config_columns = ['modelCnf', 'prediction_path', 'concat_file_path', '_model_name', 'model']
        self._group_by_columns = self._concat_df.columns.to_list()
        for i in metrics + config_columns:
            self._group_by_columns.remove(i)
