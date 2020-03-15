import yaml
import pandas as pd
import numpy as np
import pickledb
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from time import time, strftime, gmtime
from pyramid.arima import auto_arima
from math import fabs
from sys import argv


class ARIMAMultiStepForecaster:
    def __init__(self, config_filename):
        self.config = yaml.load(open(config_filename))
        self.data = pd.read_csv(self.config["dataset_path"], header=0, index_col=0)
        self.db = pickledb.load("db/index.db", False)
        self.endog_train = None
        self.endog_test = None
        self.exog_train = None
        self.exog_test = None
        self.predict_next = None
        self.predictions = []
        self.apes = []

        self.endogenous_variables = self.config.get("endogenous_variables", None)
        self.exogenous_variables = self.config.get("exogenous_variables", None)
        self.train_data_percentage = self.config.get("train_data_percentage", None)
        self.predict_next = self.config.get("predict_next", None)
        self.figure_path = self.config.get("figure_path", None)

        if self.endogenous_variables is None:
            raise Exception("endogenous_variables is not specified in config. You need to add column names which will be predicted!")

        if self.train_data_percentage is None:
            raise Exception("train_data_percentage is not specified in config. You need to enter percentage of training data!")

        if self.predict_next is None:
            raise Exception("predict_next is not specified in config. You need to step number will be predicted!")

        self.endog_train, self.endog_test = self.__split_data(self.__select(self.data, self.endogenous_variables), self.train_data_percentage)

        if not (self.exogenous_variables is None):
            self.exog_train, self.exog_test = self.__split_data(self.__select(self.data, self.exogenous_variables), self.train_data_percentage)

        self.endog_train_arr = [endog_obs for endog_obs in self.endog_train]

    def __get_model_name(self):
        model_name = "ARIMA"

        if not (self.config.get("seasonal", None) is None):
            model_name = "S" + model_name

        if not (self.exog_train is None):
            model_name = model_name + "X"

        return model_name

    def __split_data(self, data, percent):
        train_data_count = int((len(data) * percent) / 100)
        return data[:train_data_count], data[train_data_count:]

    def __select(self, data, columns):
        return data[columns].values

    def __save_plot(self, datas, path, plot_colors, title=None):
        fig = plt.figure(figsize=(10, 6))
        plt.title(title)
        for (data, plot_color) in zip(datas, plot_colors):
            plt.plot(data, color=plot_color)
        fig.savefig(path)
        fig.clear()

    def __train_common_phase(self, endog_test_obs, predicted_obs):
        ape = (fabs(endog_test_obs - predicted_obs) / endog_test_obs) * 100
        self.predictions.append(predicted_obs)
        self.endog_train_arr.append(predicted_obs)
        self.apes.append(ape)

        print("Prediction=%.3f, Expected=%.3f, APE=%.3f, MAPE=%.3f" % (predicted_obs, endog_test_obs, ape, np.mean(np.array(self.apes))))

    def __train_with_exog_multi_step(self, model):
        for (endog_test_obs, exog_test_obs) in zip(self.endog_test[:self.predict_next], self.exog_test[:self.predict_next]):
            predicted_obs = np.round(model.predict(n_periods=1, exogenous=exog_test_obs.reshape(1, -1)), 2)

            self.__train_common_phase(endog_test_obs, predicted_obs)

            self.exog_train = np.append(self.exog_train, exog_test_obs.reshape(1, -1), axis=0)

            # Re-fitting the model with new observations
            model.fit(self.endog_train_arr, exogenous=self.exog_train)

    def __train_multi_step(self, model):
        for endog_test_obs in self.endog_test[:self.predict_next]:
            predicted_obs = np.round(model.predict(n_periods=1), 2)

            self.__train_common_phase(endog_test_obs, predicted_obs)

            # Re-fitting the model with new observations
            model.fit(self.endog_train_arr)

    def run(self):
        begin = time()

        print("Training with %s model..." % self.__get_model_name())

        # Estimating parameters
        model = auto_arima(self.endog_train_arr,
                           seasonal=self.config.get("seasonal", False),
                           m=self.config.get("m", 1),
                           error_action="ignore",
                           exogenous=self.exog_train,
                           maxiter=self.config.get("max_iter", 100),
                           suppress_warnings=True)

        if not (self.exog_test is None):
            self.__train_with_exog_multi_step(model)
        else:
            self.__train_multi_step(model)

        end = time()

        print("Parameters: %s" % str(self.config))
        print("Elapsed time: %s" % strftime("%H:%M:%S", gmtime(end - begin)))

        mape = np.mean(np.array(self.apes))
        print("MAPE: %.3f" % mape)

        # Get last experiment index of running algorithm
        model_index = int(self.db.get(self.__get_model_name().lower())) + 1

        self.__save_plot(
            [self.predictions, self.endog_test[:self.predict_next]],
            "%s/result_%d.png" % (self.figure_path, model_index),
            ["red", "blue"],
            "Training Data Percentage: %.2f, MAPE Score: %.3f, Exogenous Variables: %s" % (self.train_data_percentage, mape, self.exogenous_variables))

        # Update current model's experiment index
        self.db.set(self.__get_model_name().lower(), model_index)
        self.db.dump()


if __name__ == "__main__":
    ARIMAMultiStepForecaster(argv[1]).run()