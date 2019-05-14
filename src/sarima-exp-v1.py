# grid search sarima hyperparameters for daily female dataset
import pandas as pd
import numpy as np

import joblib
import multiprocessing as mp


from math import sqrt
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(start_params=40, disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history),len(history))
    for i in ra
    return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
"""
def measure_mape(val_1_data, n_test):
    abs_err = np.divide(np.abs(np.subtract(val_1_data[:24], val_1_data[-n_test:])),val_1_data[-n_test:])
    ape = np.multiply(abs_err, np.full(shape=abs_err.shape, fill_value=100))
    shift_adjusted_mape = np.mean(ape, axis=0)
    print("MAPE with adjusted time series: %s" % np.array2string(shift_adjusted_mape))
"""
# split a univariate dataset into train/test sets
def train_test_split(val_1_data, n_test):
	return val_1_data[:-n_test], val_1_data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(val_1_data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(val_1_data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
  #  error2 = measure_mape(test, n_test)
    return error


# score a model, return None on failure
def score_model(val_1_data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(val_1_data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        # never show warnings when grid searching, too noisy
        with catch_warnings():
            filterwarnings("ignore")
            result = walk_forward_validation(val_1_data, n_test, cfg)
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(val_1_data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = joblib.Parallel(n_jobs=mp.cpu_count(), backend='multiprocessing')
        print("executer = ", executor)
        tasks = (delayed(score_model)(val_1_data, n_test, cfg) for cfg in cfg_list)
        print("tasks = ", tasks)
        scores = executor(tasks)
        print("Scores in if clause = ", scores)
    else:
        scores = [score_model(val_1_data, n_test, cfg) for cfg in cfg_list]
        print("Scores in else case = ", scores)

    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models


if __name__ == '__main__':
    # load dataset

    test_data = pd.read_csv("data/test.csv", delimiter=";", decimal=",")
    test_data.head(10).append(test_data.tail(10))

    val_1_data = test_data["Val_1"]
    val_2_data = test_data["Val_2"]
    val_3_data = test_data["Val_3"]

    print("Values shape = " ,val_1_data.shape)
    # data split
    print("***********************************")
    print("data values : ", val_1_data.values)
    print("***********************************")
    n_test = 44
    # model configs
    cfg_list = sarima_configs()
    print(cfg_list)
    # grid search
    scores = grid_search(val_1_data, cfg_list, n_test)
    print('done')
    # list top 3 configs
    print('scoresasd', scores)
    for cfg, error in scores[:3]:
        print("hello")
        print(cfg, error)
