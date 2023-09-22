# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel
import openturns as ot

def bayesianoptimization(X, y, candidates_of_X, acquisition_function_flag, cumulative_variance=None):

    X = np.array(X)
    y = np.array(y)
    if cumulative_variance is None:
        cumulative_variance = np.empty(len(y))
    else:
        cumulative_variance = np.array(cumulative_variance)

    relaxation_value = 0.01
    delta = 10 ** -6
    alpha = np.log(2 / delta)

    autoscaled_X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1))
    autoscaled_candidates_of_X = (candidates_of_X - X.mean(axis=0)) / (X.std(axis=0, ddof=1))
    autoscaled_y = (y - y.mean(axis=0)) / (y.std(axis=0, ddof=1))
    
    
    gaussian_process_model = GaussianProcessRegressor(n_restarts_optimizer=50)
    gaussian_process_model.fit(autoscaled_X, autoscaled_y)
    autoscaled_estimated_y_test, autoscaled_std_of_estimated_y_test = gaussian_process_model.predict(
        autoscaled_candidates_of_X, return_std=True)


    if acquisition_function_flag == 1:
        acquisition_function_values = autoscaled_estimated_y_test + alpha ** 0.5 * (
                (autoscaled_std_of_estimated_y_test ** 2 + cumulative_variance) ** 0.5 - cumulative_variance ** 0.5)
        cumulative_variance = cumulative_variance + autoscaled_std_of_estimated_y_test ** 2
    elif acquisition_function_flag == 2:
        acquisition_function_values = (autoscaled_estimated_y_test 
                                       - max(autoscaled_y)
                                       - relaxation_value) * \
                                      norm.cdf((autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) /
                                               (autoscaled_std_of_estimated_y_test+0.000001)) + \
                                      autoscaled_std_of_estimated_y_test * \
                                      norm.pdf((autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) /
                                               (autoscaled_std_of_estimated_y_test+0.000001))
    elif acquisition_function_flag == 3:
        acquisition_function_values = norm.cdf(
            (autoscaled_estimated_y_test - max(autoscaled_y) - relaxation_value) / autoscaled_std_of_estimated_y_test)
    elif acquisition_function_flag == 0:
        acquisition_function_values = autoscaled_estimated_y_test

    
    selected_candidate_number = np.where(acquisition_function_values == np.max(acquisition_function_values))[0][0]
    selected_X_candidate = candidates_of_X[selected_candidate_number, :]

    return selected_candidate_number, selected_X_candidate, cumulative_variance




