import gridlib
import gridlib.io
import gridlib.plot

# Load the data
data = gridlib.io.read_data_survival_function("examples/data/example1.csv")

#######################################################################################
# Option A: fit the data to GRID and multi-exponential seperately

# Set the parameters
parameters_grid = {
    "k_min": 10 ** (-3),
    "k_max": 10**1,
    "N": 200,
    "scale": "log",
    "reg_weight": 0.01,
    "fit_a": True,
    "a_fixed": None,
}

# Perform GRID fitting
fit_results_grid = gridlib.fit_grid(parameters_grid, data)


# Perform multi-exponential fitting
# Set the parameters
parameters_multi_exp = {
    "n_exp": [1, 2, 3],  # fit a 1-, 2- and 3- exponential
    "k_min": 10 ** (-3),
    "k_max": 10**1,
    "fit_a": True,
    "a_fixed": None,
}

fit_results_multi_exp = gridlib.fit_multi_exp(parameters_multi_exp, data)


#######################################################################################
# Option B: fit the data to GRID and multi-exponential with one function call.
# Set the parameters
parameters_all = {
    "k_min": 10 ** (-3),
    "k_max": 10**1,
    "N": 200,
    "scale": "log",
    "reg_weight": 0.01,
    "fit_a": True,
    "a_fixed": None,
    "n_exp": [1, 2, 3],  # fit a 1-, 2- and 3- exponential
}

fit_results_grid, fit_results_multi_exp = gridlib.fit_all(parameters_all, data)
