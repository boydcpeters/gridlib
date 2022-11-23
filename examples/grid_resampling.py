import gridlib
import gridlib.io
import gridlib.plot

# Load the data
data = gridlib.io.read_data_survival_function("examples/data/example1.csv")

# Set the parameters
parameters = {
    "k_min": 10 ** (-3),
    "k_max": 10**1,
    "N": 200,
    "scale": "log",
    "reg_weight": 0.01,
    "fit_a": True,
    "a_fixed": None,
}

# Perform the resampling
fit_results_all, fit_results_resampled = gridlib.resampling_grid(
    parameters, data, n=10, perc=0.8
)

print(len(fit_results_all), len(fit_results_resampled))
