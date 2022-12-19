from typing import Tuple, Dict, Union, List
import math
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psutil
from tqdm import tqdm

from . import compute
from . import data_utils
from . import fit


def _resample_data(data, perc: float = 0.8):
    """
    Resample the data for a given dataset, with a given percentage of data points.

    Parameters
    ----------
    data : dict
        A dictionary containing survival time and value data. The dictionary should have keys
        corresponding to different time-to-event values, and values that are dictionaries themselves,
        containing "time" and "value" keys. The "time" and "value" keys should contain arrays of
        equal length with the corresponding data.
    perc : float, optional
        The percentage of data points to include in the resampled dataset. Defaults to 0.8.

    Returns
    -------
    dict
        A dictionary containing the resampled time and value data. The structure of the returned
        dictionary is the same as the input data dictionary, with the exception that only the
        specified percentage of data points will be included.
    """
    rng = np.random.default_rng()

    data_sampled = dict()
    for t_tl in data.keys():

        # Retrieve the time in seconds
        t_tl_s = data_utils.get_time_sec(t_tl)

        # The total number of data points for a time-lapse condition in the complete
        # dataset
        data_point_all = data[t_tl]["value"][0]

        # Determine how many data points should be in the resampled version
        # math.ceil returns an integer
        num_sample = math.ceil(data_point_all * perc)

        # Round the possible track lifes to the closest integer
        track_lifes = np.rint(data[t_tl]["time"] / t_tl_s)
        # Cast it to the integer dtype, since the compute survival functions needs
        # an array with dtype integer.
        track_lifes = track_lifes.astype(np.int64)

        # Calculate the probability for every track life
        # ----------------------------------------------
        # First retrieve the true values and invert these, so the counts go from
        # low to high
        temp = data[t_tl]["value"][::-1]

        # Calculate the difference between every track life step
        diff = temp[1:] - temp[:-1]  # check out np.diff?
        # invert the values back so they align again with the track lifes
        diff = diff[::-1]

        # Concatenate the last value, because it equals "temp[0] - 0 = temp[0]"
        values = np.concatenate(
            (diff, np.array([temp[0]]))
        )  # add the last element again
        p = values / np.sum(values)
        # THE NEXT LINE IS WRONG!! THIS IS THE CUMULATIVE SUM, NOT THE INDIVIDUAL COUNTS
        # p = data[t_tl]["value"] / np.sum(data[t_tl]["value"])

        track_lifes_sampled = rng.choice(
            track_lifes,
            size=num_sample,
            replace=True,
            p=p,
        )

        time, value = compute.compute_survival_function(track_lifes_sampled, t_tl_s)

        # Save the results to the appropriate keys
        data_sampled[t_tl] = dict()
        data_sampled[t_tl]["time"] = time
        data_sampled[t_tl]["value"] = value

    # Returns non-normalized survival function
    return data_sampled


def _resample_and_fit(parameters, data, perc: float = 0.8):
    """Function that can be easily run with multiprocessing."""
    data_resampled = _resample_data(data, perc=perc)

    fit_results_temp = fit.fit_grid(parameters, data_resampled, disp=False)

    return fit_results_temp


def resampling_grid(
    parameters,
    data,
    n: int = 10,
    perc: float = 0.8,
    multiprocess_flag: bool = True,
    max_workers: int = None,
) -> Tuple[
    Dict[str, Union[np.ndarray, float]], List[Dict[str, Union[np.ndarray, float]]]
]:
    """
    Fit survival time and value data with a given set of parameters, using a resampling procedure.

    This function performs a resampling procedure to fit survival time and value data with a given set
    of parameters. It first fits the full dataset with the parameters, then resamples the data and fits
    the resampled datasets with the parameters. The function returns the fit results for the full dataset
    and the resampled datasets.

    Parameters
    ----------
    parameters : dict
        A dictionary containing the parameters to use for fitting the data. UPDATE THIS TEXT
    data : dict
        A dictionary containing survival time and value data. The dictionary should have keys corresponding to
        different time-to-event values, and values that are dictionaries themselves, containing "time" and
        "value" keys. The "time" and "value" keys should contain arrays of equal length with the
        corresponding data.
    n : int, optional
        The number of resampled datasets to create. Defaults to 10.
    perc : float, optional
        The percentage of data points to include in each resampled dataset. Defaults to 0.8.
    multiprocess_flag : bool, optional
        A flag indicating whether to use parallel processing when fitting the resampled datasets.
        Defaults to True.
    max_workers: int, optional
        The maximum number of logical cores to use for the multiprocessing. This number
        is only used if multiprocess_flag is set to True and the value has to be between
        1 and the maximum # logical cores - 1. If value is set to None, than it will
        be set to # logical cores - 1. Default is None.

    Returns
    -------
    tuple
        A tuple containing the fit results for the full dataset, and a list of fit results for the
        resampled datasets. The fit results are dictionaries themselves, with keys corresponding to
        the names of the fitted parameters, and values that are either arrays or floats, depending on the
        parameter.
    """
    print("----------------------------------")
    print("Fitting of all data starts now...")
    # Determine the full GRID fit results based on all the data.
    fit_result_full = fit.fit_grid(parameters, data, disp=False)
    print("Fitting of all data finished now.")
    print("----------------------------------")
    print("Start resampling...")

    if multiprocess_flag:
        # The number of logical cores minus 1, for the background tasks
        max_workers_limit = psutil.cpu_count(logical=False) - 1

        if max_workers is None:
            max_workers = max_workers_limit
        elif max_workers < 1:
            raise ValueError("max_workers should be >=1")
        elif max_workers > max_workers_limit:
            max_workers = max_workers_limit
            print(
                f"WARNING: max_workers is set to {max_workers}, since initial value",
                "was set higher than the allowed number of workers.",
            )

    # Determine the GRID fit results for the resampled data
    fit_results_resampled = []

    with tqdm(total=n) as pbar:

        t0 = time.time()

        data_resampled = _resample_data(data, perc=perc)

        fit_results_temp = fit.fit_grid(parameters, data_resampled, disp=False)

        fit_results_resampled.append(fit_results_temp)

        t1 = time.time()

        print(f"Estimated time: {((t1-t0)*n)/60.0/max_workers:.0f} min")

        pbar.update(1)
        print()
        print(f"1-th resampling is finished.")

        if multiprocess_flag:
            i = 2
            with ProcessPoolExecutor(max_workers=max_workers) as executor:

                # Submit the tasks
                futures = [
                    executor.submit(_resample_and_fit, parameters, data, perc)
                    for _ in range(n - 1)
                ]

                # Check if a task is finished
                for future in concurrent.futures.as_completed(futures):
                    fit_results_temp = future.result()
                    fit_results_resampled.append(fit_results_temp)
                    pbar.update(1)
                    print()
                    print(f"{i}-th resampling is finished.")
                    i += 1
        else:
            for i in range(2, n + 1):
                fit_results_temp = _resample_and_fit(parameters, data, perc=perc)
                fit_results_resampled.append(fit_results_temp)
                pbar.update(1)
                print()
                print(f"{i}-th resampling is finished.")

    print("Resampling is completely finished.")
    print("----------------------------------")

    return fit_result_full, fit_results_resampled
