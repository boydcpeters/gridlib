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
) -> Tuple[
    Dict[str, Union[np.ndarray, float]], List[Dict[str, Union[np.ndarray, float]]]
]:
    """_summary_

    Parameters
    ----------
    parameters : _type_
        _description_
    data : _type_
        _description_
    n : int, optional
        Number of times the data should be resampled and should be fitted with GRID, by default 10.
    perc : float, optional
        The percentage of the data that should be resampled, by default 0.8
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A seed to initialize the BitGenerator. If None, then fresh, unpredictable
        entropy will be pulled from the OS. If an int or array_like[ints] is passed,
        then it will be passed to SeedSequence to derive the initial BitGenerator state.
        One may also pass in a SeedSequence instance. Additionally, when passed a
        BitGenerator, it will be wrapped by Generator. (THIS IS FROM NUMPY DOC, MAYBE CHANGE THIS)
    Returns
    -------
    _type_
        _description_
    """

    print("----------------------------------")
    print("Fitting of all data starts now...")
    # Determine the full GRID fit results based on all the data.
    fit_result_full = fit.fit_grid(parameters, data, disp=False)
    print("Fitting of all data finished now.")
    print("----------------------------------")
    print("Start resampling...")

    max_workers = 1
    if multiprocess_flag:
        # The number of logical cores minus 1, for the background tasks
        max_workers = psutil.cpu_count(logical=False) - 1

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
