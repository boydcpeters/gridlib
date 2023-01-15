import multiprocessing as mp
import concurrent.futures
from itertools import chain
import signal
import math
import time
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
import psutil
from tqdm import tqdm

from . import compute, data_utils, fit


def _resample_data(
    data: Dict[str, Dict[str, np.ndarray]], perc: float = 0.8
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Function resamples from the provided dataset with a given percentage of data points.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the survival functions. For example::

            {
                "0.05s": {
                    "time": array([0.05, 0.1, 0.15, ...]),
                    "value": array([1.000e+04, 8.464e+03, 7.396e+03, ...]),
                },
                "1s": {
                    "time": array([1., 2., 3., 4., ...]),
                    "value": array([1.000e+04, 6.925e+03, 5.541e+03, 4.756e+03, ...]),
                },
            }

    perc : float, optional
        The percentage of data points to include in the resampled dataset. Defaults to
        0.8.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the resampled survival functions. The structure is the same
        as the structure of the input data, just with resampled data values.
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


# TODO: update docstring
def _resample_data_and_fit_worker(
    fn: Callable[
        [Dict, Dict[str, Dict[str, np.ndarray]], bool],
        Dict[str, Dict[str, Union[np.ndarray, float]]],
    ],
    parameters,
    data,
    perc: float = 0.8,
):
    """
    Function resamples from the provided dataset with a given percentage of data points
    and then perform GRID and/or multi-exponential fitting on the resampled data and
    returns the results.

    Parameters
    ----------
    fn: function

    parameters: Dict

    data : Dict[str, Dict[str, np.ndarray]]
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the survival functions. For example::

            {
                "0.05s": {
                    "time": array([0.05, 0.1, 0.15, ...]),
                    "value": array([1.000e+04, 8.464e+03, 7.396e+03, ...]),
                },
                "1s": {
                    "time": array([1., 2., 3., 4., ...]),
                    "value": array([1.000e+04, 6.925e+03, 5.541e+03, 4.756e+03, ...]),
                },
            }

    perc : float, optional
        The percentage of data points to include in the resampled dataset. Defaults to
        0.8.

    Returns
    -------
    fit_results_temp: Dict[str, Dict[str, Union[np.ndarray, float]]]
        A dictionary mapping keys (fitting procedure) to the corresponding fit results.
        For example::

            {
                "grid": {
                    "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                    "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                    "a": 0.010564217803906671,
                    "loss": 0.004705659331508584,
                },
                "1-exp": {
                    "k": array([0.02563639]),
                    "s": array([1.]),
                    "a": 0.08514936433699753,
                    "loss": 1.2825570522448484
                },
                "2-exp": {
                    "k": array([0.03715506, 1.7248619]),
                    "s": array([0.17296989, 0.82703011]),
                    "a": 0.011938572088673213,
                    "loss": 0.2868809590425386
                },
                "3-exp": {
                    "k": array([0.0137423 , 0.27889073, 3.6560956]),
                    "s": array([0.06850312, 0.23560175, 0.69589513]),
                    "a": 0.011125323730424764,
                    "loss": 0.0379697542735324
                },
            }

    """
    data_resampled = _resample_data(data, perc=perc)

    fit_results_temp = fn(parameters, data_resampled, disp=False)

    return fit_results_temp


# TODO: add docstring
def _resample_and_fit_sequential(fn, parameters, data, n, perc):

    t0 = time.time()

    # Then perform fitting on the full dataset
    # ----------------------------------------
    print("----------------------------------------")
    print("Fitting of all data starts now...")
    # Determine the full fit results based on all the data.
    t1 = time.time()
    fit_results_full = fn(parameters, data, disp=False)
    t2 = time.time()
    print("Fitting of all data finished now.")
    print("----------------------------------------")

    print(f"Estimated time for resampling: ~{((t2-t1)*n)/60.0:.0f} min")

    print("Start resampling...")
    print("----------------------------------------")

    # Determine the GRID fit results for the resampled data
    fit_results_resampled = []
    with tqdm(total=n) as pbar:
        # Just start at 1 for the print statement later
        for i in range(1, n + 1):
            fit_results_temp = _resample_data_and_fit_worker(
                parameters, data, perc=perc
            )
            fit_results_resampled.append(fit_results_temp)
            pbar.update(1)
            print()
            print(f"{i}-th resampling is finished.")

    t3 = time.time()
    print("Resampling is finished.")
    print(f"Total time: {(t3-t0)/60.0:.0f} min")
    print("----------------------------------------")

    return fit_results_full, fit_results_resampled


def init_pool():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# TODO: add docstring
def _resample_and_fit_multiprocess(
    fn: Callable[
        [Dict, Dict[str, Dict[str, np.ndarray]], bool],
        Dict[str, Dict[str, Union[np.ndarray, float]]],
    ],
    parameters: Dict,
    data: Dict[str, Dict[str, np.ndarray]],
    n: int = 10,
    perc: float = 0.8,
    max_workers: int = None,
):

    # First determine the max number of workers
    # -----------------------------------------
    # The limit for the max number of workers is number of logical cores minus 1 (there
    # needs to be one logical core for the background tasks)
    max_workers_limit = psutil.cpu_count(logical=False) - 1

    if max_workers is None:
        max_workers = max_workers_limit
    elif max_workers < 1:
        raise ValueError("max_workers should be >=1")
    elif max_workers > max_workers_limit:
        max_workers = max_workers_limit
        print(
            f"WARNING: max_workers is set to {max_workers}, since initial value",
            "was set higher than the allowed number of workers ({max_workers_limit}).",
        )

    t0 = time.time()

    # Then perform fitting on the full dataset
    # ----------------------------------------
    print("----------------------------------------")
    print("Fitting of all data starts now...")
    # Determine the full fit results based on all the data.
    t1 = time.time()
    fit_results_full = fn(parameters, data, disp=False)
    t2 = time.time()
    print("Fitting of all data finished now.")
    print("----------------------------------------")

    print(f"Estimated time for resampling: ~{((t2-t1)*n)/60.0/max_workers:.0f} min")

    print("Start resampling...")
    print("----------------------------------------")

    # Determine the GRID fit results for the resampled data
    fit_results_resampled = []
    i = 1
    with tqdm(total=n) as pbar:
        # Parallelization
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, initializer=init_pool
        ) as executor:
            try:

                # Submit the tasks
                futures = [
                    executor.submit(
                        _resample_data_and_fit_worker,
                        fn,
                        parameters,
                        data,
                        perc,
                    )
                    for _ in range(n)
                ]
                # Check if a task is finished
                for future in concurrent.futures.as_completed(futures):
                    fit_results_temp = future.result()
                    fit_results_resampled.append(fit_results_temp)
                    pbar.update(1)
                    print()
                    print(f"{i}-th resampling is finished.")
                    i += 1
            except KeyboardInterrupt:
                print("Caught keyboardinterrupt")

    t3 = time.time()
    print("Resampling is finished.")
    print(f"Total time: {(t3-t0)/60.0:.0f} min")
    print("----------------------------------------")

    return fit_results_full, fit_results_resampled


def resample_and_fit(
    parameters: Dict,
    data: Dict[str, Dict[str, np.ndarray]],
    n: int = 200,
    perc: float = 0.8,
    fit_mode: str = "grid",
    multiprocess_flag: bool = True,
    max_workers: int = None,
) -> Tuple[
    Dict[str, Dict[str, Union[np.ndarray, float]]],
    List[Dict[str, Dict[str, Union[np.ndarray, float]]]],
]:
    """
    This function performs a resampling procedure to fit survival time and value data
    with a given set of parameters. It first fits the full dataset with the parameters,
    then resamples the data and fits the resampled datasets with the parameters. The
    function returns the fit results for the full dataset and the resampled datasets.

    Parameters
    ----------
    parameters : Dict
        Dictionary containing all the parameters needed to perform the GRID and/or
        multi-exponential fitting.
    data : Dict[str, Dict[str, np.ndarray]]
        A dictionary mapping keys (time-lapse conditions) to the corresponding time and
        value arrays of the survival functions. For example::

            {
                "0.05s": {
                    "time": array([0.05, 0.1, 0.15, ...]),
                    "value": array([1.000e+04, 8.464e+03, 7.396e+03, ...]),
                }, "1s": {
                    "time": array([1., 2., 3., ...]),
                    "value": array([1.000e+04, 6.925e+03, 5.541e+03, ...]),
                },
            }

    n : int, optional
        The number of times you want to resample from the full dataset. Defaults to 200.
    perc : float, optional
        The percentage of data points to include in each resampled dataset. Defaults to
        0.8.
    fit_mode : {"grid", "multi-exp", "all"}, optional
        The fitting procedure to perform:

        * "grid": perform GRID fitting procedure on the data.
        * "multi-exp": perform multi-exponential fitting procedure on the data.
        * "all": perform both GRID fitting and multi-exponential fitting procedure
            on the data.

    multiprocess_flag : bool, optional
        A flag indicating whether to use parallel processing when fitting the resampled
        datasets. Defaults to True.
    max_workers: int, optional
        The maximum number of logical cores to use for the multiprocessing. This number
        is only used if multiprocess_flag is set to True and the value has to be between
        1 and the maximum # logical cores - 1. If value is set to None, than it will be
        set to # logical cores - 1. Default is None.

    Returns
    -------
    Tuple consisting of the following two objects:

    fit_results_full: Dict[str, Dict[str, Union[np.ndarray, float]]]
        A dictionary mapping keys (fitting procedure) to the corresponding fit results.
        For example::

            {
                "grid": {
                    "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                    "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                    "a": 0.010564217803906671,
                    "loss": 0.004705659331508584,
                },
            }

    fit_results_resampled: List[Dict[str, Dict[str, Union[np.ndarray, float]]]]
        A list of dictionaries, each a dictionary mapping keys (fitting procedure) to
        the corresponding fit results. For example::

            [
                {
                    "grid": {
                        "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                        "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                        "a": 0.010564217803906671,
                        "loss": 0.004705659331508584,
                    },
                },
                {
                    "grid": {
                        "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                        "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                        "a": 0.010564217803906671,
                        "loss": 0.004705659331508584,
                    },
                },
                ...
            ]
    """

    if fit_mode not in {"grid", "multi-exp", "all"}:
        raise ValueError("fit_mode is not a valid, see docstring for valid modes.")

    # Set the correct function for the fitting
    if fit_mode == "grid":
        fn = fit.fit_grid
    elif fit_mode == "multi-exp":
        fn = fit.fit_multi_exp
    elif fit_mode == "all":
        fn = fit.fit_all

    if multiprocess_flag:
        fit_results_full, fit_results_resampled = _resample_and_fit_multiprocess(
            fn, parameters, data, n=n, perc=perc, max_workers=max_workers
        )
    else:
        fit_results_full, fit_results_resampled = _resample_and_fit_sequential(
            fn, parameters, data, n=n, perc=perc
        )

    return fit_results_full, fit_results_resampled
