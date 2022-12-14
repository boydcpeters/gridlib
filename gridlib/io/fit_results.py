"""
Module with functions to read and write fit results.
"""
import pathlib
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.io as sio

# TODO: likely not going to do .csv, .mat files seem great kind of hdf5 file format
# TODO: add function to save fit results to .csv file (also as pickle file??)
# TODO: add function to read fit results to .csv file
# TODO: add function to read and write fit results from .mat file from original GRID
# package


def write_fit_results(
    path: Union[str, pathlib.Path],
    fit_results: Dict[str, Dict[str, Union[np.ndarray, float]]],
) -> None:
    """
    Write fit results to a .mat file.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        The file path to write the fit results to.
    fit_results : Dict[str, Dict[str, Union[np.ndarray, float]]]
        A dictionary containing the fit results, where the keys are the names of the fit
        results and the values are dictionaries. The dictionaries have keys 'grid' or
        f"{n}-exp", where n is the number of exponentials, and values that are either
        numpy arrays or floats.

    Returns
    -------
    None

    Examples
    --------
    >>> fit_results = {'grid': {'k': np.array([1.0e-03, ..., 10.0]),
    ... 's': np.array([1.000e-03, ..., 2.500e-03]), 'a': 0.022, 'loss': 0.0047},
    ... '1-exp': {'k': array([0.02563639]), 's': array([1.]), 'a': 0.08514936433699753,
    ... 'loss': 1.2825570522448484}}
    >>> write_fit_results("/path/to/file.mat", fit_results)

    Fit results are saved in "/path/to/file.mat".
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    sio.savemat(path, fit_results)
    print(f"Fit results are saved in {path}")


def read_fit_results(
    path: Union[str, pathlib.Path]
) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """
    Function reads the fit results from a .mat file.

    Parameters
    ----------
    path : Union[str, pathlib.Path]
        The file path to read the fit results from.

    Returns
    -------
    fit_results : Dict[str, Dict[str, Union[np.ndarray, float]]]
        A dictionary containing the fit results, where the keys are the names of the fit results and the values are
        dictionaries. The dictionaries have keys 'grid' or f"{n}-exp", where n is the number of exponentials, and
        values that are either numpy arrays or floats.

    Examples
    --------
    >>> fit_results = read_fit_results("/path/to/file.mat")
    >>> {'grid': {'k': np.array([1.0e-03, ..., 10.0]), 's': np.array([1.000e-03, ..., 2.500e-03]), 'a': 0.022, 'loss': 0.0047}, '1-exp': {'k': array([0.02563639]), 's': array([1.]), 'a': 0.08514936433699753, 'loss': 1.2825570522448484}}
    """

    mat_contents = sio.loadmat(path, simplify_cells=True)

    return mat_contents


def read_fit_results_trackit(
    path: Union[str, pathlib.Path]
) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """Function loads and parses the fit results from the original GRID package,
    which saves the data in a different data structure, so hence there is some extra
    parsing involved."""
    # TODO: improve documentation

    if isinstance(path, str):
        path = pathlib.Path(path)

    mat_contents = sio.loadmat(path, simplify_cells=True)

    # Create the dictionary with the spectrum values
    fit_results_grid = dict()
    fit_results_grid["k"] = mat_contents["spectrum"]["dissociation_rates"]
    fit_results_grid["S"] = mat_contents["spectrum"]["Spectrum"]
    fit_results_grid["a1"] = mat_contents["spectrum"]["bleachingnumber_1"]
    fit_results_grid["error_test"] = mat_contents["spectrum"]["error_test"]

    # Single-exponential data
    fit_results_single_exp = dict()
    fit_results_single_exp["k"] = np.array(
        [mat_contents["monoexponential"]["dissociation_rate"]], dtype=np.float64
    )
    fit_results_single_exp["s"] = np.ones(1, dtype=np.float64)
    fit_results_single_exp["a"] = mat_contents["monoexponential"]["bleachingnumber_1"]
    fit_results_single_exp["adj_R_squared"] = mat_contents["monoexponential"][
        "Adj_R_Sqared"
    ]

    # Double-exponential data
    fit_results_double_exp = dict()
    fit_results_double_exp["k"] = mat_contents["twoexponential"]["dissociation_rates"]
    fit_results_double_exp["s"] = mat_contents["twoexponential"]["Amplitudes"]
    fit_results_double_exp["a"] = mat_contents["twoexponential"]["bleachingnumber_1"]
    fit_results_double_exp["adj_R_squared"] = mat_contents["twoexponential"][
        "Adj_R_Sqared"
    ]

    # Triple-exponential data
    fit_results_triple_exp = dict()
    fit_results_triple_exp["k"] = mat_contents["threeexponential"]["dissociation_rates"]
    fit_results_triple_exp["s"] = mat_contents["threeexponential"]["Amplitudes"]
    fit_results_triple_exp["a"] = mat_contents["threeexponential"]["bleachingnumber_1"]
    fit_results_triple_exp["adj_R_squared"] = mat_contents["threeexponential"][
        "Adj_R_Sqared"
    ]

    # Throw all the results into one dictionary
    fit_results = {
        "grid": fit_results_grid,
        "1-exp": fit_results_single_exp,
        "2-exp": fit_results_double_exp,
        "3-exp": fit_results_triple_exp,
    }

    return fit_results


def write_data_grid_resampling(
    path: Union[str, pathlib.Path],
    fit_result_full: Tuple[Dict[str, Union[np.ndarray, float]]],
    fit_results_resampled: List[Dict[str, Union[np.ndarray, float]]],
):
    """Function writes the resampling results to a .mat file."""

    if isinstance(path, str):
        path = pathlib.Path(path)

    data_save = {"result100": fit_result_full, "results": fit_results_resampled}

    sio.savemat(path, data_save)
    print(f"Resampling results are saved in {path}")


def read_data_grid_resampling(
    path: str,
) -> Tuple[
    Dict[str, Union[np.ndarray, float]], List[Dict[str, Union[np.ndarray, float]]]
]:
    """Function loads and parses the resampling data from GRID.

    Parameters
    ----------
    path: str
        Path to the file with all the resampling data

    Returns
    -------
    fit_result_full: Dict[str, Union[np.ndarray, float]]
        Dictionary with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "s": np.ndarray with the corresponding weights
            "a": bleaching number (a = kb * t_int)
            "loss": final cost value
    fit_results_resampled: List[Dict[str, Union[np.ndarray, float]]]
        List of dictionaries with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "s": np.ndarray with the corresponding weights
            "a": bleaching number (a = kb * t_int)
            "loss": final cost value
        Every dictionary entry in the list contains the results of one data resample.
    """
    mat_contents = sio.loadmat(path, simplify_cells=True)

    fit_result_full = mat_contents["result100"]
    fit_results_resampled = mat_contents["results"]

    return fit_result_full, fit_results_resampled


def read_data_grid_resampling_trackit(
    path: str,
) -> Tuple[
    Dict[str, Union[np.ndarray, float]], List[Dict[str, Union[np.ndarray, float]]]
]:
    """Function loads and parses the resampling data from GRID.

    Parameters
    ----------
    path: str
        Path to the file with all the resampling data

    Returns
    -------
    fit_result_full: Dict[str, Union[np.ndarray, float]]
        Dictionary with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "s": np.ndarray with the corresponding weights
            "a": bleaching number (a = kb * t_int)
            "loss": np.NaN (since this is not provided by TrackIt resampling data)
    fit_results_resampled: List[Dict[str, Union[np.ndarray, float]]]
        List of dictionaries with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "s": np.ndarray with the corresponding weights
            "a": bleaching number (a = kb * t_int)
            "loss": np.NaN (since this is not provided by TrackIt resampling data)
        Every dictionary entry in the list contains the results of one data resample.
    """
    mat_contents = sio.loadmat(path, simplify_cells=True)

    fit_result_full_temp = mat_contents["result100"]
    fit_results_resampled_temp = mat_contents["results"]

    # Update the keys, so consistent with API requirements
    fit_result_full = {
        "grid": {
            "k": fit_result_full_temp["k"],
            "s": fit_result_full_temp["S"],
            "a": fit_result_full_temp["a1"],
            "loss": np.NaN,
        }
    }

    fit_results_resampled = list()
    for fit_result_resampled_temp in fit_results_resampled_temp:

        # Update the keys, so consistent with API requirements
        result_resampled = {
            "grid": {
                "k": fit_result_resampled_temp["k"],
                "s": fit_result_resampled_temp["S"],
                "a": fit_result_resampled_temp["a1"],
                "loss": np.NaN,
            }
        }

        fit_results_resampled.append(result_resampled)

    return fit_result_full, fit_results_resampled
