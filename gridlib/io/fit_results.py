"""
Module with functions to read and write fit results.
"""
from typing import Tuple, Union, Dict
import pathlib
import numpy as np
import scipy.io as sio

# TODO: likely not going to do .csv, .mat files seem great kind of hdf5 file format
# TODO: add function to save fit results to .csv file (also as pickle file??)
# TODO: add function to read fit results to .csv file
# TODO: add function to read and write fit results from .mat file from original GRID
# package


def write_fit_results(
    path: Union[str, pathlib.Path],
    fit_results: Tuple[Dict[str, Union[np.ndarray, float]]],
):
    """Function writes fit results to a .mat file."""

    if isinstance(path, str):
        path = pathlib.Path(path)

    # Dictionary of which the information is stored in the .mat file.
    mdic = dict()

    # Loop over all the fit results and check the number of exponentials
    # The assumption is made that for a number of more than 10 exponentials, it is the
    # result of GRID fitting.
    for fit_result in fit_results:
        n = fit_result["k"].shape[0]
        if n >= 10:
            key = "grid"
        else:
            key = f"{n}-exp"

        mdic[key] = fit_result

    sio.savemat(path, mdic)
    print(f"Fit results are saved in {path}")


def read_fit_results(
    path: Union[str, pathlib.Path]
) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """Function reads the fit results from a .mat file."""

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
