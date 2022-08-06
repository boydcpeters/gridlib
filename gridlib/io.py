import csv
import pathlib
import re
from typing import Dict, List, Tuple, Union

import numpy as np
import scipy.io as sio

# Initially only allow .csv files?
# TODO: function to read track files and determine the track lifes (.csv/.txt?)
# TODO: function to read number of frames, with as header the t_tl (.csv/.txt?)
# TODO: function to read the survival functions (.csv/.txt?)
# TODO: function to write the survival functions (.csv/.txt?)


def reformat_t_tl(t_tl):
    """Function formats the time unit pattern to all seconds

    eg 100ms -> 0.10s

    """
    # pattern to match time pattern in the files, eg. 100ms or 2s
    time_pattern = re.compile(
        r"[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)", flags=re.IGNORECASE
    )
    unit_pattern = re.compile("(ms|s)", flags=re.IGNORECASE)

    # Match time pattern
    m_time = time_pattern.search(t_tl)

    # Match unit pattern
    m_unit = unit_pattern.search(t_tl)

    # get the number value of the match, eg. 50, 100, 390, 4
    t_tl_val = float(m_time.group(0))
    # get the unit value of the match, eg. ms or s
    t_tl_unit = m_unit.group(0)

    # Calculate the frame cycle time
    if t_tl_unit == "ms":
        frame_cycle_time = t_tl_val / 1000.0
    elif t_tl_unit == "s":
        frame_cycle_time = t_tl_val

    return f"{frame_cycle_time:.2f}s"


def read_track_file_txt(path: Union[str, pathlib.Path]) -> List[Tuple]:
    """Function reads track data from a .txt file that has the data format of track files created
    by SOS and returns the data.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the file with track information.

    Returns
    -------
    data: list of tuples
        List with tuples, where every tuple contains the information of one part of a track. Every
        tuple has the following structure: (f, t, x, y, track_id, disp, intensity, sigma, fit_error).
            f (int): frame number
            t (float): time
            x (float): x-coordinate in pixel value
            y (float): y-coordinate in pixel value
            track_id (int): id of the track
            disp (float): displacement in pixel values between current localization of track and
                previous one. The value is -1 if it is the first localization of the track.
            intensity (float): TODO is from SOS
            sigma (float): TODO is from SOS
            fit_error (float): TODO is from SOS

    Raises
    ------
    ValueError
        If the path suffix does not end with .txt
    FileNotFoundError
        If the file is not present at path.
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix != ".txt":
        raise ValueError(f"path should end with .txt, but ends with {path.suffix}")

    data = []
    # data = [None] * sum(1 for line in open(path, "r"))
    i = 0
    with open(path, "r") as rf:
        for line in rf:
            line = line.strip()
            values = line.split("\t")
            if len(values) == 8:
                f = int(float(values[0]))
                t = None
                x = float(values[1])
                y = float(values[2])
                track_id = int(float(values[3]))
                disp = float(values[4])
                intensity = float(values[5])
                sigma = float(values[6])
                fit_error = float(values[7])
                data.append((f, t, x, y, track_id, disp, intensity, sigma, fit_error))
            else:
                print(
                    f"Something went wrong at {path}, line {i} does not have 8 values."
                )
            i += 1
    return data


def read_track_file_csv(path: Union[str, pathlib.Path]) -> List[Tuple]:
    """Function reads track data from a .csv file that has the data format of track files created
    by the TrackIt framework and returns the data.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the file with track information.

    Returns
    -------
    data: list of tuples
        List with tuples, where every tuple contains the information of one part of a track. Every
        tuple has the following structure: (f, t, x, y, track_id, disp, intensity, sigma, fit_error).
            f (int): frame number
            t (float): time
            x (float): x-coordinate in um
            y (float): y-coordinate in um
            track_id (int): id of the track
            disp (float): displacement in um between current localization of track and previous one.
                The value is -1 if it is the first localization of the track.
            intensity (float): TODO is from SOS
            sigma (float): TODO is from SOS
            fit_error (float): TODO is from SOS

    Raises
    ------
    ValueError
        If the path suffix does not end with .csv
    FileNotFoundError
        If the file is not present at path.
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix != ".csv":
        raise ValueError(f"path should end with .csv, but ends with {path.suffix}")

    data = []

    with open(path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if len(row) == 6:
                f = int(float(row["frame"]))
                t = float(row["t"])
                x = float(row["x"])
                y = float(row["y"])
                track_id = int(float(row["trajectory"]))

                # If the track_id is the same as the previous point
                # then we can calculate a displacement otherwise set at -1
                if i != 0 and track_id == data[-1][3]:
                    disp = ((x - data[-1][1]) ** 2 + (y - data[-1][2]) ** 2) ** 0.5
                else:
                    disp = -1

                intensity = None
                sigma = None
                fit_error = None
                data.append((f, t, x, y, track_id, disp, intensity, sigma, fit_error))
            else:
                print(
                    f"Something went wrong at {path}, line {i} does not have 6 values."
                )

    return data


def write_track_file_csv(path: Union[str, pathlib.Path], data: List[Tuple]):
    """Function writes track data to a .csv file with the data format of track files created
    by the TrackIt framework.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the file with track information.

    data: list of tuples
        List with tuples, where every tuple contains the information of one part of a track. Every
        tuple has the following structure: (f, t, x, y, track_id, disp, intensity, sigma, fit_error).
            f (int): frame number
            t (float): time
            x (float): x-coordinate in um
            y (float): y-coordinate in um
            track_id (int): id of the track
            disp (float): displacement in um between current localization of track and previous one.
                The value is -1 if it is the first localization of the track.
            intensity (float): TODO is from SOS
            sigma (float): TODO is from SOS
            fit_error (float): TODO is from SOS

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the path suffix does not end with .csv
    """

    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix != ".csv":
        raise ValueError(f"path should end with .csv, but ends with {path.suffix}")

    with open(path, "w", newline="") as csvfile:
        fieldnames = ["", "frame", "t", "trajectory", "x", "y"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(data):
            f, t, x, y, track_id, disp, intensity, sigma, fit_error = d
            writer.writerow(
                {"": i, "frame": f, "t": t, "trajectory": track_id, "x": x, "y": y}
            )
    print(f"Writing data to {path} is finished.")


def read_grid_fit_data(path: str):
    """Function loads and parses the GRID fit data."""
    # TODO: improve documentation

    mat_contents = sio.loadmat(path, simplify_cells=True)

    # Create the dictionary with the spectrum values
    fit_values_grid = dict()
    fit_values_grid["k"] = mat_contents["spectrum"]["dissociation_rates"]
    fit_values_grid["S"] = mat_contents["spectrum"]["Spectrum"]
    fit_values_grid["a1"] = mat_contents["spectrum"]["bleachingnumber_1"]
    fit_values_grid["error_test"] = mat_contents["spectrum"]["error_test"]

    # Single-exponential data
    fit_values_single_exp = dict()
    fit_values_single_exp["k"] = np.array(
        [mat_contents["monoexponential"]["dissociation_rate"]], dtype=np.float64
    )
    fit_values_single_exp["S"] = np.ones(1, dtype=np.float64)
    fit_values_single_exp["a1"] = mat_contents["monoexponential"]["bleachingnumber_1"]
    fit_values_single_exp["adj_R_squared"] = mat_contents["monoexponential"][
        "Adj_R_Sqared"
    ]

    # Double-exponential data
    fit_values_double_exp = dict()
    fit_values_double_exp["k"] = mat_contents["twoexponential"]["dissociation_rates"]
    fit_values_double_exp["S"] = mat_contents["twoexponential"]["Amplitudes"]
    fit_values_double_exp["a1"] = mat_contents["twoexponential"]["bleachingnumber_1"]
    fit_values_double_exp["adj_R_squared"] = mat_contents["twoexponential"][
        "Adj_R_Sqared"
    ]

    # Triple-exponential data
    fit_values_triple_exp = dict()
    fit_values_triple_exp["k"] = mat_contents["threeexponential"]["dissociation_rates"]
    fit_values_triple_exp["S"] = mat_contents["threeexponential"]["Amplitudes"]
    fit_values_triple_exp["a1"] = mat_contents["threeexponential"]["bleachingnumber_1"]
    fit_values_triple_exp["adj_R_squared"] = mat_contents["threeexponential"][
        "Adj_R_Sqared"
    ]

    return (
        fit_values_grid,
        fit_values_single_exp,
        fit_values_double_exp,
        fit_values_triple_exp,
    )


def read_grid_resampling_data(
    path: str,
) -> Tuple[
    Dict[str, Union[np.ndarray, float]], List[Dict[str, Union[np.ndarray, float]]]
]:
    """Function loads and parses the resampling data from GRID.

    Parameters:
    ----------
    path: str
        Path to the file with all the resampling data

    Returns:
    -------
    results_full: Dict[str, Union[np.array, float]]
        Dictionary with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "S": np.ndarray with the corresponding weights
            "a1": bleaching number (a = kb * t_int)
    results_resampling: List[Dict[str, Union[np.array, float]]]
        List of dictionaries with the following key-value pairs:
            "k": np.ndarray with the dissociation rates
            "S": np.ndarray with the corresponding weights
            "a1": bleaching number (a = kb * t_int)
        Every dictionary entry in the list contains the results of one data resample.
    """
    mat_contents = sio.loadmat(path, simplify_cells=True)

    results_full = mat_contents["result100"]
    results_resampling = mat_contents["results"]
    return results_full, results_resampling
