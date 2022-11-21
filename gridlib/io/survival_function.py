import csv
import pathlib
from typing import Dict, Union

import numpy as np

from .. import data_utils


def write_data_survival_function(
    path: Union[str, pathlib.Path], data: Dict[str, Dict[str, np.ndarray]]
):
    """Function writes the survival function data to a csv file.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the location of the file, where to save to. If the file does not exist,
        it will be created.
    data: Dict[str, Dict[str, np.ndarray]]
        Data of the survival function to write to a csv file. The dictionary structure
        is as follows:
        {
            "t_tl": {
                "time": np.ndarray with the time points,
                "value": np.ndarray with the survival function values,
            }
        }

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the path suffix does not end with .csv

    Notes
    -----
    The .csv file with the survival time distributions will looks as follows
    (without the header):

    time-lapse 1, value, time-lapse 2, value, ...\n
    Delta t_1,f1(Delta t_1), Delta t_2, f2(Delta t_2)\n
    2*Delta t_1,f1(2*Delta t_1), 2*Delta t_2, f2(2*Delta t_2)\n
    3*Delta t_1,f1(3*Delta t_1), 3*Delta t_2, f2(3*Delta t_2)\n
    ..., ..., ..., ...
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix != ".csv":
        raise ValueError(f"path should end with .csv, but ends with {path.suffix}")

    # Make sure the units for all the time-lapse conditions is in seconds
    # This is needed for sorting of the values
    data = data_utils.fmt_t_str_data(data)

    t_tl_all = list(data.keys())

    # Sort the tl values from low to high.
    t_tl_all.sort()

    max_length = 0

    for t_tl in t_tl_all:
        val = data[t_tl]["time"].shape[0]
        if val > max_length:
            max_length = val

    rows_to_write = []
    for i in range(max_length):
        row = []
        for t_tl in t_tl_all:
            row.append(data[t_tl]["time"][i] if i < data[t_tl]["time"].shape[0] else "")
            row.append(
                data[t_tl]["value"][i] if i < data[t_tl]["value"].shape[0] else ""
            )
        row = tuple(row)
        rows_to_write.append(row)

    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows_to_write:
            writer.writerow(row)
    print(f"Writing data to {path} is finished.")


def read_data_survival_function(
    path: Union[str, pathlib.Path]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Function reads survival function from csv file.

    Parameters
    ----------
    path: str or pathlib.Path
        Path to the location of the .csv file containing the survival time
        distributions.

    Returns
    -------
    data: Dict[str, Dict[str, np.ndarray]]
        Data of the survival function to write to a csv file. The dictionary structure
        is as follows:
        {
            "t_tl": {
                "time": np.ndarray with the time points,
                "value": np.ndarray with the survival function values,
            }
        }

    Raises
    ------
    ValueError
        If the path suffix does not end with .csv
    FileNotFoundError
        If the file is not present at path.

    Notes
    -----
    The .csv file with the surivival time distributions should look like this
    (without the header):

    time-lapse 1, value, time-lapse 2, value, ...\n
    Delta t_1,f1(Delta t_1), Delta t_2, f2(Delta t_2)\n
    2*Delta t_1,f1(2*Delta t_1), 2*Delta t_2, f2(2*Delta t_2)\n
    3*Delta t_1,f1(3*Delta t_1), 3*Delta t_2, f2(3*Delta t_2)\n
    ..., ..., ..., ...
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    if path.suffix != ".csv":
        raise ValueError(f"path should end with .csv, but ends with {path.suffix}")

    data = dict()

    with open(path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)

        # Set-up the data dictionary
        row1 = next(reader)  # read the first row
        row2 = next(reader)  # read the second row
        # determine the number of tl conditions, every time-lapse condition consist of
        # two columns, the time column and the value column
        num_tls = len(row1) // 2

        for i in range(num_tls):
            # Time-lapse condition is determined by calculation the difference
            # between the first and second time point.
            time_1 = float(row1[2 * i])
            time_2 = float(row2[2 * i])

            # Store the initial key
            time_s = round(time_2 - time_1, 5)
            t_tl = f"{time_s}s"
            data[t_tl] = dict()

        # Format the keys correctly, it is possible that the key due to rounding
        # was 1.00000s, so now format them to 1s
        data = data_utils.fmt_t_str_data(data)

        # Set the pointer back to the start of the file
        csvfile.seek(0)

        for row in reader:
            for i, t_tl in zip(range(num_tls), data.keys()):

                # Read-out the time value
                time_s_str = row[(2 * i)]
                if time_s_str != "":
                    temp = data[t_tl].get("time", [])
                    time_s = round(float(time_s_str), 4)
                    temp.append(time_s)
                    data[t_tl]["time"] = temp

                # Read out the distribution value
                val_str = row[(2 * i) + 1]
                if val_str != "":
                    temp = data[t_tl].get("value", [])
                    # value could be ###.0, so first convert str to float
                    val = int(float(val_str))
                    temp.append(val)
                    data[t_tl]["value"] = temp

        # After the lists are completely filled, loop over all the lists and convert
        # them to arrays
        for t_tl in data.keys():
            for key in data[t_tl].keys():
                data[t_tl][key] = np.array(data[t_tl][key])

    return data
