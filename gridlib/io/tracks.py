"""
Module with functions to read and write tracking data
"""
import pathlib
from typing import Tuple, Union, List
import csv

# TODO: update this to make it more general


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
