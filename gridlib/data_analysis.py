"""
Module containing routines to analyze the fit results.
"""

from typing import Dict, Union
import numpy as np


def find_peaks(
    fit_results: Dict[str, Dict[str, Union[np.ndarray, float]]],
    threshold: float = 10e-6,
) -> Dict[int, Dict[str, float]]:
    """
    "Function finds the peaks in a GRID spectrum in a basic way. It loops over the
    amplitude array and if an amplitude value is higher than the set threshold than
    it is included into a cluster. Amplitudes above the threshold next to eachother
    belong to the same cluster. For each cluster is then the weighted decay-rate and
    amplitude value calculated.

    Parameters
    ----------
    fit_results : Dict[str, Dict[str, Union[np.ndarray, float]]]
        A dictionary mapping keys (fitting procedure) to the corresponding fit results.
        For example::

        {
            "grid": {
                "k": array([1.00000000e-03, 1.04737090e-03, ...]),
                "s": array([3.85818587e-17, 6.42847878e-18, ...]),
                "a": 0.010564217803906671,
                "loss": 0.004705659331508584
            }
        }

    threshold : float, optional
        Minimum weight value for a data point in the GRID spectrum to be included into
        a cluster to determine peak values, by default 10e-6

    Returns
    -------
    peaks: Dict[int, Dict[str, float]]
        Dictionary mapping the peak number to the corresponding peak values. For
        example::

            {
                1: {"k": 0.004777406500406388,"s": 0.01913411845411184},
                2: {"k": 0.028005038941836313, "s": 0.049437255750847335},
                3: {"k": 0.18991111645029066, "s": 0.11870356923055253},
                4: {"k": 1.216802975611107, "s": 0.26403855056602654},
                5: {"k": 6.387479711796379, "s": 0.5486865059984619}
            }

    Raises
    ------
    ValueError
        If no GRID fit results are provided in the fit results.
    """

    if "grid" not in fit_results:
        raise ValueError("There are no grid results in the provided fit results.")

    # Extract the decay-rates and their respective weight values
    k = fit_results["grid"]["k"]
    s = fit_results["grid"]["s"]
    # state = (1 / k) * s
    # state = s / np.sum(s)  # normalization

    cluster_num = 1
    clusters_temp = dict()

    # Flag to indicate whether the previous data point was in a cluster
    prev_flag = False
    for i in range(k.shape[0]):
        if s[i] >= threshold:

            # A cluster is now being formed, so set the flag to True
            if not prev_flag:
                prev_flag = True

            temp = clusters_temp.get(cluster_num, [])
            temp.append((k[i], s[i]))
            clusters_temp[cluster_num] = temp

        else:
            # If a cluster was being created but a data point is reached that has a
            # lower weight than the threshold, set the prev_flag to False again, to
            # prevent new data points from entering, and increment the cluster number
            if prev_flag:
                cluster_num += 1
                prev_flag = False

    # Create the final peaks dictionary
    peaks = dict()
    for cluster_num, data_points in clusters_temp.items():

        # Calculate the weighted values of the cluster
        k_weighted = 0
        s_total = 0
        for data_point in data_points:
            k, s = data_point
            k_weighted += k * s
            s_total += s

        k_weighted = k_weighted / s_total

        # Store all the data in the newly created dictionary
        peaks[cluster_num] = dict()
        peaks[cluster_num]["k"] = k_weighted
        peaks[cluster_num]["s"] = s_total

    return peaks
