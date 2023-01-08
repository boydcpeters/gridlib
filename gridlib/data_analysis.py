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
    Tuple[np.ndarray, np.ndarray]
    k_peaks: np.ndarray
        The decay rates of the peaks. The total number of peaks is equal to the size of
        the array.
    s_peaks: np.ndarray
        The event amplitudes of the peaks. The total number of peaks is equal to the
        size of the array.

    Example of a resulting tuple::

        (array([0.00478, 0.0280, 0.190, 1.22, 6.39]), array([0.0191, 0.0494, 0.119, 0.264, 0.549]))

    Raises
    ------
    ValueError
        If no GRID fit results are provided in the fit results.
    """

    if "grid" not in fit_results:
        raise ValueError("There are no grid results in the provided fit results.")

    # Extract the decay rates and their respective weight values
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

    # Create the arrays to store the peak values
    n = max(clusters_temp.keys())
    k_peaks = np.zeros(n, dtype=np.float64)
    s_peaks = np.zeros(n, dtype=np.float64)

    for cluster_num, data_points in clusters_temp.items():

        # Calculate the weighted values of the cluster
        k_weighted = 0
        s_total = 0
        for data_point in data_points:
            k, s = data_point
            k_weighted += k * s
            s_total += s

        k_weighted = k_weighted / s_total

        k_peaks[cluster_num - 1] = k_weighted
        s_peaks[cluster_num - 1] = s_total

    return k_peaks, s_peaks
