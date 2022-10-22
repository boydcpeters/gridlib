from typing import Tuple, Dict, Union, List
import math

import numpy as np

from . import compute
from . import data_utils
from gridlib.fit import fit_grid

# TODO: function works now, but the calculation of the probability is currently very
# whacky done. Clean it up a bit, so it is at least better understandable.
def _sample_data(data, perc: float = 0.8, seed=None):

    rng = np.random.default_rng(seed)

    data_sampled = dict()
    for t_tl in data.keys():

        # Retrieve the time in seconds
        t_tl_s = data_utils.get_time_sec(t_tl)

        # The total number of data points for a time-lapse condition in the complete
        # dataset
        data_point_all = data[t_tl]["value"][0]

        # Determine how many data points should be in the resampled version
        # math.ceil returns an integer
        data_points_sample = math.ceil(data_point_all * perc)

        track_lifes = np.rint(data[t_tl]["time"] / t_tl_s)
        # Cast it to the integer dtype, since the compute survival functions needs
        # an array with dtype integer.
        track_lifes = track_lifes.astype(np.int64)

        # TODO: clean this up
        # Calculate the probability for every track life
        temp = data[t_tl]["value"][::-1]
        temp2 = temp[1:] - temp[:-1]  # check out np.diff
        temp2 = temp2[::-1]  # need to invert it back
        temp2 = np.concatenate(
            (temp2, np.array([temp[0]]))
        )  # add the last element again
        p = temp2 / np.sum(temp2)
        # THE NEXT LINE IS WRONG!! THIS IS THE CUMULATIVE SUM, NOT THE INDIVIDUAL COUNTS
        # p = data[t_tl]["value"] / np.sum(data[t_tl]["value"])

        track_lifes_sampled = rng.choice(
            track_lifes,
            size=data_points_sample,
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


def resampling_grid(
    parameters,
    data,
    n: int = 10,
    perc: float = 0.8,
    seed=None,
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
    seed{None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
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

    # Determine the full GRID fit results based on all the data.
    fit_results_all = fit_grid(parameters, data)

    rng = np.random.default_rng(seed)

    # Determine the GRID fit results for the resampled data
    fit_results_resampled = []
    for _ in range(n):

        data_resampled = _sample_data(data, perc=perc, seed=rng)

        fit_results_temp = fit_grid(parameters, data_resampled)

        fit_results_resampled.append(fit_results_temp)

    return fit_results_all, fit_results_resampled