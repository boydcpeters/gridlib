from typing import Dict

import numpy as np


def approx_nested_dict_data(
    dict1: Dict[str, Dict[str, np.ndarray]], dict2: Dict[str, Dict[str, np.ndarray]]
):
    """Function tests whether two data dictionaries are approximately equal.

    Parameters
    ----------
    dict1: Dict[str, Dict[str, np.ndarray]]
        Survival function data for time-lapse conditions with the following data
        structure:
            {
                "t_tl": {
                    "time": np.ndarray with the time points,
                    "value": np.ndarray with the survival function values,
                },
                ...
            }
    dict2: Dict[str, Dict[str, np.ndarray]]
        Same as dict1.

    Returns
    -------
    bool
        Returns True if it passes all the checks and the data dictionaries are
        approximately the same.

    Raises
    ------
    AssertionError
        If a check is not passed and thus the dictionaries are not approximately the
        same.
    """

    # Check that all keys are the same
    assert dict1.keys() == dict2.keys()

    # Now check that all the nested arrays are approximately the same
    for key1 in dict1.keys():
        np.testing.assert_allclose(dict1[key1]["time"], dict2[key1]["time"])
        np.testing.assert_allclose(dict1[key1]["value"], dict2[key1]["value"])

    # If it goes through all the checks return True
    return True
