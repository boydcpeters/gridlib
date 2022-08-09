from typing import Dict
import numpy as np


def approx_nested_dict_data(
    dict1: Dict[str, Dict[str, np.ndarray]], dict2: Dict[str, Dict[str, np.ndarray]]
):
    """Function tests whether two nested dictionaries with the structure:
    {
        f"{t_tl}s": {
            "time": np.ndarray,
            "value": np.ndarray,
        },
        ...
    }
    are approximately equal.
    """

    # Check that all keys are the same
    assert dict1.keys() == dict2.keys()

    # Now check that all the nested arrays are approximately the same
    for key1 in dict1.keys():
        np.testing.assert_allclose(dict1[key1]["time"], dict2[key1]["time"])
        np.testing.assert_allclose(dict1[key1]["value"], dict2[key1]["value"])

    # If it goes through all the checks return True
    return True
