import numpy as np


def approx_nested_dict_data(dict1, dict2):
    """Function tests whether a nested dictionary with the data structure are
    approximately equal."""

    # Check that all keys are the same
    assert dict1.keys() == dict2.keys()

    # Now check that all the nested arrays are approximately the same
    for key1 in dict1.keys():
        for key2 in dict1[key1].keys():
            np.testing.assert_allclose(dict1[key1][key2], dict2[key1][key2])

    return True
