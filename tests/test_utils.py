"""Tests for `utils` module in the tests folder."""
import copy

import numpy as np
import pytest

from . import utils


def test_approx_nested_dict_data():
    """Test function `approx_nested_dict_data()`."""
    dict1 = {
        "100ms": {
            "time": np.array([0.1, 0.2, 0.3, 0.4]),
            "value": np.array([1, 2, 3, 4]),
        },
        "3.5s": {
            "time": np.array([3.5, 7.0]),
            "value": np.array([5, 6]),
        },
        "250ms": {
            "time": np.array([0.25, 0.5, 0.75]),
            "value": np.array([7, 8, 9]),
        },
    }

    dict2 = copy.deepcopy(dict1)

    # Last elements for 100ms are missing
    dict3 = {
        "100ms": {
            "time": np.array([0.1, 0.2, 0.3]),
            "value": np.array([1, 2, 3]),
        },
        "3.5s": {
            "time": np.array([3.5, 7.0]),
            "value": np.array([5, 6]),
        },
        "250ms": {
            "time": np.array([0.25, 0.5, 0.75]),
            "value": np.array([7, 8, 9]),
        },
    }

    # Last value element for 3.5s is missing
    dict4 = {
        "100ms": {
            "time": np.array([0.1, 0.2, 0.3, 0.4]),
            "value": np.array([1, 2, 3, 4]),
        },
        "3.5s": {
            "time": np.array([3.5, 7.0]),
            "value": np.array([5]),
        },
        "250ms": {
            "time": np.array([0.25, 0.5, 0.75]),
            "value": np.array([7, 8, 9]),
        },
    }

    # Time-lapse time 250ms is missing completely
    dict5 = {
        "100ms": {
            "time": np.array([0.1, 0.2, 0.3, 0.4]),
            "value": np.array([1, 2, 3, 4]),
        },
        "3.5s": {
            "time": np.array([3.5, 7.0]),
            "value": np.array([5, 6]),
        },
    }

    assert utils.approx_nested_dict_data(dict1, dict2)

    with pytest.raises(AssertionError):
        assert utils.approx_nested_dict_data(dict1, dict3)
        assert utils.approx_nested_dict_data(dict1, dict4)
        assert utils.approx_nested_dict_data(dict1, dict5)
