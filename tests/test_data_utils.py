"""Tests for `data_utils` module."""
import copy

import numpy as np
import pytest

from gridlib import data_utils

from . import utils


def test__num_decimal_places():
    """Test the function '_num_decimal_places()'."""
    assert data_utils._num_decimal_places("5.32") == 2
    assert data_utils._num_decimal_places("2.") == 0
    assert data_utils._num_decimal_places("98.8572") == 4
    assert data_utils._num_decimal_places("3.4120") == 3
    assert data_utils._num_decimal_places("11") == 0
    assert data_utils._num_decimal_places("0.05") == 2


def test_get_time_sec():
    """Test the function 'get_time_sec()'."""
    assert data_utils.get_time_sec("50ms") == 0.05
    assert data_utils.get_time_sec("4.2s") == 4.2
    assert data_utils.get_time_sec("1150ms") == 1.15
    assert data_utils.get_time_sec("2s") == 2
    assert data_utils.get_time_sec("100ms") == 0.1

    with pytest.raises(ValueError):
        data_utils.get_time_sec("zems")
        data_utils.get_time_sec("1150")

    with pytest.raises(ValueError):
        data_utils.get_time_sec(52)
        data_utils.get_time_sec(3.2)
        data_utils.get_time_sec({52 == 3.2})
        data_utils.get_time_sec([11 == 4.23])
        data_utils.get_time_sec({"test": 2.1})


def test__fmt_t_str_key():
    """Test function '_fmt_t_str_key()'."""
    assert data_utils._fmt_t_str_key("50ms") == "0.05s"
    assert data_utils._fmt_t_str_key("4.2s") == "4.2s"
    assert data_utils._fmt_t_str_key("1150ms") == "1.15s"
    assert data_utils._fmt_t_str_key("2s") == "2s"
    assert data_utils._fmt_t_str_key("100ms") == "0.1s"

    with pytest.raises(ValueError):
        data_utils._fmt_t_str_key("zems")
        data_utils._fmt_t_str_key("1150")

    with pytest.raises(ValueError):
        data_utils._fmt_t_str_key(52)
        data_utils._fmt_t_str_key(3.2)
        data_utils._fmt_t_str_key({52 == 3.2})
        data_utils._fmt_t_str_key([11 == 4.23])
        data_utils._fmt_t_str_key({"test": 2.1})


def test_fmt_t_str_data():
    """Test function `fmt_t_str_data()`."""

    data = {
        "50ms": {
            "time": np.array([0.05, 0.1, 0.15]),
            "value": np.array([41, 23, 8]),
        },
        "1s": {"time": np.array([1.0, 2.0]), "value": np.array([34, 9])},
    }

    data_original = copy.deepcopy(data)

    desired = {
        "0.05s": {
            "time": np.array([0.05, 0.1, 0.15]),
            "value": np.array([41, 23, 8]),
        },
        "1s": {"time": np.array([1.0, 2.0]), "value": np.array([34, 9])},
    }
    assert utils.approx_nested_dict_data(data_utils.fmt_t_str_data(data), desired)

    # Check if the original data dictionary wasn't altered
    assert utils.approx_nested_dict_data(data, data_original)


def test_process_data():
    """Test function `process_data()`."""
    data = {
        "50ms": {"time": np.array([0.05, 0.1, 0.15]), "value": np.array([41, 23, 8])},
        "1s": {"time": np.array([1.0, 2.0]), "value": np.array([34, 9])},
    }

    data_original = copy.deepcopy(data)

    desired_default = {
        "0.05s": {
            "time": np.array([0.1, 0.15]),
            "value": np.array([1.0, 0.34782609]),
        },
        "1s": {
            "time": np.array([1.0, 2.0]),
            "value": np.array([1.0, 0.26470588]),
        },
    }

    desired_delete_false = {
        "0.05s": {
            "time": np.array([0.05, 0.1, 0.15]),
            "value": np.array([1.0, 0.56097561, 0.19512195]),
        },
        "1s": {
            "time": np.array([1.0, 2.0]),
            "value": np.array([1.0, 0.26470588]),
        },
    }
    assert utils.approx_nested_dict_data(data_utils.process_data(data), desired_default)
    assert utils.approx_nested_dict_data(
        data_utils.process_data(data, delete=False), desired_delete_false
    )

    # Check if the original data dictionary wasn't altered
    assert utils.approx_nested_dict_data(data, data_original)
