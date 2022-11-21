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


def test_isvalid_parameters_grid():
    """Test function `isvalid_parameters_grid()`."""

    parameters_correct1 = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }
    assert data_utils.isvalid_parameters_grid(parameters_correct1)

    parameters_correct2 = {
        "k": [0.001, 0.05, 1.5, 5.6],
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }
    assert data_utils.isvalid_parameters_grid(parameters_correct2)

    parameters_empty = {}

    parameters_miss_k_min = {
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_k_max = {
        "k_min": 10 ** (-3),
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_N = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_scale = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_reg_weight = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_fit_a = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "a_fixed": None,
    }

    parameters_miss_fit_fixed = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": False,
    }

    parameters_extra = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
        "n_exp": 3,
    }

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_empty)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_miss_k_min)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_miss_k_max)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_miss_N)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_miss_scale)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_miss_reg_weight)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_miss_fit_a)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_miss_fit_fixed)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_grid(parameters_extra)


def test_isvalid_parameters_n_exp():

    """Test function `isvalid_parameters_n_exp()`."""

    parameters1 = {
        "n_exp": 3,
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "fit_a": True,
        "a_fixed": None,
    }
    assert data_utils.isvalid_parameters_n_exp(parameters1)

    parameters2 = {
        "k": [0.001, 0.05, 1.5, 5.6],
        "fit_a": True,
        "a_fixed": None,
    }
    assert data_utils.isvalid_parameters_n_exp(parameters2)

    parameters_empty = {}

    parameters_miss_n_exp = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_k_min = {
        "n_exp": 3,
        "k_max": 10**1,
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_k_max = {
        "n_exp": 3,
        "k_min": 10 ** (-3),
        "fit_a": True,
        "a_fixed": None,
    }

    parameters_miss_fit_a = {
        "n_exp": 3,
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "a_fixed": None,
    }

    parameters_miss_a_fixed = {
        "n_exp": 3,
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "fit_a": False,
    }

    parameters_extra = {
        "n_exp": 3,
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "fit_a": True,
        "a_fixed": None,
        "reg_weight": 0.01,
    }

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_n_exp(parameters_empty)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_n_exp(parameters_miss_n_exp)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_n_exp(parameters_miss_k_min)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_n_exp(parameters_miss_k_max)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_n_exp(parameters_miss_fit_a)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_n_exp(parameters_miss_a_fixed)

    with pytest.raises(ValueError):
        data_utils.isvalid_parameters_n_exp(parameters_extra)
