"""
Testcases for convolution_check_config file

functions to cover:

TODO: check_required
TODO: check_convolution_instruction
TODO: check_convolution_config
"""

import unittest

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_check_config import (
    check_convolution_config,
    check_convolution_instruction,
    check_metallicity,
    check_required,
    check_sfr_dict,
)
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolution_check_config", clean_path=True
)


class test_check_convolution_config(unittest.TestCase):
    def test_check_convolution_config_with_valid_input(self):
        config_with_valid_convolution = {
            "check_convolution_config": True,
            "logger": convolution_options_defaults["logger"],
            "time_type": "lookback_time",
            "convolution_lookback_time_bin_edges": np.array([1, 2]) * u.yr,
            "convolution_instructions": [
                {
                    "input_data_type": "event",
                    "input_data_name": "event_data",
                    "output_data_name": "output_event_data",
                    "data_column_dict": {
                        "delay_time": "delay",
                        "yield_rate": "rate",
                        "metallicity": "metallicity",
                    },
                },
                {
                    "input_data_type": "ensemble",
                    "input_data_name": "ensemble_data",
                    "output_data_name": "output_ensemble_data",
                    "data_layer_dict": {
                        "delay_time": "delay",
                        "metallicity": "metallicity",
                    },
                },
            ],
            "SFR_info": [
                {
                    "name": "test",
                    "lookback_time_bin_edges": np.array([0, 1, 2, 3]) * 1e9 * u.yr,
                    "starformation_array": np.array([1, 2, 3]),
                    "metallicity_bin_edges": np.array([0.1, 0.2, 0.3]),
                    "metallicity_weighted_starformation_array": np.array(
                        [[0.5, 0.6, 0.7], [0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]
                    ),
                }
            ],
        }

        check_convolution_config(config_with_valid_convolution)
        # No exception should be raised

    def test_check_convolution_config_missing_convolution_instruction(self):

        config_with_missing_convolution_instruction = {
            "check_convolution_config": True,
            "logger": convolution_options_defaults["logger"],
            "time_type": "redshift",
            "convolution_instructions": [],  # Missing convolution instruction
            "SFR_info": [
                {
                    "lookback_time_bin_edges": [0, 1, 2, 3],
                    "starformation_array": [1, 2, 3],
                    "metallicity_bin_edges": [0.1, 0.2, 0.3],
                    "metallicity_weighted_starformation_array": [0.5, 0.6, 0.7],
                }
            ],
        }

        with self.assertRaises(ValueError):
            check_convolution_config(config_with_missing_convolution_instruction)

    def test_check_convolution_config_missing_SFR_info(self):
        #
        config_with_missing_SFR_info = {
            "check_convolution_config": True,
            "logger": convolution_options_defaults["logger"],
            "time_type": "redshift",
            "convolution_instructions": [
                {
                    "input_data_type": "event",
                    "input_data_name": "event_data",
                    "output_data_name": "output_event_data",
                    "data_column_dict": {"delay_time": "delay", "yield_rate": "rate"},
                }
            ],
            "SFR_info": [],  # Missing SFR info
        }

        with self.assertRaises(ValueError):
            check_convolution_config(config_with_missing_SFR_info)


class test_check_convolution_instruction(unittest.TestCase):
    def setUp(self):
        self.event_convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "event_data",
            "output_data_name": "output_event_data",
            "ignore_metallicity": True,
            "data_column_dict": {"delay_time": "delay", "yield_rate": "rate"},
        }

        self.ensemble_convolution_instruction = {
            "input_data_type": "ensemble",
            "input_data_name": "ensemble_data",
            "output_data_name": "output_ensemble_data",
            "ignore_metallicity": True,
            "data_layer_dict": {"delay_time": "delay"},
        }

    def test_check_convolution_instruction_event_type(self):
        check_convolution_instruction(self.event_convolution_instruction)
        # No exception should be raised

    def test_check_convolution_instruction_ensemble_type(self):
        check_convolution_instruction(self.ensemble_convolution_instruction)
        # No exception should be raised

    def test_check_convolution_instruction_missing_event_required_key(self):
        event_convolution_instruction_missing_key = {
            "input_data_type": "event",
            "input_data_name": "event_data",
            "data_column_dict": {"delay_time": "delay", "yield_rate": "rate"},
        }
        with self.assertRaises(ValueError):
            check_convolution_instruction(event_convolution_instruction_missing_key)

    def test_check_convolution_instruction_missing_ensemble_required_key(self):
        ensemble_convolution_instruction_missing_key = {
            "input_data_type": "ensemble",
            "input_data_name": "ensemble_data",
            "data_layer_dict": {"delay_time": "delay"},
        }
        with self.assertRaises(ValueError):
            check_convolution_instruction(ensemble_convolution_instruction_missing_key)

    def test_check_convolution_instruction_event_missing_metallicity(self):
        event_convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "event_data",
            "output_data_name": "output_event_data",
            "data_column_dict": {"delay_time": "delay", "yield_rate": "rate"},
        }
        with self.assertRaises(ValueError):
            check_convolution_instruction(event_convolution_instruction)

    def test_check_convolution_instruction_ensemble_missing_metallicity(self):
        ensemble_convolution_instruction = {
            "input_data_type": "ensemble",
            "input_data_name": "ensemble_data",
            "output_data_name": "output_ensemble_data",
            "data_layer_dict": {"delay_time": "delay"},
        }

        with self.assertRaises(ValueError):
            check_convolution_instruction(ensemble_convolution_instruction)


class test_check_metallicity(unittest.TestCase):
    def setUp(self):
        self.convolution_instruction_with_metallicity = {
            "data_column_dict": {"metallicity": "Fe/H"}
        }
        self.convolution_instruction_with_metallicity_value = {
            "metallicity_value": "0.0"
        }
        self.convolution_instruction_with_ignore_metallicity = {
            "ignore_metallicity": True
        }
        self.convolution_instruction_missing_metallicity = {
            "data_column_dict": {"no_metallicity_key": "some_value"}
        }

    def test_check_metallicity_with_metallicity(self):
        data_key = "data_column_dict"
        check_metallicity(self.convolution_instruction_with_metallicity, data_key)
        # No exception should be raised

    def test_check_metallicity_with_metallicity_value(self):
        data_key = "data_column_dict"
        check_metallicity(self.convolution_instruction_with_metallicity_value, data_key)
        # No exception should be raised

    def test_check_metallicity_with_ignore_metallicity(self):
        data_key = "data_column_dict"
        check_metallicity(
            self.convolution_instruction_with_ignore_metallicity, data_key
        )
        # No exception should be raised

    def test_check_metallicity_missing_metallicity(self):
        data_key = "data_column_dict"
        with self.assertRaises(ValueError):
            check_metallicity(
                self.convolution_instruction_missing_metallicity, data_key
            )


class test_check_required(unittest.TestCase):
    def setUp(self):
        self.config = {
            "input_shape": (32, 32, 3),
            "output_shape": (10,),
            "learning_rate": 0.001,
        }

    def test_check_required_all_present(self):
        required_list = ["input_shape", "output_shape", "learning_rate"]
        check_required(self.config, required_list)
        # No exception should be raised

    def test_check_required_missing_key(self):
        required_list = ["input_shape", "output_shape", "learning_rate", "batch_size"]
        with self.assertRaises(ValueError):
            check_required(self.config, required_list)

    def test_check_required_empty_list(self):
        required_list = []
        check_required(self.config, required_list)
        # No exception should be raised


class test_check_sfr_dict(unittest.TestCase):
    def setUp(self):
        self.sfr_dict = {
            "name": "test_sfr_dict",
            "lookback_time_bin_edges": np.array([1, 2, 3]) * 1e9 * u.yr,
            "starformation_array": np.array([10, 20, 30]),
            "metallicity_bin_edges": np.array([0.01, 0.1, 0.2]),
            "metallicity_weighted_starformation_array": np.array(
                [[1, 2, 3], [4, 5, 6]]
            ),
        }

    def test_check_sfr_dict_with_name(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"
        check_sfr_dict(
            self.sfr_dict, requires_name, requires_metallicity_info, time_type
        )
        # No exception should be raised

    def test_check_sfr_dict_without_name(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"
        del self.sfr_dict["name"]  # Removing the name key
        with self.assertRaises(ValueError):
            check_sfr_dict(
                self.sfr_dict, requires_name, requires_metallicity_info, time_type
            )

    def test_check_sfr_dict_without_metallicity_info(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"
        del self.sfr_dict[
            "metallicity_bin_edges"
        ]  # Removing the metallicity_bin_edges key
        with self.assertRaises(ValueError):
            check_sfr_dict(
                self.sfr_dict, requires_name, requires_metallicity_info, time_type
            )

    def test_check_sfr_dict_without_time_type_info(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"
        del self.sfr_dict[
            "lookback_time_bin_edges"
        ]  # Removing the lookback_time_bin_edges key
        with self.assertRaises(ValueError):
            check_sfr_dict(
                self.sfr_dict, requires_name, requires_metallicity_info, time_type
            )

    def test_check_sfr_dict_without_lookback_time_unit(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"
        self.sfr_dict["lookback_time_bin_edges"] = np.array([1, 2, 3]) * 1e9

        with self.assertRaises(AttributeError):
            check_sfr_dict(
                self.sfr_dict, requires_name, requires_metallicity_info, time_type
            )

    def test_check_sfr_dict_wrong_lookback_time_unit(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"
        self.sfr_dict["lookback_time_bin_edges"] = np.array([1, 2, 3]) * 1e9 * u.ms

        with self.assertRaises(ValueError):
            check_sfr_dict(
                self.sfr_dict, requires_name, requires_metallicity_info, time_type
            )

    def test_check_sfr_dict_redshift_wrong_bin_edges(self):
        # self.sfr_dict['lookback_time_bin_edges']

        # self.sfr_dict = {
        #     "name": "test_sfr_dict",
        #     "lookback_time_bin_edges": np.array([1, 2, 3]) * 1e9 * u.yr,
        #     "starformation_array": np.array([10, 20, 30]),
        #     "metallicity_bin_edges": np.array([0.01, 0.1, 0.2]),
        #     "metallicity_weighted_starformation_array": np.array(
        #         [[1, 2, 3], [4, 5, 6]]
        #     ),
        # }

        requires_name = True
        requires_metallicity_info = True
        time_type = "redshift"

        with self.assertRaises(ValueError):
            check_sfr_dict(
                self.sfr_dict, requires_name, requires_metallicity_info, time_type
            )

    def test_check_sfr_dict_redshift_correct_bin_edges(self):
        self.sfr_dict["redshift_bin_edges"] = np.array([0, 1, 2])

        requires_name = True
        requires_metallicity_info = True
        time_type = "redshift"

        check_sfr_dict(
            self.sfr_dict, requires_name, requires_metallicity_info, time_type
        )


if __name__ == "__main__":
    unittest.main()
