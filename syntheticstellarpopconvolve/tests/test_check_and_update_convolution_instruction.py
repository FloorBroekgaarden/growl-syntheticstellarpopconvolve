"""
This is the unittest file for the check_and_update_convolution_instruction.py source file
"""

import unittest

from syntheticstellarpopconvolve import default_convolution_config
from syntheticstellarpopconvolve.check_and_update_convolution_instruction import (
    check_and_update_convolution_instructions,
    check_convolution_instruction,
    check_metallicity,
)


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


class test_check_convolution_instruction(unittest.TestCase):
    def setUp(self):
        self.event_convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "event_data",
            "output_data_name": "output_event_data",
            "convolution_type": "integrate",
            "ignore_metallicity": True,
            "data_column_dict": {"delay_time": "delay", "normalized_yield": "rate"},
        }

        self.ensemble_convolution_instruction = {
            "input_data_type": "ensemble",
            "input_data_name": "ensemble_data",
            "output_data_name": "output_ensemble_data",
            "convolution_type": "integrate",
            "ignore_metallicity": True,
            "data_layer_dict": {"delay_time": "delay"},
        }

        self.config = default_convolution_config

    def test_check_convolution_instruction_event_type(self):
        check_convolution_instruction(
            convolution_instruction=self.event_convolution_instruction,
            config=self.config,
        )
        # No exception should be raised

    def test_check_convolution_instruction_ensemble_type(self):
        check_convolution_instruction(
            convolution_instruction=self.ensemble_convolution_instruction,
            config=self.config,
        )
        # No exception should be raised

    def test_check_convolution_instruction_missing_event_required_key(self):
        event_convolution_instruction_missing_key = {
            "input_data_type": "event",
            "input_data_name": "event_data",
            "data_column_dict": {"delay_time": "delay", "normalized_yield": "rate"},
        }
        with self.assertRaises(ValueError):
            check_convolution_instruction(
                convolution_instruction=event_convolution_instruction_missing_key,
                config=self.config,
            )

    def test_check_convolution_instruction_missing_ensemble_required_key(self):
        ensemble_convolution_instruction_missing_key = {
            "input_data_type": "ensemble",
            "convolution_type": "integrate",
            "input_data_name": "ensemble_data",
            "data_layer_dict": {"delay_time": "delay"},
        }
        with self.assertRaises(ValueError):
            check_convolution_instruction(
                convolution_instruction=ensemble_convolution_instruction_missing_key,
                config=self.config,
            )

    def test_check_convolution_instruction_event_missing_metallicity(self):
        event_convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "event_data",
            "output_data_name": "output_event_data",
            "convolution_type": "integrate",
            "data_column_dict": {"delay_time": "delay", "normalized_yield": "rate"},
        }
        with self.assertRaises(ValueError):
            check_convolution_instruction(
                convolution_instruction=event_convolution_instruction,
                config=self.config,
            )

    def test_check_convolution_instruction_ensemble_missing_metallicity(self):
        ensemble_convolution_instruction = {
            "input_data_type": "ensemble",
            "input_data_name": "ensemble_data",
            "output_data_name": "output_ensemble_data",
            "convolution_type": "integrate",
            "data_layer_dict": {"delay_time": "delay"},
        }

        with self.assertRaises(ValueError):
            check_convolution_instruction(
                convolution_instruction=ensemble_convolution_instruction,
                config=self.config,
            )


class test_check_and_update_convolution_instructions(unittest.TestCase):

    def test_check_and_update_convolution_instructions_no_convolution_instructions(
        self,
    ):

        config = {}

        with self.assertRaises(ValueError):
            check_and_update_convolution_instructions(config=config)


if __name__ == "__main__":
    unittest.main()
