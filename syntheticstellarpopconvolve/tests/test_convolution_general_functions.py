"""
Testcases for convolution_general_functions file
"""

import copy
import json
import logging
import os
import unittest

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import pkg_resources

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_general_functions import (
    calculate_bincenters,
    calculate_digitized_sfr_rates,
    calculate_edge_values,
    calculate_origin_time_array,
    extract_arguments,
    generate_group_name,
    get_tmp_dir,
    handle_custom_scaling_or_conversion,
    handle_extra_weights_function,
    pad_function,
    temp_dir,
)
from syntheticstellarpopconvolve.convolve_populations import update_sfr_dict
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file
from syntheticstellarpopconvolve.prepare_redshift_interpolator import (
    prepare_redshift_interpolator,
)

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolution_general_functions", clean_path=True
)


class test_calculate_digitized_sfr_rates(unittest.TestCase):
    def setUp(self):
        #
        input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        ##############
        # SET UP DATA
        self.dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]) * u.yr,
            "probability": np.array([1, 2, 3, 4]),
        }
        dummy_df = pd.DataFrame.from_records(self.dummy_data)

        #############
        # create input HDF5 file
        with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

            ######################
            # Create groups
            input_hdf5_file.create_group("input_data")
            input_hdf5_file.create_group("input_data/events")
            input_hdf5_file.create_group("config")

            ###############
            # Readout population settings
            population_settings_filename = pkg_resources.resource_filename(
                "syntheticstellarpopconvolve",
                "example_data/example_population_settings.json",
            )

            with open(population_settings_filename, "r") as f:
                population_settings = json.loads(f.read())

            # Delete some stuff from the settings
            del population_settings["population_settings"]["bse_options"]["metallicity"]

            # Write population config to file
            input_hdf5_file.create_dataset(
                "config/population", data=json.dumps(population_settings)
            )

        ##############
        # Store data in pandas
        dummy_df.to_hdf(input_hdf5_filename, key="input_data/events/{}".format("dummy"))

        #
        self.convolution_config = copy.copy(convolution_options_defaults)

        # Set up SFR
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * 1e9 * u.yr,
            "starformation_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr / u.Gpc**3,
        }

        # set up convolution bins
        self.convolution_config["convolution_time_bin_edges"] = (
            np.array([0, 1, 2, 3, 4]) * 1e9 * u.yr
        )

        # lookback time convolution only
        self.convolution_config["time_type"] = "lookback_time"

        #
        self.convolution_config["input_filename"] = input_hdf5_filename
        self.convolution_config["output_filename"] = output_hdf5_filename

        self.convolution_config["redshift_interpolator_data_output_filename"] = (
            os.path.join(TMP_DIR, "interpolator_dict.p")
        )

        #
        self.convolution_config["convolution_instructions"] = [
            {
                "input_data_type": "event",
                "input_data_name": "dummy",
                "output_data_name": "dummy",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "yield_rate": "probability",
                },
                "ignore_metallicity": True,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        prepare_output_file(config=self.convolution_config)

    def test_calculate_digitized_sfr_rates_sfr_only(self):

        #
        sfr_dict = update_sfr_dict(
            sfr_dict=self.convolution_config["SFR_info"], config=self.convolution_config
        )

        digitized_sfr_rates = calculate_digitized_sfr_rates(
            config=self.convolution_config,
            convolution_time_bin_center=0.5 * 1e9 * u.yr,
            data_dict={"delay_time": np.array([-1, 1, 2, 3, 100]) * 1e9 * u.yr},
            sfr_dict=sfr_dict,
        )

        np.testing.assert_array_equal(
            digitized_sfr_rates, np.array([0.0, 2.0, 3.0, 4.0, 0.0])
        )

    # def test_calculate_digitized_sfr_rates_metallicity(self):

    #     self.convolution_config["SFR_info"]["metallicity_bin_edges"] = (
    #         self.convolution_config["SFR_info"]["starformation_array"]
    #         * np.ones(
    #             (self.convolution_config["SFR_info"]["starformation_array"].shape[0], 3)
    #         ).T
    #     )

    #     # print(self.convolution_config["SFR_info"]["metallicity_bin_edges"])

    #     #
    #     sfr_dict = update_sfr_dict(
    #         sfr_dict=self.convolution_config["SFR_info"], config=self.convolution_config
    #     )

    #     digitized_sfr_rates = calculate_digitized_sfr_rates(
    #         config=self.convolution_config,
    #         convolution_time_bin_center=0.5 * 1e9 * u.yr,
    #         data_dict={"delay_time": np.array([-1, 1, 2, 3, 100]) * 1e9},
    #         sfr_dict=sfr_dict,
    #     )

    #     np.testing.assert_array_equal(
    #         digitized_sfr_rates, np.array([0.0, 2.0, 3.0, 4.0, 0.0])
    #     )


class test_calculate_origin_time_array(unittest.TestCase):
    def test_calculate_origin_time_array_lookback(self):
        convolution_config = copy.copy(convolution_options_defaults)
        convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
            TMP_DIR, "interpolator_dict.p"
        )
        convolution_config = prepare_redshift_interpolator(convolution_config)
        convolution_config["time_type"] = "lookback_time"

        origin_time_array = calculate_origin_time_array(
            config=convolution_config,
            data_dict={"delay_time": np.array([1, 2, 3]) * 1e9 * u.yr},
            convolution_time_bin_center=0.5 * 1e9,
        )
        np.testing.assert_array_equal(
            origin_time_array, np.array([1.5, 2.5, 3.5]) * 1e9
        )

    def test_calculate_origin_time_array_redshift(self):
        convolution_config = copy.copy(convolution_options_defaults)
        convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
            TMP_DIR, "interpolator_dict.p"
        )
        convolution_config = prepare_redshift_interpolator(convolution_config)
        convolution_config["time_type"] = "redshift"

        origin_time_array = calculate_origin_time_array(
            config=convolution_config,
            data_dict={"delay_time": np.array([1, 2, 3]) * 1e9 * u.yr},
            convolution_time_bin_center=0.5,
        )
        np.testing.assert_array_almost_equal(
            origin_time_array,
            np.array([0.6501032923316669, 0.8336451543045214, 1.0661079791875108]),
        )


class test_handle_extra_weights_function(unittest.TestCase):
    def setUp(self):
        #
        input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        ##############
        # SET UP DATA
        self.dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]),
            "probability": np.array([1, 2, 3, 4]),
        }
        dummy_df = pd.DataFrame.from_records(self.dummy_data)

        #############
        # create input HDF5 file
        with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

            ######################
            # Create groups
            input_hdf5_file.create_group("input_data")
            input_hdf5_file.create_group("input_data/events")
            input_hdf5_file.create_group("config")

            ###############
            # Readout population settings
            population_settings_filename = pkg_resources.resource_filename(
                "syntheticstellarpopconvolve",
                "example_data/example_population_settings.json",
            )

            with open(population_settings_filename, "r") as f:
                population_settings = json.loads(f.read())

            # Delete some stuff from the settings
            del population_settings["population_settings"]["bse_options"]["metallicity"]

            # Write population config to file
            input_hdf5_file.create_dataset(
                "config/population", data=json.dumps(population_settings)
            )

        ##############
        # Store data in pandas
        dummy_df.to_hdf(input_hdf5_filename, key="input_data/events/{}".format("dummy"))

        #
        self.convolution_config = copy.copy(convolution_options_defaults)

        # Set up SFR
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]),
            "starformation_array": np.array([1, 1, 1, 1, 1]) * u.Msun / u.yr / u.Gpc**3,
        }

        # set up convolution bins
        self.convolution_config["convolution_time_bin_edges"] = np.array(
            [0, 1, 2, 3, 4]
        )

        # lookback time convolution only
        self.convolution_config["time_type"] = "lookback_time"

        #
        self.convolution_config["input_filename"] = input_hdf5_filename
        self.convolution_config["output_filename"] = output_hdf5_filename

        self.convolution_config["redshift_interpolator_data_output_filename"] = (
            os.path.join(TMP_DIR, "interpolator_dict.p")
        )

        #
        self.convolution_config["convolution_instructions"] = [
            {
                "input_data_type": "event",
                "input_data_name": "dummy",
                "output_data_name": "dummy",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "yield_rate": "probability",
                },
                "ignore_metallicity": True,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        prepare_output_file(config=self.convolution_config)

    def test_handle_extra_weights_function_normal(self):
        def extra_weights_function(config, data_dict):
            return np.zeros(data_dict["yield_rate"].shape)

        convolution_instruction = self.convolution_config["convolution_instructions"][0]
        convolution_instruction["extra_weights_function"] = extra_weights_function

        self.dummy_data["yield_rate"] = self.dummy_data["probability"]

        extra_weights = handle_extra_weights_function(
            config=self.convolution_config,
            convolution_time_bin_center=0.2,
            convolution_instruction=convolution_instruction,
            data_dict=self.dummy_data,
            output_shape=self.dummy_data["yield_rate"].shape,
        )

        #
        np.testing.assert_array_equal(
            extra_weights, np.zeros(self.dummy_data["yield_rate"].shape)
        )

    def test_handle_extra_weights_function_extra_input_fail(self):
        # test should fail since we don't provide the input for the function
        def extra_weights_function(config, data_dict, a, b):
            return np.zeros(data_dict["yield_rate"].shape) + a + b

        convolution_instruction = self.convolution_config["convolution_instructions"][0]
        convolution_instruction["extra_weights_function"] = extra_weights_function

        #
        self.dummy_data["yield_rate"] = self.dummy_data["probability"]

        with self.assertRaises(KeyError):
            _ = handle_extra_weights_function(
                config=self.convolution_config,
                convolution_time_bin_center=0.2,
                convolution_instruction=convolution_instruction,
                data_dict=self.dummy_data,
                output_shape=self.dummy_data["yield_rate"].shape,
            )

    def test_handle_extra_weights_function_extra_function_fail(self):
        # test should fail since the function does not return anything
        def extra_weights_function(config, data_dict, a, b):
            pass

        convolution_instruction = self.convolution_config["convolution_instructions"][0]
        convolution_instruction["extra_weights_function"] = extra_weights_function
        convolution_instruction["extra_weights_function_additional_parameters"] = {
            "a": 10,
            "b": 2.5,
        }

        #
        self.dummy_data["yield_rate"] = self.dummy_data["probability"]

        with self.assertRaises(ValueError):
            _ = handle_extra_weights_function(
                config=self.convolution_config,
                convolution_time_bin_center=0.2,
                convolution_instruction=convolution_instruction,
                data_dict=self.dummy_data,
                output_shape=self.dummy_data["yield_rate"].shape,
            )

    def test_handle_extra_weights_function_extra_input_pass(self):
        # test should fail since we don't provide the input for the function
        def extra_weights_function(config, data_dict, a, b):
            return np.zeros(data_dict["yield_rate"].shape) + a + b

        convolution_instruction = self.convolution_config["convolution_instructions"][0]
        convolution_instruction["extra_weights_function"] = extra_weights_function
        convolution_instruction["extra_weights_function_additional_parameters"] = {
            "a": 10,
            "b": 2.5,
        }

        #
        self.dummy_data["yield_rate"] = self.dummy_data["probability"]

        extra_weights = handle_extra_weights_function(
            config=self.convolution_config,
            convolution_time_bin_center=0.2,
            convolution_instruction=convolution_instruction,
            data_dict=self.dummy_data,
            output_shape=self.dummy_data["yield_rate"].shape,
        )

        np.testing.assert_array_equal(
            extra_weights, np.zeros(self.dummy_data["yield_rate"].shape) + 12.5
        )

    def test_handle_extra_weights_function_no_function_shape_fail(self):
        def extra_weights_function(config, data_dict, a, b):
            return np.zeros(data_dict["yield_rate"].shape) + a + b

        # test should fail since the output shape doesnt match
        convolution_instruction = self.convolution_config["convolution_instructions"][0]
        convolution_instruction["extra_weights_function"] = extra_weights_function
        convolution_instruction["extra_weights_function_additional_parameters"] = {
            "a": 10,
            "b": 2.5,
        }

        #
        self.dummy_data["yield_rate"] = self.dummy_data["probability"]

        with self.assertRaises(ValueError):
            _ = handle_extra_weights_function(
                config=self.convolution_config,
                convolution_time_bin_center=0.2,
                convolution_instruction=convolution_instruction,
                data_dict=self.dummy_data,
                output_shape=np.shape([1]),
            )

    def test_handle_extra_weights_function_no_function(self):
        # test should fail since the output shape doesnt match
        convolution_instruction = self.convolution_config["convolution_instructions"][0]

        #
        self.dummy_data["yield_rate"] = self.dummy_data["probability"]

        extra_weights = handle_extra_weights_function(
            config=self.convolution_config,
            convolution_time_bin_center=0.2,
            convolution_instruction=convolution_instruction,
            data_dict=self.dummy_data,
            output_shape=np.shape([1]),
        )

        #
        np.testing.assert_array_equal(extra_weights, np.ones(np.shape([1])))


class test_handle_custom_scaling_or_conversion(unittest.TestCase):
    def setUp(self):
        self.data_layer_dict = {
            "factor": {"layer_depth": 2, "conversion_factor": 2},
            "function": {"layer_depth": 4, "conversion_function": lambda x: x**2},
            "both": {
                "layer_depth": 4,
                "conversion_function": lambda x: x**2,
                "conversion_factor": 2,
            },
        }

        self.logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s ] %(asctime)s: %(message)s"
        logging.basicConfig(format=FORMAT)
        self.logger.setLevel(logging.INFO)

    def test_factor_array(self):
        array = handle_custom_scaling_or_conversion(
            config={"logger": self.logger},
            data_layer_or_column_dict_entry=self.data_layer_dict["factor"],
            value=np.array([1, 2]),
        )
        np.testing.assert_array_equal(array, np.array([2, 4]))

    def test_function_array(self):
        array = handle_custom_scaling_or_conversion(
            config={"logger": self.logger},
            data_layer_or_column_dict_entry=self.data_layer_dict["function"],
            value=np.array([1, 2]),
        )
        np.testing.assert_array_equal(array, np.array([1, 4]))

    def test_factor_scalar(self):
        value = handle_custom_scaling_or_conversion(
            config={"logger": self.logger},
            data_layer_or_column_dict_entry=self.data_layer_dict["factor"],
            value=1,
        )
        self.assertEqual(value, 2)

    def test_function_scalar(self):
        value = handle_custom_scaling_or_conversion(
            config={"logger": self.logger},
            data_layer_or_column_dict_entry=self.data_layer_dict["function"],
            value=2,
        )
        self.assertEqual(value, 4)

    def test_get_deepest_data_layer_depth_both(self):
        with self.assertRaises(ValueError):
            handle_custom_scaling_or_conversion(
                config={"logger": self.logger},
                data_layer_or_column_dict_entry=self.data_layer_dict["both"],
                value=2,
            )


class test_extract_arguments(unittest.TestCase):
    def test_extract_arguments_1_extra(self):
        def funca(a, b):
            pass

        args = extract_arguments(funca, {"a": 2, "b": 3, "c": 4})
        self.assertEqual(args, {"a": 2, "b": 3})

    def test_extract_arguments_exact(self):
        def funca(a, b):
            pass

        args = extract_arguments(funca, {"a": 2, "b": 3})
        self.assertEqual(args, {"a": 2, "b": 3})

    def test_extract_arguments_default_args_only(self):
        def funcb(a, b, c=3):
            pass

        args = extract_arguments(funcb, {"a": 2, "b": 3})
        self.assertEqual(args, {"a": 2, "b": 3})

    def test_extract_arguments_default_all(self):
        def funcb(a, b, c=3):
            pass

        args = extract_arguments(funcb, {"a": 2, "b": 3, "c": 4})
        self.assertEqual(args, {"a": 2, "b": 3, "c": 4})

    def test_extract_arguments_default_1_extra(self):
        def funcb(a, b, c=3):
            pass

        args = extract_arguments(funcb, {"a": 2, "b": 3, "c": 4, "d": 5})
        self.assertEqual(args, {"a": 2, "b": 3, "c": 4})

    def test_extract_arguments_missing(self):
        def funca(a, b):
            pass

        with self.assertRaises(KeyError):
            extract_arguments(funca, {"a": 2})


class test_calculate_bincenters(unittest.TestCase):
    def test_calculate_bincenters_linear(self):
        array = np.array([1.0, 2, 3, 4, 5])
        expected_bincenters = np.array([1.5, 2.5, 3.5, 4.5])
        bincenters = calculate_bincenters(array, convert="linear")
        np.testing.assert_array_equal(bincenters, expected_bincenters)


class test_calculate_edge_values(unittest.TestCase):
    def test_calculate_edge_values(self):
        arr = np.array([1.0, 2, 3, 4, 5])
        expected_edge_values = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        edge_values = calculate_edge_values(arr)
        np.testing.assert_array_equal(edge_values, expected_edge_values)


class test_pad_function(unittest.TestCase):
    def test_pad_function_relative_to_edge_val_axis_0(self):
        array = np.array([1.0, 2, 3, 4, 5])
        left_val = -0.5
        right_val = 0.5
        relative_to_edge_val = True
        expected_padded_array = np.array([0.5, 1, 2, 3, 4, 5, 5.5])
        padded_array = pad_function(
            array, left_val, right_val, relative_to_edge_val, axis=0
        )
        np.testing.assert_array_equal(padded_array, expected_padded_array)

    def test_pad_function_absolute_axis_1(self):
        array = np.array([[1.0, 2, 3], [4, 5, 6]])
        left_val = 0
        right_val = 0
        relative_to_edge_val = False
        expected_padded_array = np.array([[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]])
        padded_array = pad_function(
            array, left_val, right_val, relative_to_edge_val, axis=1
        )
        np.testing.assert_array_equal(padded_array, expected_padded_array)


class test_generate_group_name(unittest.TestCase):
    def setUp(self):
        self.convolution_instruction = {
            "input_data_type": "image",
            "input_data_name": "input_image",
            "output_data_name": "output_image",
        }
        self.sfr_dict = {"name": "test_group"}

    def test_generate_group_name_with_sfr(self):
        groupname, elements = generate_group_name(
            self.convolution_instruction, self.sfr_dict
        )
        expected_groupname = "test_group/image/input_image/output_image"
        expected_elements = ["test_group", "image", "input_image", "output_image"]
        self.assertEqual(groupname, expected_groupname)
        self.assertListEqual(elements, expected_elements)

    def test_generate_group_name_without_sfr(self):
        groupname, elements = generate_group_name(self.convolution_instruction, {})
        expected_groupname = "image/input_image/output_image"
        expected_elements = ["image", "input_image", "output_image"]
        self.assertEqual(groupname, expected_groupname)
        self.assertListEqual(elements, expected_elements)


class test_get_tmp_dir(unittest.TestCase):
    def setUp(self):
        self.convolution_instruction = {
            "input_data_type": "image",
            "input_data_name": "input_image",
            "output_data_name": "output_image",
        }

    def test_get_tmp_dir(self):
        tmp_dir = get_tmp_dir(
            config={"tmp_dir": TMP_DIR},
            convolution_instruction=self.convolution_instruction,
        )
        self.assertEqual(
            tmp_dir, os.path.join(TMP_DIR, "image/input_image/output_image")
        )


if __name__ == "__main__":
    unittest.main()
