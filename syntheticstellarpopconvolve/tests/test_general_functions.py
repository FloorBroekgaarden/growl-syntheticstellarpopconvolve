"""
Testcases for general_functions file
"""

import copy
import json
import logging
import os
import tempfile
import unittest

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import pkg_resources

from syntheticstellarpopconvolve import (
    default_convolution_config,
    default_convolution_instruction,
)
from syntheticstellarpopconvolve.check_and_update_convolution_config import (
    check_and_update_convolution_config,
)
from syntheticstellarpopconvolve.general_functions import (
    calculate_bin_edges,
    calculate_bincenters,
    calculate_digitized_sfr_rates,
    calculate_origin_time_array,
    check_required,
    generate_group_name,
    get_tmp_dir,
    get_username,
    handle_custom_scaling_or_conversion,
    has_unit,
    is_time_unit,
    pad_function,
    temp_dir,
)
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file
from syntheticstellarpopconvolve.prepare_redshift_interpolator import (
    prepare_redshift_interpolator,
)

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_general_functions", clean_path=True
)


class test_is_time_unit(unittest.TestCase):
    """ """

    def test_is_time_unit(self):
        time_unit_value = 1 * u.s

        self.assertTrue(is_time_unit(time_unit_value))

    def test_is_not_time_unit(self):
        no_unit_value = 1
        self.assertFalse(is_time_unit(no_unit_value))

    def test_is_unit_but_not_time_unit(self):
        wrong_unit_value = 1 * u.m
        self.assertFalse(is_time_unit(wrong_unit_value))


class test_has_unit(unittest.TestCase):
    """ """

    def test_unit(self):
        unit_value = 1 * u.m

        self.assertTrue(has_unit(unit_value))

    def test_no_unit(self):
        no_unit_value = 1

        self.assertFalse(has_unit(no_unit_value))

    def test_dimensionless_unit(self):

        dimensionless_unit = u.m / u.m

        dimensionless_value = 1 * dimensionless_unit

        self.assertTrue(has_unit(dimensionless_value, fail_on_dimensionless=False))

        self.assertFalse(has_unit(dimensionless_value, fail_on_dimensionless=True))


class test_get_username(unittest.TestCase):
    """ """

    def test_get_username(self):
        username = get_username()

        # should be a string
        self.assertTrue(isinstance(username, str))

        # should be of some lenght
        self.assertTrue(len(username) > 0)


class test_temp_dir(unittest.TestCase):
    """
    Unittests for temp_dir
    """

    def test_create_temp_dir(self):
        """
        Test making a temp directory and comparing that to what it should be
        """

        #
        username = get_username()
        general_temp_dir = tempfile.gettempdir()

        # Get username
        username = get_username()
        sspc_temp_dir = os.path.join(temp_dir())

        #
        self.assertTrue(
            os.path.isdir(os.path.join(general_temp_dir, "sspc-{}".format(username)))
        )
        self.assertTrue(
            os.path.join(general_temp_dir, "sspc-{}".format(username)) == sspc_temp_dir
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
        dummy_df.to_hdf(input_hdf5_filename, key="input_data/{}".format("dummy"))

        #
        self.convolution_config = copy.copy(default_convolution_config)

        # Set up SFR
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * 1e9 * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5])
            * u.Msun
            / u.yr
            / u.Gpc**3,
        }

        # set up convolution bins
        self.convolution_config["convolution_lookback_time_bin_edges"] = (
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
                **default_convolution_instruction,
                "input_data_name": "dummy",
                "output_data_name": "dummy",
                "convolution_type": "integrate",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "normalized_yield": "probability",
                },
                "ignore_metallicity": True,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        check_and_update_convolution_config(self.convolution_config)

        #
        prepare_output_file(config=self.convolution_config)

    def test_calculate_digitized_sfr_rates_sfr_only(self):

        digitized_sfr_rates = calculate_digitized_sfr_rates(
            config=self.convolution_config,
            convolution_time_bin_center=0.5 * 1e9 * u.yr,
            data_dict={"delay_time": np.array([-1, 1, 2, 3, 100]) * 1e9 * u.yr},
            sfr_dict=self.convolution_config["SFR_info"],
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
        )
        output_unit = u.Msun / u.yr / u.Gpc**3

        np.testing.assert_array_equal(
            digitized_sfr_rates, np.array([0.0, 2.0, 3.0, 4.0, 0.0]) * output_unit
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
        convolution_config = copy.copy(default_convolution_config)
        convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
            TMP_DIR, "interpolator_dict.p"
        )
        convolution_config = prepare_redshift_interpolator(convolution_config)
        convolution_config["time_type"] = "lookback_time"

        origin_time_array = calculate_origin_time_array(
            config=convolution_config,
            data_dict={"delay_time": np.array([1, 2, 3]) * 1e9 * u.yr},
            convolution_time_bin_center=0.5 * 1e9 * u.yr,
        )

        np.testing.assert_array_equal(
            origin_time_array, np.array([1.5, 2.5, 3.5]) * 1e9 * u.yr
        )

    def test_calculate_origin_time_array_redshift(self):
        convolution_config = copy.copy(default_convolution_config)
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
        # output_unit = u.Msun/u.yr/u.Gpc**3

        np.testing.assert_array_almost_equal(
            origin_time_array,
            np.array([0.6501032923316669, 0.8336451543045214, 1.0661079791875108]),
        )


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


class test_calculate_bincenters(unittest.TestCase):
    def test_calculate_bincenters_linear(self):
        array = np.array([1.0, 2, 3, 4, 5])
        expected_bincenters = np.array([1.5, 2.5, 3.5, 4.5])
        bincenters = calculate_bincenters(array, convert="linear")
        np.testing.assert_array_equal(bincenters, expected_bincenters)


class test_calculate_bin_edges(unittest.TestCase):
    def test_calculate_bin_edges(self):
        arr = np.array([1.0, 2, 3, 4, 5])
        expected_bin_edges = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        bin_edges = calculate_bin_edges(arr)
        np.testing.assert_array_equal(bin_edges, expected_bin_edges)


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
            "input_data_name": "input_image",
            "output_data_name": "output_image",
        }
        self.sfr_dict = {"name": "test_group"}

    def test_generate_group_name_with_sfr(self):
        groupname, elements = generate_group_name(
            self.convolution_instruction, self.sfr_dict
        )
        expected_groupname = "test_group/input_image/output_image"
        expected_elements = ["test_group", "input_image", "output_image"]
        self.assertEqual(groupname, expected_groupname)
        self.assertListEqual(elements, expected_elements)

    def test_generate_group_name_without_sfr(self):
        groupname, elements = generate_group_name(self.convolution_instruction, {})
        expected_groupname = "input_image/output_image"
        expected_elements = ["input_image", "output_image"]
        self.assertEqual(groupname, expected_groupname)
        self.assertListEqual(elements, expected_elements)


class test_get_tmp_dir(unittest.TestCase):
    def setUp(self):
        self.convolution_instruction = {
            "input_data_name": "input_image",
            "output_data_name": "output_image",
        }

    def test_get_tmp_dir(self):
        tmp_dir = get_tmp_dir(
            config={"tmp_dir": TMP_DIR},
            convolution_instruction=self.convolution_instruction,
        )
        self.assertEqual(tmp_dir, os.path.join(TMP_DIR, "input_image/output_image"))


if __name__ == "__main__":
    unittest.main()
