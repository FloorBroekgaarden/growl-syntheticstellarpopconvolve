"""
Unit tests for convolution on the fly
"""

import copy
import os
import unittest

import astropy.units as u
import h5py
import numpy as np

from syntheticstellarpopconvolve import convolve, default_convolution_config
from syntheticstellarpopconvolve.check_and_update_convolution_config import (
    check_and_update_convolution_config,
)
from syntheticstellarpopconvolve.convolve_on_the_fly import convolve_on_the_fly
from syntheticstellarpopconvolve.general_functions import temp_dir
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "test_convolution_on_the_fly", clean_path=True
)


def dummy_on_the_fly_function(total_star_formation_in_bin):
    """
    on-the-fly function that should work
    """

    return {}


def wrong_arguments_on_the_fly_function():
    """
    on-the-fly function does not have the correct arguments
    """

    return {}


def wrong_return_type_on_the_fly_function(total_star_formation_in_bin):
    """
    On-the-fly function that returns the wrong type of object (None in this case)
    """


def metallicity_required_not_included_on_the_fly_function(total_star_formation_in_bin):
    """
    On-the-fly function that returns the wrong type of object (None in this case)
    """


class test_convolve_on_the_fly(unittest.TestCase):
    """ """

    def setUp(self):
        #
        input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        #############
        # create input HDF5 file
        with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

            ######################
            # Create groups
            input_hdf5_file.create_group("input_data")
            input_hdf5_file.create_group("input_data/events")
            input_hdf5_file.create_group("config")

        #
        self.convolution_config = copy.copy(default_convolution_config)

        # Set up SFR
        self.convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 1, 1, 1, 1]) * u.Msun / u.yr,
        }

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
                "input_data_name": "binary_c",
                "output_data_name": "BHBH",
                "convolution_type": "on-the-fly",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "normalized_yield": "probability",
                },
                "ignore_metallicity": True,
                "on_the_fly_function": dummy_on_the_fly_function,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        check_and_update_convolution_config(self.convolution_config)

        #
        prepare_output_file(config=self.convolution_config)

    def test_convolve_on_the_fly_wrong_arguments_on_the_fly_function(self):
        #
        self.convolution_config["convolution_instructions"] = [
            {
                "input_data_name": "binary_c",
                "output_data_name": "BHBH",
                "convolution_type": "on-the-fly",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "normalized_yield": "probability",
                },
                "ignore_metallicity": True,
                "on_the_fly_function": wrong_arguments_on_the_fly_function,
            },
        ]

        #
        sfr_dict = self.convolution_config["SFR_info"]

        time_bin_info_dict = {
            "bin_number": 0,
            "bin_center": 0.5 * u.yr,
            "bin_edge_lower": 0,
            "bin_size": 1 * u.yr,
            "bin_type": "starformation time",
            "time_type": self.convolution_config["time_type"],
        }

        with self.assertRaises(ValueError):
            convolve_on_the_fly(
                config=self.convolution_config,
                sfr_dict=sfr_dict,
                convolution_instruction=self.convolution_config[
                    "convolution_instructions"
                ][0],
                time_bin_info_dict=time_bin_info_dict,
            )

    def test_convolve_on_the_fly_normal(self):
        #
        sfr_dict = self.convolution_config["SFR_info"]

        time_bin_info_dict = {
            "bin_number": 0,
            "bin_center": 0.5 * u.yr,
            "bin_edge_lower": 0,
            "bin_size": 1 * u.yr,
            "bin_type": "starformation time",
            "time_type": self.convolution_config["time_type"],
        }

        #
        convolve_on_the_fly(
            config=self.convolution_config,
            sfr_dict=sfr_dict,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
            time_bin_info_dict=time_bin_info_dict,
        )

    def test_convolve_on_the_fly_wrong_return_type_on_the_fly_function(self):

        #
        self.convolution_config["convolution_instructions"] = [
            {
                "input_data_name": "binary_c",
                "output_data_name": "BHBH",
                "convolution_type": "on-the-fly",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "normalized_yield": "probability",
                },
                "ignore_metallicity": True,
                "on_the_fly_function": wrong_return_type_on_the_fly_function,
            },
        ]

        #
        sfr_dict = self.convolution_config["SFR_info"]

        time_bin_info_dict = {
            "bin_number": 0,
            "bin_center": 0.5 * u.yr,
            "bin_edge_lower": 0,
            "bin_size": 1 * u.yr,
            "bin_type": "starformation time",
            "time_type": self.convolution_config["time_type"],
        }

        with self.assertRaises(ValueError):
            convolve_on_the_fly(
                config=self.convolution_config,
                sfr_dict=sfr_dict,
                convolution_instruction=self.convolution_config[
                    "convolution_instructions"
                ][0],
                time_bin_info_dict=time_bin_info_dict,
            )

    def test_convolve_on_the_fly_metallicity_required_not_included_on_the_fly_function(
        self,
    ):

        #
        self.convolution_config["convolution_instructions"] = [
            {
                "input_data_name": "binary_c",
                "output_data_name": "BHBH",
                "convolution_type": "on-the-fly",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "normalized_yield": "probability",
                },
                "on_the_fly_function": metallicity_required_not_included_on_the_fly_function,
            },
        ]

        # Set up SFR
        sfr_dict = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3]) * u.yr,
            "starformation_rate_array": np.array([1, 1, 1]) * u.Msun / u.yr,
            "metallicity_bin_edges": np.array([0.01, 0.1, 0.2, 0.3]),
            "metallicity_distribution_array": np.array(
                [[1, 2, 3], [4, 5, 6], [4, 5, 6]]
            ),
        }

        #
        self.convolution_config["SFR_info"] = sfr_dict

        #
        check_and_update_convolution_config(self.convolution_config)

        #
        sfr_dict = self.convolution_config["SFR_info"]

        time_bin_info_dict = {
            "bin_number": 0,
            "bin_center": 0.5 * u.yr,
            "bin_edge_lower": 0,
            "bin_size": 1 * u.yr,
            "bin_type": "starformation time",
            "time_type": self.convolution_config["time_type"],
        }

        with self.assertRaises(ValueError):
            convolve_on_the_fly(
                config=self.convolution_config,
                sfr_dict=sfr_dict,
                convolution_instruction=self.convolution_config[
                    "convolution_instructions"
                ][0],
                time_bin_info_dict=time_bin_info_dict,
            )

    def test_convolve_on_the_fly_convolve(self):

        #
        self.convolution_config["convolution_instructions"] = [
            {
                "input_data_name": "binary_c",
                "output_data_name": "BHBH",
                "convolution_type": "on-the-fly",
                "data_column_dict": {
                    "delay_time": "delay_time",
                    "normalized_yield": "probability",
                },
                "ignore_metallicity": True,
                "on_the_fly_function": dummy_on_the_fly_function,
            },
        ]

        # Set up SFR
        sfr_dict = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3]) * u.yr,
            "starformation_rate_array": np.array([1, 1, 1]) * u.Msun / u.yr,
        }

        #
        self.convolution_config["SFR_info"] = sfr_dict

        convolve(config=self.convolution_config)


if __name__ == "__main__":
    test_convolve_on_the_fly_obj = test_convolve_on_the_fly()
    test_convolve_on_the_fly_obj.setUp()
    test_convolve_on_the_fly_obj.test_convolve_on_the_fly_normal()
    test_convolve_on_the_fly_obj.test_convolve_on_the_fly_wrong_arguments_on_the_fly_function()
    test_convolve_on_the_fly_obj.test_convolve_on_the_fly_wrong_return_type_on_the_fly_function()
    test_convolve_on_the_fly_obj.test_convolve_on_the_fly_metallicity_required_not_included_on_the_fly_function()
    test_convolve_on_the_fly_obj.test_convolve_on_the_fly_convolve()
