"""
Testcases for convolve_events file
"""

import copy
import json
import os
import unittest

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import pkg_resources

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_check_config import (
    check_convolution_config,
)
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.convolve_events import (
    event_convolution_function,
    extract_event_data,
)
from syntheticstellarpopconvolve.convolve_populations import update_sfr_dict
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolve_events", clean_path=True
)


class test_extract_event_data(unittest.TestCase):
    """ """

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

    def test_extract_event_data_normal(self):
        #
        normal_convolution_instructions = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_column_dict": {
                "delay_time": "delay_time",
                "yield_rate": "probability",
            },
            "ignore_metallicity": True,
        }

        #
        _, data_dict, _ = extract_event_data(
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
        )

        #
        np.testing.assert_array_equal(
            data_dict["delay_time"], self.dummy_data["delay_time"] * u.yr
        )

    def test_extract_event_data_factor_multiply(self):
        factor_convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_column_dict": {
                "delay_time": {"column_name": "delay_time", "conversion_factor": 2},
                "yield_rate": "probability",
            },
            "ignore_metallicity": True,
        }

        #
        _, data_dict, _ = extract_event_data(
            config=self.convolution_config,
            convolution_instruction=factor_convolution_instruction,
        )

        #
        np.testing.assert_array_equal(
            data_dict["delay_time"], 2 * self.dummy_data["delay_time"] * u.yr
        )

    def test_extract_event_data_function_multiply(self):
        ###########
        # function multiplying
        function_convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_column_dict": {
                "delay_time": {
                    "column_name": "delay_time",
                    "conversion_function": lambda x: x**2,
                },
                "yield_rate": "probability",
            },
            "ignore_metallicity": True,
        }

        #
        _, data_dict, _ = extract_event_data(
            config=self.convolution_config,
            convolution_instruction=function_convolution_instruction,
        )

        #
        np.testing.assert_array_equal(
            data_dict["delay_time"], (self.dummy_data["delay_time"] ** 2) * u.yr
        )

    def test_extract_event_data_not_existing(self):
        ###########
        # Non existent
        faulty_convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "dummy2",
            "output_data_name": "dummy",
            "data_column_dict": {
                "delay_time": {
                    "column_name": "delay_time",
                    "conversion_function": lambda x: x**2,
                },
                "yield_rate": "probability",
            },
            "ignore_metallicity": True,
        }

        with self.assertRaises(KeyError):

            #
            _, data_dict, _ = extract_event_data(
                config=self.convolution_config,
                convolution_instruction=faulty_convolution_instruction,
            )


class test_event_convolution_function(unittest.TestCase):
    """ """

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
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_array": np.array([1, 1, 1, 1, 1]) * u.Msun / u.yr / u.Gpc**3,
        }

        # set up convolution bins
        self.convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1, 2, 3, 4]) * u.yr
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
        check_convolution_config(self.convolution_config)

        #
        prepare_output_file(config=self.convolution_config)

    def test_event_convolution_function_normal(self):
        #
        normal_convolution_instructions = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_column_dict": {
                "delay_time": "delay_time",
                "yield_rate": "probability",
            },
            "ignore_metallicity": True,
        }

        #
        self.convolution_config, data_dict, _ = extract_event_data(
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
        )

        #
        sfr_dict = update_sfr_dict(
            sfr_dict=self.convolution_config["SFR_info"], config=self.convolution_config
        )

        #
        convolution_result = event_convolution_function(
            convolution_time_bin_center=0.5 * u.yr,
            job_dict={"sfr_dict": sfr_dict},
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
            data_dict=data_dict,
        )

        #
        np.testing.assert_array_equal(
            convolution_result["convolution_result"], np.array([1, 2, 3, 4.0])
        )

    def test_event_convolution_function_extra_weights(self):
        def extra_weights_function(config, data_dict):
            return np.zeros(data_dict["yield_rate"].shape)

        #
        normal_convolution_instructions = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_column_dict": {
                "delay_time": "delay_time",
                "yield_rate": "probability",
            },
            "ignore_metallicity": True,
            "extra_weights_function": extra_weights_function,
        }

        #
        self.convolution_config, data_dict, _ = extract_event_data(
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
        )

        #
        sfr_dict = update_sfr_dict(
            sfr_dict=self.convolution_config["SFR_info"], config=self.convolution_config
        )

        #
        convolution_result = event_convolution_function(
            convolution_time_bin_center=0.5 * u.yr,
            job_dict={"sfr_dict": sfr_dict},
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
            data_dict=data_dict,
        )

        #
        np.testing.assert_array_equal(
            convolution_result["convolution_result"],
            np.zeros(self.dummy_data["probability"].shape),
        )


if __name__ == "__main__":
    unittest.main()
