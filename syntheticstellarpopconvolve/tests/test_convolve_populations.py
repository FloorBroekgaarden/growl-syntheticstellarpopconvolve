"""
Testcases for convolve_populations file
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
from astropy.cosmology import Planck13 as cosmo  # Planck 2013

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_check_config import (
    check_convolution_config,
)
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.convolve_populations import (
    generate_data_dict,
    pad_sfr_dict,
    update_sfr_dict,
)
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolve_populations", clean_path=True
)


class test_update_sfr_dict(unittest.TestCase):
    def setUp(self):

        logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s ] %(asctime)s: %(message)s"
        logging.basicConfig(format=FORMAT)
        logger.setLevel(logging.INFO)

        self.config = {
            "logger": logger,
            "time_type": "lookback_time",
            "cosmology": cosmo,
        }  # Example config
        self.sfr_dict = {
            "lookback_time_bin_edges": np.array([1, 2, 3]),
            "redshift_bin_edges": np.array([0.1, 0.2, 0.3]),
            "starformation_array": np.array([10, 20, 30]),
            "metallicity_bin_edges": np.array([0.01, 0.1, 0.2]),
            "metallicity_weighted_starformation_array": np.array(
                [[1, 2, 3], [4, 5, 6]]
            ),
        }

    def test_update_sfr_dict_lookback(self):
        updated_sfr_dict = update_sfr_dict(config=self.config, sfr_dict=self.sfr_dict)
        padded_keys = [
            key for key in updated_sfr_dict.keys() if key.startswith("padded")
        ]

        self.assertTrue(len(padded_keys) > 0)
        self.assertTrue("redshift_shell_volume_dict" not in updated_sfr_dict.keys())

    def test_update_sfr_dict_redshift(self):
        self.config["time_type"] = "redshift"
        updated_sfr_dict = update_sfr_dict(config=self.config, sfr_dict=self.sfr_dict)
        padded_keys = [
            key for key in updated_sfr_dict.keys() if key.startswith("padded")
        ]
        self.assertTrue(len(padded_keys) > 0)
        self.assertTrue("redshift_shell_volume_dict" in updated_sfr_dict.keys())


class test_generate_data_dict(unittest.TestCase):
    def test_generate_data_dict_events(self):

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
        prepare_output_file(config=self.convolution_config)

        #
        check_convolution_config(self.convolution_config)

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

        _, data_dict, _ = generate_data_dict(
            config=self.convolution_config,
            convolution_instruction=normal_convolution_instructions,
        )

        #
        np.testing.assert_array_equal(
            data_dict["delay_time"], self.dummy_data["delay_time"] * u.yr
        )

    def test_generate_data_dict_ensembles(self):
        #
        input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        ##############
        # SET UP DATA
        self.dummy_ensemble = {
            "metallicity": {
                "0": {
                    "delay_time": {
                        "0": {"a": {"1": 1}, "b": {"1": 1}},
                        "1": {"a": {"1": 2}, "b": {"1": 2}},
                        "2": {"a": {"1": 3}, "b": {"1": 3}},
                        "3": {"a": {"1": 4}, "b": {"1": 4}},
                    }
                }
            }
        }

        #############
        # create input HDF5 file
        with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

            ######################
            # Create groups
            input_hdf5_file.create_group("input_data")
            input_hdf5_file.create_group("input_data/ensemble")
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

            #
            input_hdf5_file.create_dataset(
                "input_data/ensemble/dummy", data=json.dumps(self.dummy_ensemble)
            )

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
                "input_data_type": "ensemble",
                "input_data_name": "dummy",
                "output_data_name": "dummy",
                "data_layer_dict": {
                    "delay_time": 3,
                },
                "ignore_metallicity": True,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        prepare_output_file(config=self.convolution_config)

        _, data_dict, _ = generate_data_dict(
            config=self.convolution_config,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
        )

        #
        self.assertTrue(data_dict["ensemble_data"]["dummy"], self.dummy_ensemble)

    def test_generate_data_dict_custom(self):

        #
        input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
        output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

        ##############
        # SET UP DATA
        self.dummy_ensemble = {
            "metallicity": {
                "0": {
                    "delay_time": {
                        "0": {"a": {"1": 1}, "b": {"1": 1}},
                        "1": {"a": {"1": 2}, "b": {"1": 2}},
                        "2": {"a": {"1": 3}, "b": {"1": 3}},
                        "3": {"a": {"1": 4}, "b": {"1": 4}},
                    }
                }
            }
        }

        #############
        # create input HDF5 file
        with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

            ######################
            # Create groups
            input_hdf5_file.create_group("input_data")
            input_hdf5_file.create_group("input_data/ensemble")
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

            #
            input_hdf5_file.create_dataset(
                "input_data/ensemble/dummy", data=json.dumps(self.dummy_ensemble)
            )

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
                "input_data_type": "custom",
                "input_data_name": "dummy",
                "output_data_name": "dummy",
                "data_layer_dict": {
                    "delay_time": 3,
                },
                "ignore_metallicity": True,
            },
        ]

        #
        self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

        #
        prepare_output_file(config=self.convolution_config)

        with self.assertRaises(NotImplementedError):

            _, data_dict, _ = generate_data_dict(
                config=self.convolution_config,
                convolution_instruction=self.convolution_config[
                    "convolution_instructions"
                ][0],
            )


class test_pad_sfr_dict(unittest.TestCase):
    def setUp(self):

        logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s ] %(asctime)s: %(message)s"
        logging.basicConfig(format=FORMAT)
        logger.setLevel(logging.INFO)

        self.config = {"logger": logger, "time_type": "lookback_time"}  # Example config
        self.sfr_dict = {
            "lookback_time_bin_edges": np.array([1, 2, 3]),
            "redshift_bin_edges": np.array([0.1, 0.2, 0.3]),
            "starformation_array": np.array([10, 20, 30]),
            "metallicity_bin_edges": np.array([0.01, 0.1, 0.2]),
            "metallicity_weighted_starformation_array": np.array(
                [[1, 2, 3], [4, 5, 6]]
            ),
        }

    def test_pad_sfr_dict_lookback_time(self):
        padded_sfr_dict = pad_sfr_dict(self.config, self.sfr_dict)
        self.assertTrue("padded_lookback_time_bin_edges" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_lookback_time_bin_edges"],
                np.array([1 - 1e13, 1, 2, 3, 3 + 1e13]),
            )
        )
        self.assertTrue("padded_starformation_array" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_starformation_array"],
                np.array([0, 10, 20, 30, 0]),
            )
        )

    def test_pad_sfr_dict_redshift(self):
        self.config["time_type"] = "redshift"
        padded_sfr_dict = pad_sfr_dict(self.config, self.sfr_dict)
        self.assertTrue("padded_redshift_bin_edges" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_redshift_bin_edges"],
                np.array([0.1 - 1e13, 0.1, 0.2, 0.3, 0.3 + 1e13]),
            )
        )
        self.assertTrue("padded_starformation_array" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_starformation_array"],
                np.array([0, 10, 20, 30, 0]),
            )
        )

    def test_pad_sfr_dict_metallicity(self):
        padded_sfr_dict = pad_sfr_dict(self.config, self.sfr_dict)
        self.assertTrue("padded_metallicity_bin_edges" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_metallicity_bin_edges"],
                np.array([1e-20, 0.01, 0.1, 0.2, 1]),
            )
        )
        self.assertTrue(
            "padded_metallicity_weighted_starformation_array" in padded_sfr_dict
        )
        expected_array = np.array(
            [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 0, 0, 0, 0]]
        )
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_metallicity_weighted_starformation_array"],
                expected_array,
            )
        )


if __name__ == "__main__":
    unittest.main()
