"""
Testcases for convolve_ensembles file
"""

import copy
import json
import os
import unittest

import astropy.units as u
import h5py
import numpy as np
import pkg_resources

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_check_config import (
    check_convolution_config,
)
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.convolve_ensembles import (
    _get_ensemble_structure,
    attach_endpoints,
    check_if_value_layer,
    check_if_value_layer_and_get_layer_iterable,
    ensemble_convolution_function,
    ensemble_handle_marginalisation,
    ensemble_handle_SFR_multiplication,
    ensemble_marginalise_layer,
    extract_endpoints,
    extract_ensemble_data,
    get_data_layer_dict_values,
    get_deepest_data_layer_depth,
    get_depth_ensemble_all_endpoints,
    get_depth_ensemble_first_endpoint,
    get_ensemble_binsizes,
    get_ensemble_structure,
    get_layer_iterable,
    get_max_depth_ensemble,
    handle_binsize_multiplication_factor,
    invert_data_layer_dict,
    multiply_ensemble,
    set_endpoints,
    shift_data_layer,
    shift_layers_dict,
    shift_layers_list,
    strip_ensemble_endpoints,
)
from syntheticstellarpopconvolve.convolve_populations import update_sfr_dict
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolve_ensembles", clean_path=True
)


class test_handle_binsize_multiplication_factor(unittest.TestCase):
    def test_handle_binsize_multiplication_factor_no_binsize_multiplication(self):
        convolution_config = copy.copy(convolution_options_defaults)

        binsizes, extra_value_dict = handle_binsize_multiplication_factor(
            config=convolution_config,
            ensemble={"0.1": 2, "0.2": 2, "0.4": 3},
            data_layer_dict_entry={"binsizes": [1, 2]},
            key="0.1",
            key_i=0,
            binsizes=None,
            extra_value_dict={},
            name="delay_time",
        )

        self.assertFalse(extra_value_dict)

    def test_handle_binsize_multiplication_factor_binsizes_passed(self):
        convolution_config = copy.copy(convolution_options_defaults)

        binsizes, extra_value_dict = handle_binsize_multiplication_factor(
            config=convolution_config,
            ensemble={"0.1": 2, "0.2": 2, "0.4": 3},
            data_layer_dict_entry={"multiply_by_binsize": True},
            key="0.1",
            key_i=0,
            binsizes=[0.1, 0.15, 0.2],
            extra_value_dict={},
            name="delay_time",
        )
        self.assertTrue(extra_value_dict == {"delay_time_binsize": 0.1})

    def test_handle_binsize_multiplication_factor_binsizes_calculated(self):
        convolution_config = copy.copy(convolution_options_defaults)

        binsizes, extra_value_dict = handle_binsize_multiplication_factor(
            config=convolution_config,
            ensemble={"0.1": 2, "0.2": 2, "0.4": 3},
            data_layer_dict_entry={"multiply_by_binsize": True},
            key="0.2",
            key_i=1,
            binsizes=None,
            extra_value_dict={},
            name="delay_time",
        )

        self.assertAlmostEqual(extra_value_dict["delay_time_binsize"], 0.15, 6)


class test_get_ensemble_binsizes(unittest.TestCase):
    def test_get_ensemble_binsizes_predetermined_binsizes(self):
        binsizes = get_ensemble_binsizes(
            config={},
            ensemble={"0.1": 2, "0.2": 2},
            data_layer_dict_entry={"binsizes": [1, 2]},
        )

        self.assertTrue(binsizes == [1, 2])

    def test_get_ensemble_binsizes_no_scaling(self):

        convolution_config = copy.copy(convolution_options_defaults)

        binsizes = get_ensemble_binsizes(
            config=convolution_config,
            ensemble={"0.1": 2, "0.2": 2, "0.4": 3},
            data_layer_dict_entry={},
        )

        #
        np.testing.assert_array_almost_equal(binsizes, np.array([0.1, 0.15, 0.2]))

    def test_get_ensemble_binsizes_factor_scaling(self):

        convolution_config = copy.copy(convolution_options_defaults)

        binsizes = get_ensemble_binsizes(
            config=convolution_config,
            ensemble={"0.1": 2, "0.2": 2, "0.4": 3},
            data_layer_dict_entry={"conversion_factor": 2},
        )

        #
        np.testing.assert_array_almost_equal(binsizes, np.array([0.2, 0.3, 0.4]))

    def test_get_ensemble_binsizes_factor_function(self):

        convolution_config = copy.copy(convolution_options_defaults)

        binsizes = get_ensemble_binsizes(
            config=convolution_config,
            ensemble={"0.1": 2, "0.2": 2, "0.4": 3},
            data_layer_dict_entry={"conversion_function": lambda x: 10**x},
        )

        #
        np.testing.assert_array_almost_equal(
            binsizes, np.array([0.29051909, 0.58272477, 1.16701535])
        )


class test_ensemble_handle_SFR_multiplication(unittest.TestCase):
    def setUp(self):
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
            "starformation_array": np.array([2, 1, 1, 1, 1]) * u.Msun / u.yr / u.Gpc**3,
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

    def test_ensemble_handle_SFR_multiplication_normal(self):

        #
        ensemble = self.dummy_ensemble["metallicity"]["0"]["delay_time"]["0"]

        #
        data_dict = {"delay_time": 0}

        #
        sfr_dict = update_sfr_dict(
            sfr_dict=self.convolution_config["SFR_info"], config=self.convolution_config
        )

        ensemble = ensemble_handle_SFR_multiplication(
            convolution_time_bin_center=0.5,
            job_dict={"sfr_dict": sfr_dict, "job_number": 0},
            config=self.convolution_config,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
            ensemble=ensemble,
            data_dict=data_dict,
            extra_value_dict=None,
        )

        expected_ensemble = {"a": {"1": 2.0}, "b": {"1": 2.0}}

        self.assertTrue(ensemble == expected_ensemble)

    def test_ensemble_handle_SFR_multiplication_extra_value(self):

        #
        ensemble = self.dummy_ensemble["metallicity"]["0"]["delay_time"]["0"]

        #
        data_dict = {"delay_time": 0}
        extra_value_dict = {"time_bin": 3, "metallicity_bin": 4}

        #
        sfr_dict = update_sfr_dict(
            sfr_dict=self.convolution_config["SFR_info"], config=self.convolution_config
        )

        ensemble = ensemble_handle_SFR_multiplication(
            convolution_time_bin_center=0.5,
            job_dict={"sfr_dict": sfr_dict, "job_number": 0},
            config=self.convolution_config,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
            ensemble=ensemble,
            data_dict=data_dict,
            extra_value_dict=extra_value_dict,
        )

        expected_ensemble = {"a": {"1": 24.0}, "b": {"1": 24.0}}

        self.assertTrue(ensemble == expected_ensemble)


class test_ensemble_convolution_function(unittest.TestCase):
    def setUp(self):
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

        #
        check_convolution_config(self.convolution_config)

    def test_normal(self):
        _, data_dict, _ = extract_ensemble_data(
            config=self.convolution_config,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
        )

        #
        sfr_dict = update_sfr_dict(
            sfr_dict=self.convolution_config["SFR_info"], config=self.convolution_config
        )

        #
        result_dict = ensemble_convolution_function(
            convolution_time_bin_center=0.5 * u.yr,
            job_dict={"sfr_dict": sfr_dict, "job_number": 0},
            config=self.convolution_config,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
            data_dict=data_dict,
        )

        #
        expected_stripped_ensemble = {
            "metallicity": {
                "0": {
                    "delay_time": {
                        "0": {"a": {"1": 0}, "b": {"1": 0}},
                        "1": {"a": {"1": 0}, "b": {"1": 0}},
                        "2": {"a": {"1": 0}, "b": {"1": 0}},
                        "3": {"a": {"1": 0}, "b": {"1": 0}},
                    }
                }
            }
        }

        #
        self.assertTrue("convolution_result" in result_dict)
        np.testing.assert_array_equal(
            result_dict["convolution_result"], [1, 1, 2, 2, 3, 3, 4, 4]
        )

        #
        self.assertTrue("stripped_ensemble" in result_dict)
        self.assertTrue(
            result_dict["stripped_ensemble"] == {"dummy": expected_stripped_ensemble}
        )


class test_ensemble_marginalise_layer(unittest.TestCase):
    def test_ensemble_marginalise_layer(self):
        dummy_ensemble = {
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

        target_ensemble = {
            "metallicity": {
                "0": {
                    "delay_time": {
                        "0": {"1": 2},
                        "1": {"1": 4},
                        "2": {"1": 6},
                        "3": {"1": 8},
                    }
                }
            }
        }

        #
        marginalized_ensemble = ensemble_marginalise_layer(
            ensemble=dummy_ensemble, marginalisation_depth=4
        )
        self.assertTrue(target_ensemble == marginalized_ensemble)


class test_ensemble_handle_marginalisation(unittest.TestCase):
    def setUp(self):
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
                "input_data_type": "event",
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

    def test_ensemble_handle_marginalisation_pre_conv(self):

        #
        convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_layer_dict": {
                "delay_time": 3,
            },
            "ignore_metallicity": True,
            "marginalisation_list": [3, 4],
        }

        #
        target_ensemble = {
            "metallicity": {
                "0": {
                    "delay_time": {
                        "0": {"1": 2},
                        "1": {"1": 4},
                        "2": {"1": 6},
                        "3": {"1": 8},
                    }
                }
            }
        }

        #
        config, ensemble, convolution_instruction = ensemble_handle_marginalisation(
            config=self.convolution_config,
            ensemble=self.dummy_ensemble,
            convolution_instruction=convolution_instruction,
            is_pre_conv=True,
        )

        self.assertTrue(convolution_instruction["marginalisation_list"] == [3])
        self.assertTrue(ensemble, target_ensemble)

    def test_ensemble_handle_marginalisation_post_conv(self):

        #
        convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_layer_dict": {
                "delay_time": 3,
            },
            "ignore_metallicity": True,
            "marginalisation_list": [3, 4],
        }

        #
        target_ensemble = {"metallicity": {"0": {"delay_time": {"1": 20}}}}
        #
        config, ensemble, convolution_instruction = ensemble_handle_marginalisation(
            config=self.convolution_config,
            ensemble=self.dummy_ensemble,
            convolution_instruction=convolution_instruction,
            is_pre_conv=False,
        )

        self.assertTrue(convolution_instruction["marginalisation_list"] == [])
        self.assertTrue(ensemble, target_ensemble)

    def test_ensemble_handle_marginalisation_pre_conv_updated_data_dict(self):

        #
        convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_layer_dict": {
                "delay_time": 3,
            },
            "ignore_metallicity": True,
            "marginalisation_list": [1],
        }

        #
        target_ensemble = {
            "metallicity": {
                "delay_time": {
                    "0": {"a": {"1": 1}, "b": {"1": 1}},
                    "1": {"a": {"1": 2}, "b": {"1": 2}},
                    "2": {"a": {"1": 3}, "b": {"1": 3}},
                    "3": {"a": {"1": 4}, "b": {"1": 4}},
                }
            }
        }

        #
        _, ensemble, convolution_instruction = ensemble_handle_marginalisation(
            config=self.convolution_config,
            ensemble=self.dummy_ensemble,
            convolution_instruction=convolution_instruction,
            is_pre_conv=False,
        )

        self.assertTrue(convolution_instruction["marginalisation_list"] == [])
        self.assertTrue(ensemble == target_ensemble)
        self.assertTrue(convolution_instruction["data_layer_dict"]["delay_time"] == 3)

    def test_ensemble_handle_marginalisation_pre_conv_depth_error(self):

        #
        convolution_instruction = {
            "input_data_type": "event",
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "data_layer_dict": {
                "delay_time": 3,
            },
            "ignore_metallicity": True,
            "marginalisation_list": [0],
        }

        #
        with self.assertRaises(ValueError):
            _, _, _ = ensemble_handle_marginalisation(
                config=self.convolution_config,
                ensemble=self.dummy_ensemble,
                convolution_instruction=convolution_instruction,
                is_pre_conv=False,
            )


class test_extract_ensemble_data(unittest.TestCase):
    def setUp(self):
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

    def test_extract_ensemble_data_normal(self):
        _, data_dict, convolution_instruction = extract_ensemble_data(
            config=self.convolution_config,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
        )

        #
        self.assertTrue(data_dict["ensemble_data"]["dummy"], self.dummy_ensemble)

        #
        self.assertTrue(convolution_instruction["data_layer_dict"]["delay_time"] == 4)

    def test_extract_ensemble_data_marginalise(self):
        self.convolution_config["convolution_instructions"][0][
            "marginalisation_list"
        ] = [3, 4]

        _, data_dict, convolution_instruction = extract_ensemble_data(
            config=self.convolution_config,
            convolution_instruction=self.convolution_config["convolution_instructions"][
                0
            ],
        )

        target_ensemble = {
            "metallicity": {
                "0": {
                    "delay_time": {
                        "0": {"1": 2},
                        "1": {"1": 4},
                        "2": {"1": 6},
                        "3": {"1": 8},
                    }
                }
            }
        }

        #
        self.assertTrue(data_dict["ensemble_data"]["dummy"], target_ensemble)

        # the remaining marginalisation layer (which was 3 before) will be shifted
        self.assertTrue(convolution_instruction["marginalisation_list"] == [4])


class test_get_deepest_data_layer_depth(unittest.TestCase):
    def setUp(self):
        self.data_layer_dict = {
            "key1": 1,
            "key2": {"layer_depth": 2},
            "key3": 3,
            "key4": {"layer_depth": 4},
        }

    def test_get_deepest_data_layer_depth(self):
        deepest_depth = get_deepest_data_layer_depth(self.data_layer_dict)
        self.assertEqual(
            deepest_depth, 4
        )  # Deepest data layer depth in the provided dictionary

    def test_get_deepest_data_layer_depth_empty_dict(self):
        empty_dict = {}
        with self.assertRaises(ValueError):
            get_deepest_data_layer_depth(empty_dict)

    def test_get_deepest_data_layer_depth_invalid_input(self):
        invalid_data_layer_dict = {"key1": [1, 2, 3]}
        with self.assertRaises(ValueError):
            get_deepest_data_layer_depth(invalid_data_layer_dict)


class test_get_data_layer_dict_values(unittest.TestCase):
    def setUp(self):
        self.data_layer_dict = {
            "key1": 1,
            "key2": {"layer_depth": 2},
            "key3": 3,
            "key4": {"layer_depth": 4},
        }

    def test_get_data_layer_dict_values(self):
        data_layer_values = get_data_layer_dict_values(self.data_layer_dict)
        expected_values = [1, 2, 3, 4]
        self.assertListEqual(data_layer_values, expected_values)

    def test_get_data_layer_dict_values_invalid_input(self):
        invalid_data_layer_dict = {"key1": [1, 2, 3]}
        with self.assertRaises(ValueError):
            get_data_layer_dict_values(invalid_data_layer_dict)


class test_multiply_ensemble(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {
            "a": {"b": {"c": 1, "d": 2}, "e": 3},
            "f": {"g": {"h": {"i": {"j": {"k": {"l": 10}}}}}},
        }

    def test_multiply_ensemble(self):
        factor = 2
        multiplied_dict = {
            "a": {"b": {"c": 2, "d": 4}, "e": 6},
            "f": {"g": {"h": {"i": {"j": {"k": {"l": 20}}}}}},
        }
        multiply_ensemble(self.nested_dict, factor)
        self.assertEqual(self.nested_dict, multiplied_dict)


class test_invert_data_layer_dict(unittest.TestCase):
    def setUp(self):
        self.data_layer_dict = {
            "key1": 1,
            "key2": {"layer_depth": 2},
            "key3": 3,
            "key4": {"layer_depth": 4},
        }

    def test_invert_data_layer_dict(self):
        inverted_dict = invert_data_layer_dict(self.data_layer_dict)
        expected_inverted_dict = {1: "key1", 2: "key2", 3: "key3", 4: "key4"}
        self.assertDictEqual(inverted_dict, expected_inverted_dict)

    def test_invert_data_layer_dict_invalid_input(self):
        invalid_data_layer_dict = {"key1": [1, 2, 3]}
        with self.assertRaises(ValueError):
            invert_data_layer_dict(invalid_data_layer_dict)


class test_get_depth_ensemble_first_endpoint(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {
            "a": {"b": {"c": 1, "d": 2}, "e": 3},
            "f": {"g": {"h": {"i": {"j": {"k": {"l": 10}}}}}},
        }

    def test_get_depth_ensemble_first_endpoint(self):
        depth = get_depth_ensemble_first_endpoint(self.nested_dict)
        self.assertEqual(
            depth, 3
        )  # Depth of the first endpoint in the nested dictionary


class test_get_max_depth_ensemble(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {
            "a": {"b": {"c": 1, "d": 2}, "e": 3},
            "f": {"g": {"h": {"i": {"j": {"k": {"l": 10}}}}}},
        }

    def test_find_max_depth(self):
        max_depth = get_max_depth_ensemble(self.nested_dict)
        self.assertEqual(max_depth, 7)  # Maximum depth is 7 in this nested dictionary


class test_get_depth_ensemble_all_endpoints(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {
            "a": {"b": {"c": 1, "d": 2}, "e": 3},
            "f": {"g": {"h": {"i": {"j": {"k": {"l": 10}}}}}},
        }

    def test_get_depth_ensemble_all_endpoints(self):
        endpoint_depths = get_depth_ensemble_all_endpoints(self.nested_dict)
        expected_depths = [
            3,
            3,
            2,
            7,
        ]  # Depths of all endpoints in the nested dictionary
        self.assertEqual(endpoint_depths, expected_depths)


class test_shift_layers_dict(unittest.TestCase):
    def setUp(self):
        self.data_layer_dict = {
            "layer1": 1,
            "layer2": {"layer_depth": 2},
            "layer3": {"layer_depth": 3},
        }

    def test_shift_layers_dict_positive_shift(self):
        shift_value = 5
        expected_result = {
            "layer1": 6,
            "layer2": {"layer_depth": 7},
            "layer3": {"layer_depth": 8},
        }
        self.assertEqual(
            shift_layers_dict(self.data_layer_dict, shift_value), expected_result
        )

    def test_shift_layers_dict_negative_shift(self):
        shift_value = -2
        expected_result = {
            "layer1": -1,
            "layer2": {"layer_depth": 0},
            "layer3": {"layer_depth": 1},
        }
        self.assertEqual(
            shift_layers_dict(self.data_layer_dict, shift_value), expected_result
        )

    def test_shift_layers_dict_zero_shift(self):
        shift_value = 0
        expected_result = {
            "layer1": 1,
            "layer2": {"layer_depth": 2},
            "layer3": {"layer_depth": 3},
        }
        self.assertEqual(
            shift_layers_dict(self.data_layer_dict, shift_value), expected_result
        )

    def test_shift_layers_dict_empty_dict(self):
        empty_dict = {}
        shift_value = 5
        expected_result = {}
        self.assertEqual(shift_layers_dict(empty_dict, shift_value), expected_result)

    def test_shift_layers_dict_unsupported(self):
        shift_value = 1
        data_layer_dict = {
            "layer1": 1,
            "layer2": {"layer_depth": 2},
            "layer3": "unsupported",
        }
        with self.assertRaises(ValueError):
            shift_layers_dict(data_layer_dict, shift_value)


class test_shift_data_layer(unittest.TestCase):
    def test_shift_data_layer_int_value(self):
        data_layer_dict = {"layer1": 1}
        key = "layer1"
        shift = 5
        expected_result = {"layer1": 6}
        shift_data_layer(data_layer_dict, key, shift)
        self.assertEqual(data_layer_dict, expected_result)

    def test_shift_data_layer_dict_value(self):
        data_layer_dict = {"layer2": {"layer_depth": 2}}
        key = "layer2"
        shift = -2
        expected_result = {"layer2": {"layer_depth": 0}}
        shift_data_layer(data_layer_dict, key, shift)
        self.assertEqual(data_layer_dict, expected_result)

    def test_shift_data_layer_unsupported_input(self):
        data_layer_dict = {"layer3": "unsupported"}
        key = "layer3"
        shift = 5
        with self.assertRaises(ValueError):
            shift_data_layer(data_layer_dict, key, shift)


class test_shift_layers_list(unittest.TestCase):
    def test_shift_layers_list_positive_shift(self):
        layer_list = [1, 2, 3, 4]
        shift_value = 5
        expected_result = [6, 7, 8, 9]
        self.assertEqual(shift_layers_list(layer_list, shift_value), expected_result)

    def test_shift_layers_list_negative_shift(self):
        layer_list = [1, 2, 3, 4]
        shift_value = -2
        expected_result = [-1, 0, 1, 2]
        self.assertEqual(shift_layers_list(layer_list, shift_value), expected_result)

    def test_shift_layers_list_zero_shift(self):
        layer_list = [1, 2, 3, 4]
        shift_value = 0
        expected_result = [1, 2, 3, 4]
        self.assertEqual(shift_layers_list(layer_list, shift_value), expected_result)

    def test_shift_layers_list_empty_list(self):
        layer_list = []
        shift_value = 5
        expected_result = []
        self.assertEqual(shift_layers_list(layer_list, shift_value), expected_result)


class test_strip_ensemble_endpoints(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": {"g": 4}, "h": 5}

    def test_strip_ensemble_endpoints(self):
        expected_endpoints = [1, 2, 3, 4, 5]
        ensemble, endpoints = strip_ensemble_endpoints(self.nested_dict)
        self.assertEqual(endpoints, expected_endpoints)

        # Ensure original ensemble endpoints are set to 0
        extracted_endpoints = extract_endpoints(ensemble)
        self.assertTrue(all(endpoint == 0 for endpoint in extracted_endpoints))


class test_check_if_value_layer(unittest.TestCase):
    def setUp(self):
        self.name_keys = {"name1": 1, "name2": 2, "name3": 3}
        self.value_keys = {"1": 1, "2": 2, "3": 3}
        self.mixed_keys = {"1": 1, "name2": 2, "3": 3}

    def test_value_layer(self):
        self.assertTrue(check_if_value_layer(self.value_keys.keys()))

    def test_name_layer(self):
        self.assertFalse(check_if_value_layer(self.name_keys.keys()))

    def test_mixed_layer(self):
        self.assertFalse(check_if_value_layer(self.mixed_keys.keys()))


class test_get_layer_iterable(unittest.TestCase):
    def setUp(self):
        self.name_keys = {"name1": 1, "name2": 2, "name3": 3}
        self.value_keys = {"1": 1, "2": 2, "3": 3}
        self.unsorted_value_keys = {"1": 1, "-2": 2, "3": 3}
        self.mixed_keys = {"1": 1, "name2": 2, "3": 3}

    def test_get_layer_iterable_value_layer(self):
        self.assertEqual(
            list(get_layer_iterable(self.value_keys, True)), ["1", "2", "3"]
        )

    def test_get_layer_iterable_unsorted_value_layer(self):
        self.assertEqual(
            list(get_layer_iterable(self.unsorted_value_keys, True)), ["-2", "1", "3"]
        )

    def test_get_layer_iterable_name_layer(self):
        self.assertEqual(
            list(get_layer_iterable(self.name_keys, False)), ["name1", "name2", "name3"]
        )

    def test_get_layer_iterable_mixed_layer(self):
        self.assertEqual(
            list(get_layer_iterable(self.mixed_keys, False)), ["1", "name2", "3"]
        )


class test_check_if_value_layer_and_get_layer_iterable(unittest.TestCase):
    def setUp(self):
        self.name_keys = {"name1": 1, "name2": 2, "name3": 3}
        self.value_keys = {"1": 1, "2": 2, "3": 3}
        self.unsorted_value_keys = {"1": 1, "-2": 2, "3": 3}
        self.mixed_keys = {"1": 1, "name2": 2, "3": 3}

    def test_value_layer_and_get_layer_iterable(self):
        is_value_layer, iterable = check_if_value_layer_and_get_layer_iterable(
            self.value_keys
        )
        self.assertTrue(is_value_layer)
        self.assertEqual(list(iterable), ["1", "2", "3"])

    def test_unsorted_value_layer_and_get_layer_iterable(self):
        is_value_layer, iterable = check_if_value_layer_and_get_layer_iterable(
            self.unsorted_value_keys
        )
        self.assertTrue(is_value_layer)
        self.assertEqual(list(iterable), ["-2", "1", "3"])

    def test_name_layer_and_get_layer_iterable(self):
        is_value_layer, iterable = check_if_value_layer_and_get_layer_iterable(
            self.name_keys
        )
        self.assertFalse(is_value_layer)
        self.assertEqual(list(iterable), ["name1", "name2", "name3"])

    def test_mixed_layer_and_get_layer_iterable(self):
        is_value_layer, iterable = check_if_value_layer_and_get_layer_iterable(
            self.mixed_keys
        )
        self.assertFalse(is_value_layer)
        self.assertEqual(list(iterable), ["1", "name2", "3"])


class test_extract_endpoints(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": {"g": 4}, "h": 5}
        self.flat_dict = {"a": 1, "b": 2, "c": 3}
        self.empty_dict = {}

    def test_extract_endpoints_nested_dict(self):
        expected_endpoints = [1, 2, 3, 4, 5]
        self.assertEqual(extract_endpoints(self.nested_dict), expected_endpoints)

    def test_extract_endpoints_flat_dict(self):
        expected_endpoints = [1, 2, 3]
        self.assertEqual(extract_endpoints(self.flat_dict), expected_endpoints)

    def test_extract_endpoints_empty_dict(self):
        self.assertEqual(extract_endpoints(self.empty_dict), [])


class test_attach_endpoints(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {
            "a": {"b": {"c": None, "d": None}, "e": None},
            "f": {"g": None},
            "h": None,
        }
        self.endpoint_array = [1, 2, 3, 4, 5]

    def test_attach_endpoints_nested_dict(self):
        expected_dict = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": {"g": 4}, "h": 5}
        attach_endpoints(self.nested_dict, self.endpoint_array)
        self.assertEqual(self.nested_dict, expected_dict)

    def test_attach_endpoints_empty_dict(self):
        empty_dict = {}
        with self.assertRaises(ValueError):
            attach_endpoints(empty_dict, self.endpoint_array)


class test_set_endpoints(unittest.TestCase):
    def setUp(self):
        self.nested_dict = {
            "a": {"b": {"c": None, "d": None}, "e": None},
            "f": {"g": None},
            "h": None,
        }
        self.value = 100

    def test_set_endpoints_nested_dict(self):
        expected_dict = {
            "a": {"b": {"c": 100, "d": 100}, "e": 100},
            "f": {"g": 100},
            "h": 100,
        }
        set_endpoints(self.nested_dict, self.value)
        self.assertEqual(self.nested_dict, expected_dict)

    def test_set_endpoints_empty_dict(self):
        empty_dict = {}
        with self.assertRaises(ValueError):
            set_endpoints(empty_dict, self.value)


class test__get_ensemble_structure(unittest.TestCase):
    def test_basic_structure(self):
        ensemble = {"a": {"b": {}, "c": {}}, "d": {"e": {}}}
        expected_structure = {0: ["a", "d"], 1: ["b", "c", "e"]}
        self.assertEqual(
            _get_ensemble_structure(ensemble, {0: [], 1: []}, 2), expected_structure
        )

    def test_max_depth_reached(self):
        ensemble = {"a": {"b": {}}, "c": {"d": {}}}
        expected_structure = {0: ["a", "c"]}
        self.assertEqual(
            _get_ensemble_structure(ensemble, {0: []}, 1), expected_structure
        )

    def test_empty_ensemble(self):
        ensemble = {}
        expected_structure = {0: []}
        self.assertEqual(
            _get_ensemble_structure(ensemble, {0: []}, 1), expected_structure
        )

    def test_nested_empty_ensemble(self):
        ensemble = {"a": {"b": {"c": {}}}}
        expected_structure = {0: ["a"], 1: ["b"], 2: ["c"]}
        self.assertEqual(
            _get_ensemble_structure(ensemble, {0: [], 1: [], 2: []}, 3),
            expected_structure,
        )


class test_get_ensemble_structure(unittest.TestCase):
    def setUp(self):
        # Define some sample ensembles for testing
        self.ensemble_single_depth = {"layer1": {"node1": 1, "node2": 2, "node3": 3}}

        self.ensemble_multiple_depth = {
            "layer1": {
                "node1": {"subnode1": 11, "subnode2": 12},
                "node2": {"subnode1": 21, "subnode2": 22},
            }
        }

        self.ensemble_named_layer = {
            "layer1": {
                "node1": {"subnode1": 11, "subnode2": 12},
                "node2": {"subnode1": 21, "subnode2": 22},
            },
            "layer2": {
                "node1": {"subnode1": 31, "subnode2": 32},
                "node2": {"subnode1": 41, "subnode2": 42},
            },
        }

    def test_get_ensemble_structure_single_depth(self):
        expected_structure = {0: ["layer1"], 1: ["node1", "node2", "node3"]}
        self.assertEqual(
            get_ensemble_structure(self.ensemble_single_depth), expected_structure
        )

    def test_get_ensemble_structure_multiple_depth(self):
        expected_structure = {
            0: ["layer1"],
            1: ["node1", "node2"],
            2: ["subnode1", "subnode2"],
        }
        self.assertEqual(
            get_ensemble_structure(self.ensemble_multiple_depth), expected_structure
        )

    def test_get_ensemble_structure_named_layer(self):
        # Test with named layer list provided

        # Test with named layer containing multiple values
        with self.assertRaises(ValueError):
            get_ensemble_structure(self.ensemble_named_layer, named_layer_list=[1])


if __name__ == "__main__":
    unittest.main()
