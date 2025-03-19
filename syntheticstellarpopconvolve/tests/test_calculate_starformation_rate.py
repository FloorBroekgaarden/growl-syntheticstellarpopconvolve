"""
This is the unittest file for the calculate_starformation_rate.py source file

TODO: nonbinned data backward integrate absolute
TODO: nonbinned data backward integrate metallicity-weighted

TODO: forward integrate absolute
TODO: forward integrate metallicity-weighted
"""

import copy
import logging
import os
import unittest

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve import (
    default_convolution_config,
    default_convolution_instruction,
)
from syntheticstellarpopconvolve.calculate_starformation_rate import (
    calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution,
    calculate_origin_time_array,
    general_sfr_digitise_function,
)
from syntheticstellarpopconvolve.check_and_update_sfr_dict import (
    check_and_update_sfr_dict,
)
from syntheticstellarpopconvolve.general_functions import calculate_bincenters, temp_dir
from syntheticstellarpopconvolve.prepare_redshift_interpolator import (
    prepare_redshift_interpolator,
)

np.random.seed(0)

TMP_DIR = temp_dir(
    "tests",
    "tests_convolution",
    "tests_calculat_starformation_rate",
    clean_path=True,
)


class test_general_sfr_digitise_function(unittest.TestCase):
    def test_general_sfr_digitise_function_absolute(self):

        ##############
        dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]) * u.yr,
            "value": np.array([3, 2, 1, 0]),
            "probability": np.array([1, 2, 3, 4]),
        }

        convolution_config = copy.copy(default_convolution_config)
        convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr,
        }

        # Set up SFR
        convolution_config = copy.copy(default_convolution_config)
        convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr,
        }
        convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1]) * u.yr
        )
        convolution_config["logger"].setLevel(logging.CRITICAL)
        convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")
        convolution_config["SFR_info"] = check_and_update_sfr_dict(
            sfr_dict=convolution_config["SFR_info"],
            requires_name=False,
            requires_metallicity_info=False,
            time_type=convolution_config["time_type"],
            config=convolution_config,
        )

        starformation_rate_array = general_sfr_digitise_function(
            config=convolution_config,
            sfr_dict=convolution_config["SFR_info"],
            time_values=dummy_data["delay_time"] + 0.5 * u.yr,
        )

        np.testing.assert_array_almost_equal(
            starformation_rate_array.value, np.array([1, 2, 3, 4])
        )
        self.assertEqual(starformation_rate_array.unit, u.Msun / u.yr)

    def test_general_sfr_digitise_function_metallicity_weighted(self):

        ##############
        dummy_data_with_metallicity = {
            "delay_time": np.array([0, 1, 2, 3, 0, 1, 2, 3]) * u.yr,
            "value": np.array([3, 2, 1, 0, 3, 2, 1, 0]),
            "probability": np.array([1, 2, 3, 4, 1, 2, 3, 4]),
            "metallicity": np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]),
        }

        convolution_config = copy.copy(default_convolution_config)
        convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr,
        }

        # Set up SFR
        convolution_config = copy.copy(default_convolution_config)
        convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr,
            "metallicity_bin_edges": np.array([0.0, 0.5, 1.0]),
            "metallicity_distribution_array": np.array(
                [[0.4, 0.6, 0.8, 0.9, 1], [1.6, 1.4, 1.2, 1.1, 1]]
            ).T,
        }

        convolution_config["logger"].setLevel(logging.CRITICAL)
        convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")
        convolution_config["SFR_info"] = check_and_update_sfr_dict(
            sfr_dict=convolution_config["SFR_info"],
            requires_name=False,
            requires_metallicity_info=True,
            time_type=convolution_config["time_type"],
            config=convolution_config,
        )

        starformation_rate_array = general_sfr_digitise_function(
            config=convolution_config,
            sfr_dict=convolution_config["SFR_info"],
            time_values=dummy_data_with_metallicity["delay_time"] + 0.5 * u.yr,
            metallicity_values=dummy_data_with_metallicity["metallicity"],
        )

        np.testing.assert_array_almost_equal(
            starformation_rate_array.value,
            np.array(
                [
                    0.2,
                    0.6,
                    1.2,
                    1.8,
                    0.8,
                    1.4,
                    1.8,
                    2.2,
                ]
            ),
        )
        self.assertEqual(starformation_rate_array.unit, u.Msun / u.yr)


class test_calculate_origin_time_array(unittest.TestCase):
    def test_calculate_origin_time_array_lookback(self):
        convolution_config = copy.copy(default_convolution_config)
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
        convolution_config["time_type"] = "redshift"
        convolution_config = prepare_redshift_interpolator(convolution_config)

        origin_time_array = calculate_origin_time_array(
            config=convolution_config,
            data_dict={"delay_time": np.array([1, 2, 3]) * 1e9 * u.yr},
            convolution_time_bin_center=0.5,  # redshift
        )
        # output_unit = u.Msun/u.yr/u.Gpc**3

        np.testing.assert_array_almost_equal(
            origin_time_array,
            np.array([0.6501032923316669, 0.8336451543045214, 1.0661079791875108]),
        )


class test_calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution(
    unittest.TestCase
):

    def test_calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution_absolute(
        self,
    ):

        ##############
        dummy_data = {
            "delay_time": np.array([0, 1, 2, 3]) * u.yr,
            "value": np.array([3, 2, 1, 0]),
            "probability": np.array([1, 2, 3, 4]),
        }

        # Set up SFR
        convolution_config = copy.copy(default_convolution_config)
        convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr,
        }
        convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1]) * u.yr
        )
        convolution_config["logger"].setLevel(logging.CRITICAL)
        convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")
        convolution_config["SFR_info"] = check_and_update_sfr_dict(
            sfr_dict=convolution_config["SFR_info"],
            requires_name=False,
            requires_metallicity_info=False,
            time_type=convolution_config["time_type"],
            config=convolution_config,
        )

        #########
        #
        data_dict = dummy_data
        config = convolution_config
        convolution_instruction = {
            **default_convolution_instruction,
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "convolution_direction": "backward",
            "convolution_type": "integrate",
            "data_column_dict": {
                "delay_time": "delay_time",
                "normalized_yield": "probability",
            },
        }
        sfr_dict = convolution_config["SFR_info"]
        convolution_lookback_time_bin_edges = convolution_config[
            "convolution_lookback_time_bin_edges"
        ]
        convolution_time_bin_center = calculate_bincenters(
            convolution_lookback_time_bin_edges
        )[0]

        #
        starformation_rate_array = (
            calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution(
                config=config,
                convolution_instruction=convolution_instruction,
                convolution_time_bin_center=convolution_time_bin_center,
                data_dict=data_dict,
                sfr_dict=sfr_dict,
            )
        )

        np.testing.assert_array_almost_equal(
            starformation_rate_array.value, np.array([1, 2, 3, 4])
        )
        self.assertEqual(starformation_rate_array.unit, u.Msun / u.yr)

    def test_calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution_metallicity_weighted(
        self,
    ):

        ##############
        dummy_data_with_metallicity = {
            "delay_time": np.array([0, 1, 2, 3, 0, 1, 2, 3]) * u.yr,
            "value": np.array([3, 2, 1, 0, 3, 2, 1, 0]),
            "probability": np.array([1, 2, 3, 4, 1, 2, 3, 4]),
            "metallicity": np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75]),
        }

        # Set up SFR
        convolution_config = copy.copy(default_convolution_config)
        convolution_config["SFR_info"] = {
            "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]) * u.yr,
            "starformation_rate_array": np.array([1, 2, 3, 4, 5]) * u.Msun / u.yr,
            "metallicity_bin_edges": np.array([0.0, 0.5, 1.0]),
            "metallicity_distribution_array": np.array(
                [[0.4, 0.6, 0.8, 0.9, 1], [1.6, 1.4, 1.2, 1.1, 1]]
            ).T,
        }
        convolution_config["convolution_lookback_time_bin_edges"] = (
            np.array([0, 1]) * u.yr
        )
        convolution_config["logger"].setLevel(logging.CRITICAL)
        convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")
        convolution_config["SFR_info"] = check_and_update_sfr_dict(
            sfr_dict=convolution_config["SFR_info"],
            requires_name=False,
            requires_metallicity_info=True,
            time_type=convolution_config["time_type"],
            config=convolution_config,
        )

        #########
        #
        data_dict = dummy_data_with_metallicity
        config = convolution_config
        convolution_instruction = {
            **default_convolution_instruction,
            "input_data_name": "dummy",
            "output_data_name": "dummy",
            "convolution_direction": "backward",
            "convolution_type": "integrate",
            "data_column_dict": {
                "delay_time": "delay_time",
                "normalized_yield": "probability",
            },
        }
        sfr_dict = convolution_config["SFR_info"]
        convolution_lookback_time_bin_edges = convolution_config[
            "convolution_lookback_time_bin_edges"
        ]
        convolution_time_bin_center = calculate_bincenters(
            convolution_lookback_time_bin_edges
        )[0]

        #
        starformation_rate_array = (
            calculate_digitized_sfr_rates_non_binned_data_for_backward_convolution(
                config=config,
                convolution_instruction=convolution_instruction,
                convolution_time_bin_center=convolution_time_bin_center,
                data_dict=data_dict,
                sfr_dict=sfr_dict,
            )
        )

        np.testing.assert_array_almost_equal(
            starformation_rate_array.value,
            np.array(
                [
                    0.2,
                    0.6,
                    1.2,
                    1.8,
                    0.8,
                    1.4,
                    1.8,
                    2.2,
                ]
            ),
        )
        self.assertEqual(starformation_rate_array.unit, u.Msun / u.yr)


if __name__ == "__main__":
    unittest.main()
