import logging
import unittest

import astropy.units as u
import numpy as np
from astropy.cosmology import Planck13 as cosmo  # Planck 2013

from syntheticstellarpopconvolve import default_convolution_config
from syntheticstellarpopconvolve.check_and_update_sfr_dict import (
    check_sfr_dict,
    pad_sfr_dict,
    update_sfr_dict,
)
from syntheticstellarpopconvolve.general_functions import temp_dir

TMP_DIR = temp_dir(
    "tests",
    "tests_convolution",
    "tests_check_and_update_sfr_dict",
    clean_path=True,
)


class test_pad_sfr_dict(unittest.TestCase):
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
            "starformation_rate_array": np.array([10, 20, 30]),
            "metallicity_bin_edges": np.array([0.01, 0.1, 0.2]),
            "metallicity_distribution_array": np.array([[1, 2, 3], [4, 5, 6]]),
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
        self.assertTrue("padded_starformation_rate_array" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_starformation_rate_array"],
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
        self.assertTrue("padded_starformation_rate_array" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_starformation_rate_array"],
                np.array([0, 10, 20, 30, 0]),
            )
        )

    def test_pad_sfr_dict_metallicity(self):
        self.sfr_dict = update_sfr_dict(sfr_dict=self.sfr_dict, config=self.config)
        padded_sfr_dict = pad_sfr_dict(self.config, self.sfr_dict)
        self.assertTrue("padded_metallicity_bin_edges" in padded_sfr_dict)
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_metallicity_bin_edges"],
                np.array([1e-20, 0.01, 0.1, 0.2, 1]),
            )
        )

        #
        self.assertTrue("padded_metallicity_distribution_array" in padded_sfr_dict)
        expected_array = np.array(
            [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 0, 0, 0, 0]]
        )
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_metallicity_distribution_array"],
                expected_array,
            )
        )

        #
        self.assertTrue(
            "padded_metallicity_weighted_starformation_rate_array" in padded_sfr_dict
        )
        expected_array = np.array(
            [[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 0, 0, 0, 0]]
        ) * np.array([0, 10, 20, 30, 0])
        self.assertTrue(
            np.array_equal(
                padded_sfr_dict["padded_metallicity_weighted_starformation_rate_array"],
                expected_array,
            )
        )


class test_check_sfr_dict(unittest.TestCase):
    def setUp(self):
        self.sfr_dict = {
            "name": "test_sfr_dict",
            "lookback_time_bin_edges": np.array([1, 2, 3]) * 1e9 * u.yr,
            "starformation_rate_array": np.array([10, 20, 30]) * u.Msun / u.yr,
            "metallicity_bin_edges": np.array([0.01, 0.1, 0.2]),
            "metallicity_distribution_array": np.array([[1, 2, 3], [4, 5, 6]]),
        }

        self.config = default_convolution_config

    def test_check_sfr_dict_with_name(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"

        # No exception should be raised
        check_sfr_dict(
            sfr_dict=self.sfr_dict,
            requires_name=requires_name,
            requires_metallicity_info=requires_metallicity_info,
            time_type=time_type,
            config=self.config,
        )

    def test_check_sfr_dict_without_name(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"

        del self.sfr_dict["name"]  # Removing the name key

        with self.assertRaises(ValueError):
            check_sfr_dict(
                sfr_dict=self.sfr_dict,
                requires_name=requires_name,
                requires_metallicity_info=requires_metallicity_info,
                time_type=time_type,
                config=self.config,
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
                sfr_dict=self.sfr_dict,
                requires_name=requires_name,
                requires_metallicity_info=requires_metallicity_info,
                time_type=time_type,
                config=self.config,
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
                sfr_dict=self.sfr_dict,
                requires_name=requires_name,
                requires_metallicity_info=requires_metallicity_info,
                time_type=time_type,
                config=self.config,
            )

    def test_check_sfr_dict_without_lookback_time_unit(self):
        requires_name = True
        requires_metallicity_info = True
        time_type = "lookback_time"

        self.sfr_dict["lookback_time_bin_edges"] = np.array([1, 2, 3]) * 1e9

        with self.assertRaises(ValueError):
            check_sfr_dict(
                sfr_dict=self.sfr_dict,
                requires_name=requires_name,
                requires_metallicity_info=requires_metallicity_info,
                time_type=time_type,
                config=self.config,
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
                sfr_dict=self.sfr_dict,
                requires_name=requires_name,
                requires_metallicity_info=requires_metallicity_info,
                time_type=time_type,
                config=self.config,
            )

    def test_check_sfr_dict_redshift_correct_bin_edges(self):
        self.sfr_dict["redshift_bin_edges"] = np.array([0, 1, 2])

        requires_name = True
        requires_metallicity_info = True
        time_type = "redshift"

        check_sfr_dict(
            sfr_dict=self.sfr_dict,
            requires_name=requires_name,
            requires_metallicity_info=requires_metallicity_info,
            time_type=time_type,
            config=self.config,
        )
