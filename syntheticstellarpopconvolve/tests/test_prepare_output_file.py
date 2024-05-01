"""
Testcases for prepare_output_file file
"""

import copy
import os
import unittest

import h5py

from syntheticstellarpopconvolve import convolution_options_defaults
from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.prepare_output_file import prepare_output_file

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "test_prepare_output_file", clean_path=True
)


class test_prepare_output_file(unittest.TestCase):
    """ """

    def setUp(self):
        """
        create input files
        """

        self.config_working = copy.copy(convolution_options_defaults)
        self.config_working["input_filename"] = os.path.join(
            TMP_DIR, "working_file.hdf5"
        )
        self.config_working["output_filename"] = os.path.join(
            TMP_DIR, "output_file.hdf5"
        )

        #
        self.working_hdf5_file = h5py.File(self.config_working["input_filename"], "w")
        self.working_hdf5_file.create_group("input_data")
        self.working_hdf5_file.create_group("config")
        self.working_hdf5_file.create_group("config/population")
        self.working_hdf5_file.close()

    def test_prepare_output_file(self):

        # run preparing of output file
        prepare_output_file(self.config_working)

        #
        self.assertTrue(os.path.isfile(self.config_working["output_filename"]))

        with h5py.File(self.config_working["output_filename"], "r") as output_hdf5file:
            self.assertTrue("convolution" in output_hdf5file["config"].keys())


if __name__ == "__main__":
    unittest.main()
