"""
Testcases for convolve_custom_data file
"""

import unittest

from syntheticstellarpopconvolve.convolution_general_functions import temp_dir
from syntheticstellarpopconvolve.convolve_custom_data import (
    custom_convolution_function,
    extract_custom_data,
)

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_convolve_custom_data", clean_path=True
)


class test_extract_custom_data(unittest.TestCase):
    """ """

    def test_extract_custom_data(self):
        """ """

        self.assertRaises(NotImplementedError, extract_custom_data, {}, {})


class test_custom_convolution_function(unittest.TestCase):
    """ """

    def test_custom_convolution_function(self):
        """ """

        self.assertRaises(
            NotImplementedError, custom_convolution_function, 0, {}, {}, {}, {}
        )


if __name__ == "__main__":
    unittest.main()
