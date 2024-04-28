"""
Main convolution test script
"""

# pylint: disable=W0611
# flake8: noqa
import unittest

from binarycpython.tests.tests_convolution.test_check_convolution_input_file import (
    test_check_convolution_input_file,
)
from binarycpython.tests.tests_convolution.test_prepare_output_file import (
    test_prepare_output_file,
)
from binarycpython.tests.tests_convolution.tests_cosmology_utils import (
    test_age_of_universe_to_redshift,
    test_lookback_time_to_redshift,
    test_redshift_to_age_of_universe,
    test_redshift_to_lookback_time,
)

if __name__ == "__main__":
    unittest.main()
