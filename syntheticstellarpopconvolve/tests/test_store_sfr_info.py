"""
Testcases for store_sfr_info file
"""

import unittest

from syntheticstellarpopconvolve.convolution_general_functions import temp_dir

TMP_DIR = temp_dir(
    "tests", "tests_convolution", "tests_store_sfr_info", clean_path=True
)

if __name__ == "__main__":
    unittest.main()
