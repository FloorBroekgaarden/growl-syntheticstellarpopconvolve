"""
Testcases for post_convolution_hook_routines file
"""

import unittest

import numpy as np

from syntheticstellarpopconvolve.general_functions import temp_dir
from syntheticstellarpopconvolve.post_convolution_hook_routines import (  # handle_extra_weights_function,
    extract_arguments,
)

np.random.seed(0)


TMP_DIR = temp_dir(
    "tests",
    "tests_convolution",
    "tests_post_convolution_hook_routines",
    clean_path=True,
)


class test_extract_arguments(unittest.TestCase):
    def test_extract_arguments_1_extra(self):
        def funca(a, b):
            pass

        args = extract_arguments(funca, {"a": 2, "b": 3, "c": 4})
        self.assertEqual(args, {"a": 2, "b": 3})

    def test_extract_arguments_exact(self):
        def funca(a, b):
            pass

        args = extract_arguments(funca, {"a": 2, "b": 3})
        self.assertEqual(args, {"a": 2, "b": 3})

    def test_extract_arguments_default_args_only(self):
        def funcb(a, b, c=3):
            pass

        args = extract_arguments(funcb, {"a": 2, "b": 3})
        self.assertEqual(args, {"a": 2, "b": 3})

    def test_extract_arguments_default_all(self):
        def funcb(a, b, c=3):
            pass

        args = extract_arguments(funcb, {"a": 2, "b": 3, "c": 4})
        self.assertEqual(args, {"a": 2, "b": 3, "c": 4})

    def test_extract_arguments_default_1_extra(self):
        def funcb(a, b, c=3):
            pass

        args = extract_arguments(funcb, {"a": 2, "b": 3, "c": 4, "d": 5})
        self.assertEqual(args, {"a": 2, "b": 3, "c": 4})

    def test_extract_arguments_missing(self):
        def funca(a, b):
            pass

        with self.assertRaises(KeyError):
            extract_arguments(funca, {"a": 2})


if __name__ == "__main__":
    unittest.main()
