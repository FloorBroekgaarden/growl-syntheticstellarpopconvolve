"""
Testcases for post_convolution_hook_routines file
"""

import unittest

from syntheticstellarpopconvolve.general_functions import temp_dir
from syntheticstellarpopconvolve.post_convolution_hook_routines import (  # handle_extra_weights_function,
    extract_arguments,
)

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


# class test_handle_extra_weights_function(unittest.TestCase):
#     def setUp(self):
#         #
#         input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5_sfr_only.h5")
#         output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5_sfr_only.h5")

#         ##############
#         # SET UP DATA
#         self.dummy_data = {
#             "delay_time": np.array([0, 1, 2, 3]),
#             "probability": np.array([1, 2, 3, 4]),
#         }
#         dummy_df = pd.DataFrame.from_records(self.dummy_data)

#         #############
#         # create input HDF5 file
#         with h5py.File(input_hdf5_filename, "w") as input_hdf5_file:

#             ######################
#             # Create groups
#             input_hdf5_file.create_group("input_data")
#             input_hdf5_file.create_group("input_data/events")
#             input_hdf5_file.create_group("config")

#             ###############
#             # Readout population settings
#             population_settings_filename = pkg_resources.resource_filename(
#                 "syntheticstellarpopconvolve",
#                 "example_data/example_population_settings.json",
#             )

#             with open(population_settings_filename, "r") as f:
#                 population_settings = json.loads(f.read())

#             # Delete some stuff from the settings
#             del population_settings["population_settings"]["bse_options"]["metallicity"]

#             # Write population config to file
#             input_hdf5_file.create_dataset(
#                 "config/population", data=json.dumps(population_settings)
#             )

#         ##############
#         # Store data in pandas
#         dummy_df.to_hdf(input_hdf5_filename, key="input_data/events/{}".format("dummy"))

#         #
#         self.convolution_config = copy.copy(default_convolution_config)

#         # Set up SFR
#         self.convolution_config["SFR_info"] = {
#             "lookback_time_bin_edges": np.array([0, 1, 2, 3, 4, 5]),
#             "starformation_rate_array": np.array([1, 1, 1, 1, 1])
#             * u.Msun
#             / u.yr
#             / u.Gpc**3,
#         }

#         # set up convolution bins
#         self.convolution_config["convolution_time_bin_edges"] = np.array(
#             [0, 1, 2, 3, 4]
#         )

#         # lookback time convolution only
#         self.convolution_config["time_type"] = "lookback_time"

#         #
#         self.convolution_config["input_filename"] = input_hdf5_filename
#         self.convolution_config["output_filename"] = output_hdf5_filename

#         self.convolution_config["redshift_interpolator_data_output_filename"] = (
#             os.path.join(TMP_DIR, "interpolator_dict.p")
#         )

#         #
#         self.convolution_config["convolution_instructions"] = [
#             {
#                 "input_data_type": "event",
#                 "input_data_name": "dummy",
#                 "output_data_name": "dummy",
#                 "data_column_dict": {
#                     "delay_time": "delay_time",
#                     "yield_rate": "probability",
#                 },
#                 "ignore_metallicity": True,
#             },
#         ]

#         #
#         self.convolution_config["tmp_dir"] = os.path.join(TMP_DIR, "tmp")

#         #
#         prepare_output_file(config=self.convolution_config)

#     def test_handle_extra_weights_function_normal(self):
#         def extra_weights_function(config, data_dict):
#             return np.zeros(data_dict["yield_rate"].shape)

#         convolution_instruction = self.convolution_config["convolution_instructions"][0]
#         convolution_instruction["extra_weights_function"] = extra_weights_function

#         self.dummy_data["yield_rate"] = self.dummy_data["probability"]

#         extra_weights = handle_extra_weights_function(
#             config=self.convolution_config,
#             bin_center=0.2,
#             convolution_instruction=convolution_instruction,
#             sfr_dict={},
#             data_dict=self.dummy_data,
#             output_shape=self.dummy_data["yield_rate"].shape,
#         )

#         #
#         np.testing.assert_array_equal(
#             extra_weights, np.zeros(self.dummy_data["yield_rate"].shape)
#         )

#     def test_handle_extra_weights_function_extra_input_fail(self):
#         # test should fail since we don't provide the input for the function
#         def extra_weights_function(config, data_dict, a, b):
#             return np.zeros(data_dict["yield_rate"].shape) + a + b

#         convolution_instruction = self.convolution_config["convolution_instructions"][0]
#         convolution_instruction["extra_weights_function"] = extra_weights_function

#         #
#         self.dummy_data["yield_rate"] = self.dummy_data["probability"]

#         with self.assertRaises(KeyError):
#             _ = handle_extra_weights_function(
#                 config=self.convolution_config,
#                 bin_center=0.2,
#                 convolution_instruction=convolution_instruction,
#                 sfr_dict={},
#                 data_dict=self.dummy_data,
#                 output_shape=self.dummy_data["yield_rate"].shape,
#             )

#     def test_handle_extra_weights_function_extra_function_fail(self):
#         # test should fail since the function does not return anything
#         def extra_weights_function(config, data_dict, a, b):
#             pass

#         convolution_instruction = self.convolution_config["convolution_instructions"][0]
#         convolution_instruction["extra_weights_function"] = extra_weights_function
#         convolution_instruction["extra_weights_function_additional_parameters"] = {
#             "a": 10,
#             "b": 2.5,
#         }

#         #
#         self.dummy_data["yield_rate"] = self.dummy_data["probability"]

#         with self.assertRaises(ValueError):
#             _ = handle_extra_weights_function(
#                 config=self.convolution_config,
#                 bin_center=0.2,
#                 convolution_instruction=convolution_instruction,
#                 sfr_dict={},
#                 data_dict=self.dummy_data,
#                 output_shape=self.dummy_data["yield_rate"].shape,
#             )

#     def test_handle_extra_weights_function_extra_input_pass(self):
#         # test should fail since we don't provide the input for the function
#         def extra_weights_function(config, data_dict, a, b):
#             return np.zeros(data_dict["yield_rate"].shape) + a + b

#         convolution_instruction = self.convolution_config["convolution_instructions"][0]
#         convolution_instruction["extra_weights_function"] = extra_weights_function
#         convolution_instruction["extra_weights_function_additional_parameters"] = {
#             "a": 10,
#             "b": 2.5,
#         }

#         #
#         self.dummy_data["yield_rate"] = self.dummy_data["probability"]

#         extra_weights = handle_extra_weights_function(
#             config=self.convolution_config,
#             bin_center=0.2,
#             convolution_instruction=convolution_instruction,
#             sfr_dict={},
#             data_dict=self.dummy_data,
#             output_shape=self.dummy_data["yield_rate"].shape,
#         )

#         np.testing.assert_array_equal(
#             extra_weights, np.zeros(self.dummy_data["yield_rate"].shape) + 12.5
#         )

#     def test_handle_extra_weights_function_no_function_shape_fail(self):
#         def extra_weights_function(config, data_dict, a, b):
#             return np.zeros(data_dict["yield_rate"].shape) + a + b

#         # test should fail since the output shape doesnt match
#         convolution_instruction = self.convolution_config["convolution_instructions"][0]
#         convolution_instruction["extra_weights_function"] = extra_weights_function
#         convolution_instruction["extra_weights_function_additional_parameters"] = {
#             "a": 10,
#             "b": 2.5,
#         }

#         #
#         self.dummy_data["yield_rate"] = self.dummy_data["probability"]

#         with self.assertRaises(ValueError):
#             _ = handle_extra_weights_function(
#                 config=self.convolution_config,
#                 bin_center=0.2,
#                 convolution_instruction=convolution_instruction,
#                 sfr_dict={},
#                 data_dict=self.dummy_data,
#                 output_shape=np.shape([1]),
#             )

#     def test_handle_extra_weights_function_no_function(self):
#         # test should fail since the output shape doesnt match
#         convolution_instruction = self.convolution_config["convolution_instructions"][0]

#         #
#         self.dummy_data["yield_rate"] = self.dummy_data["probability"]

#         extra_weights = handle_extra_weights_function(
#             config=self.convolution_config,
#             bin_center=0.2,
#             convolution_instruction=convolution_instruction,
#             sfr_dict={},
#             data_dict=self.dummy_data,
#             output_shape=np.shape([1]),
#         )

#         #
#         np.testing.assert_array_equal(extra_weights, np.ones(np.shape([1])))
