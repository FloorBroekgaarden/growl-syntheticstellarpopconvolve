"""
Main convolution test script
"""

# pylint: disable=W0611
# flake8: noqa
import unittest

from syntheticstellarpopconvolve.tests.test_check_convolution_input_file import (
    test_check_convolution_input_file,
)
from syntheticstellarpopconvolve.tests.test_convolution_calculate_birth_redshift_array import (
    test_calculate_origin_redshift_array,
)
from syntheticstellarpopconvolve.tests.test_convolution_check_config import (
    test_check_convolution_config,
    test_check_convolution_instruction,
    test_check_metallicity,
    test_check_required,
    test_check_sfr_dict,
)
from syntheticstellarpopconvolve.tests.test_convolution_default_settings import (
    test_array_validation,
    test_callable_or_none_validation,
    test_callable_validation,
    test_existing_path_validation,
    test_logger_validation,
    test_unit_validation,
)
from syntheticstellarpopconvolve.tests.test_convolution_general_functions import (
    test_calculate_bincenters,
    test_calculate_digitized_sfr_rates,
    test_calculate_edge_values,
    test_calculate_origin_time_array,
    test_extract_arguments,
    test_generate_group_name,
    test_get_tmp_dir,
    test_handle_custom_scaling_or_conversion,
    test_handle_extra_weights_function,
    test_pad_function,
)
from syntheticstellarpopconvolve.tests.test_convolution_with_ensemble import (
    test_convolution_with_ensemble,
)
from syntheticstellarpopconvolve.tests.test_convolution_with_events import (
    test_convolution_with_events,
)
from syntheticstellarpopconvolve.tests.test_convolve_custom_data import (
    test_custom_convolution_function,
    test_extract_custom_data,
)
from syntheticstellarpopconvolve.tests.test_convolve_ensembles import (
    test__get_ensemble_structure,
    test_attach_endpoints,
    test_check_if_value_layer,
    test_check_if_value_layer_and_get_layer_iterable,
    test_ensemble_convolution_function,
    test_ensemble_handle_marginalisation,
    test_ensemble_handle_SFR_multiplication,
    test_ensemble_marginalise_layer,
    test_extract_endpoints,
    test_extract_ensemble_data,
    test_get_data_layer_dict_values,
    test_get_deepest_data_layer_depth,
    test_get_depth_ensemble_all_endpoints,
    test_get_depth_ensemble_first_endpoint,
    test_get_ensemble_binsizes,
    test_get_ensemble_structure,
    test_get_layer_iterable,
    test_get_max_depth_ensemble,
    test_handle_binsize_multiplication_factor,
    test_invert_data_layer_dict,
    test_multiply_ensemble,
    test_set_endpoints,
    test_shift_data_layer,
    test_shift_layers_dict,
    test_shift_layers_list,
    test_strip_ensemble_endpoints,
)
from syntheticstellarpopconvolve.tests.test_convolve_events import (
    test_event_convolution_function,
    test_extract_event_data,
)
from syntheticstellarpopconvolve.tests.test_convolve_populations import (
    test_generate_data_dict,
    test_pad_sfr_dict,
    test_update_sfr_dict,
)
from syntheticstellarpopconvolve.tests.test_cosmology_utils import (
    test_age_of_universe_to_redshift,
    test_lookback_time_to_redshift,
    test_redshift_to_age_of_universe,
    test_redshift_to_lookback_time,
)
from syntheticstellarpopconvolve.tests.test_extract_population_settings import (
    test_extract_population_settings,
)
from syntheticstellarpopconvolve.tests.test_prepare_output_file import (
    test_prepare_output_file,
)
from syntheticstellarpopconvolve.tests.test_prepare_redshift_interpolator import (
    test_create_interpolation_datasets,
    test_load_interpolation_data,
    test_prepare_redshift_interpolator,
)
from syntheticstellarpopconvolve.tests.test_store_redshift_shell_info import (
    test_create_shell_volume_dict,
    test_store_redshift_shell_info,
)
from syntheticstellarpopconvolve.tests.test_update_convolution_config import (
    test_update_convolution_config,
)

if __name__ == "__main__":
    unittest.main()
