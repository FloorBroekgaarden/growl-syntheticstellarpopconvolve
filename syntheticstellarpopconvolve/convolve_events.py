"""
Functions to convolve events
"""

import astropy.units as u
import numpy as np
import pandas as pd

from syntheticstellarpopconvolve.general_functions import (
    calculate_digitized_sfr_rates,
    handle_custom_scaling_or_conversion,
)
from syntheticstellarpopconvolve.post_convolution_hook_routines import (
    handle_post_convolution_function,
)


def convolve_events_by_integration_post_convolution_hook_wrapper(
    config,
    sfr_dict,
    data_dict,
    convolution_instruction,
    time_bin_info_dict,
    convolution_results,
    #
    persistent_data=None,
    previous_convolution_results=None,
):
    """
    Function to wrap the post-convolution function call for event-convolution by integration.

    rules:
    - additional data can be added to the convolution_results
    - the number of systems has to be equal to before the post-convolution function.
    """

    #
    name = "convolve-events by integration"

    #
    config["logger"].warning(
        "Handling post-convolution function hook call for {}".format(name)
    )

    #############
    # pre-call setup
    num_systems_before = len(convolution_results[list(convolution_results.keys())[0]])

    #############
    # call hook
    convolution_results = handle_post_convolution_function(
        config=config,
        sfr_dict=sfr_dict,
        data_dict=data_dict,
        time_bin_info_dict=time_bin_info_dict,
        convolution_instruction=convolution_instruction,
        convolution_results=convolution_results,
        name=name,
        #
        persistent_data=persistent_data,
        previous_convolution_results=previous_convolution_results,
    )

    #############
    # check output
    if isinstance(convolution_results, list):
        for convolution_result in convolution_results:

            #############
            # check output
            num_systems_after = len(
                convolution_result[list(convolution_result.keys())[0]]
            )

            #
            if num_systems_before != num_systems_after:
                raise ValueError(
                    "post-convolution function for event-convolution by integration has changed the number of systems stored in the output dict. Due to current data structure decisions this is not supported. Please make sure that the number of systems before and after calling this function stays equal."
                )
    else:
        #############
        # check output
        num_systems_after = len(
            convolution_results[list(convolution_results.keys())[0]]
        )

        #
        if num_systems_before != num_systems_after:
            raise ValueError(
                "post-convolution function for event-convolution by integration has changed the number of systems stored in the output dict. Due to current data structure decisions this is not supported. Please make sure that the number of systems before and after calling this function stays equal."
            )

    return convolution_results


# TODO: describe properly.
# TODO: move to general function
# TODO: rename
def extract_event_data(config, convolution_instruction):
    """
    Function to extract the event-type data from the correct table and store the stuff in the correct column.
    """

    #
    data_dict = {}

    #
    event_df = pd.read_hdf(
        config["output_filename"],
        "/input_data/{}".format(convolution_instruction["input_data_name"]),
    )

    data_column_dict = convolution_instruction["data_column_dict"]

    # add all the columns to the data dictionary. This automatically handles the correct additional columns for the extra weights function
    for column in data_column_dict.keys():
        config["logger"].debug(
            "Extracting {} as the {} data".format(data_column_dict[column], column)
        )

        # if its a string we just assume its the column name
        if isinstance(data_column_dict[column], str):
            data_dict[column] = event_df[data_column_dict[column]].to_numpy()

            #################
            # Handle unit for delay-time
            if column == "delay_time":
                data_dict[column] = (
                    data_dict[column] * config["delay_time_default_unit"]
                )

        elif isinstance(data_column_dict[column], dict):
            # extract data with the explicit column name entry
            data = event_df[data_column_dict[column]["column_name"]].to_numpy()

            #################
            # Handle conversion
            data = handle_custom_scaling_or_conversion(
                config=config,
                data_layer_or_column_dict_entry=data_column_dict[column],
                value=data,
            )

            # Store
            data_dict[column] = data

            #################
            # Handle unit for delay-time
            # TODO: this should just take whatever unit is provided
            if column == "delay_time":
                if "unit" in data_column_dict[column].keys():
                    unit = data_column_dict[column]["unit"]
                else:
                    unit = config["delay_time_default_unit"]

                #
                data_dict[column] = data_dict[column] * unit
        else:
            raise ValueError("input type not supported.")

    ##########
    # If we have binned data we should addd the delay time bin indices to the
    if convolution_instruction["contains_binned_data"]:
        data_dict["delay_time_data_bin_index"] = (
            np.digitize(
                data_dict["delay_time"].to(u.yr),
                convolution_instruction["delay_time_data_bin_info_dict"][
                    "delay_time_data_bin_edges"
                ].to(u.yr),
            )
            - 1
        )

    #
    return config, data_dict, convolution_instruction


def convolve_events_by_integration(
    config,
    sfr_dict,
    data_dict,
    time_bin_info_dict,
    convolution_instruction,
    #
    persistent_data=None,
    previous_convolution_results=None,
):
    """
    Function to convolve events by integration

    This function uses backward convolution, and 'convolution-time' as the time-bin.
    """

    #
    config["logger"].debug(
        "Convolving event-based data {}->{} for {} bin_center {} using integration-based convolution".format(
            convolution_instruction["input_data_name"],
            convolution_instruction["output_data_name"],
            time_bin_info_dict["bin_type"],
            time_bin_info_dict["bin_center"],
        )
    )

    #############
    # Actual convolution (i.e. normalized yield times SFR)
    digitized_sfr_rates = (
        calculate_digitized_sfr_rates(  # TODO: consider renaming this function
            config=config,
            convolution_instruction=convolution_instruction,
            convolution_time_bin_center=time_bin_info_dict["bin_center"],
            data_dict=data_dict,
            sfr_dict=sfr_dict,
        )
    )

    ###############
    # Multiply results by normalized yield of the data and the normalized yield unit.
    convolved_rate_array = digitized_sfr_rates * data_dict["normalized_yield"]

    # Extract normalized yield unit
    normalized_yield_unit = config["default_normalized_yield_unit"]
    if isinstance(
        convolution_instruction["data_column_dict"]["normalized_yield"], dict
    ):
        if "unit" in convolution_instruction["data_column_dict"]["normalized_yield"]:
            normalized_yield_unit = convolution_instruction["data_column_dict"][
                "normalized_yield"
            ]["unit"]

    # Multiply by normalized yield unit
    convolved_rate_array = convolved_rate_array * normalized_yield_unit

    #############
    # Handle multiplication by convolution time-bin size
    # TODO: consider putting this in a separate function
    if config["multiply_by_convolution_time_binsize"]:
        if config["time_type"] == "redshift":
            raise ValueError(
                "Multiplication of yield by convolution time binsizes is not supported currently"
            )

        convolved_rate_array = convolved_rate_array * time_bin_info_dict["bin_size"]

        config["logger"].info(
            "Multiplying the yield by convolution-time binsize {} to {}".format(
                time_bin_info_dict["bin_size"], convolved_rate_array
            )
        )

    #
    convolution_results = {"yield": convolved_rate_array}

    ######
    # Handle post-convolution function
    convolution_results = convolve_events_by_integration_post_convolution_hook_wrapper(
        config=config,
        sfr_dict=sfr_dict,
        data_dict=data_dict,
        time_bin_info_dict=time_bin_info_dict,
        convolution_instruction=convolution_instruction,
        convolution_results=convolution_results,
        #
        persistent_data=persistent_data,
        previous_convolution_results=previous_convolution_results,
    )

    return {"convolution_results": convolution_results}
