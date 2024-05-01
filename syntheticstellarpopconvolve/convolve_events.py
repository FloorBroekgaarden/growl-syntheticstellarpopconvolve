"""
Functions to convolve events
"""

import pandas as pd

from syntheticstellarpopconvolve.convolution_general_functions import (
    calculate_digitized_sfr_rates,
    handle_custom_scaling_or_conversion,
    handle_extra_weights_function,
)


def extract_event_data(config, convolution_instruction):
    """
    Function to extract the event-type data from the correct table and store the stuff in the correct column.

    # TODO: describe properly.
    """

    #
    data_dict = {}

    #
    event_df = pd.read_hdf(
        config["output_filename"],
        "/input_data/events/{}".format(convolution_instruction["input_data_name"]),
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
            if column == "delay_time":
                data_dict[column] = (
                    data_dict[column] * config["delay_time_default_unit"]
                )

                if "delay_time_unit" in data_column_dict[column].keys():
                    data_dict[column].to(data_column_dict[column]["delay_time_unit"])

        else:
            raise ValueError("input type not supported.")

    #
    return config, data_dict, convolution_instruction


def event_convolution_function(
    convolution_time_bin_center, job_dict, config, convolution_instruction, data_dict
):
    """
    Function for the multiprocessing worker to convolve event-based data.
    """

    #
    config["logger"].debug(
        "Convolving event-based data {} for bin_center {}".format(
            convolution_instruction["input_data_name"], convolution_time_bin_center
        )
    )

    #############
    # Calculate array-based convolution (i.e. yield/rate times SFR)
    digitized_sfr_rates = calculate_digitized_sfr_rates(
        config=config,
        convolution_time_bin_center=convolution_time_bin_center,
        data_dict=data_dict,
        sfr_dict=job_dict["sfr_dict"],
    )
    convolved_rate_array = (
        digitized_sfr_rates * data_dict["yield_rate"] * config["yield_rate_unit"]
    )

    ################
    # run custom function afterwards
    extra_weights = handle_extra_weights_function(
        config=config,
        convolution_time_bin_center=convolution_time_bin_center,
        convolution_instruction=convolution_instruction,
        data_dict=data_dict,
        output_shape=convolved_rate_array.shape,
    )

    # re-weight
    convolved_rate_array = convolved_rate_array * extra_weights

    # TODO: do something with the units
    # convolved_rate_array_unit = convolved_rate_array.unit
    convolved_rate_array = convolved_rate_array.value

    return {"convolution_result": convolved_rate_array}
