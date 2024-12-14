"""
Functions to convolve events
"""

import pandas as pd

from syntheticstellarpopconvolve.convolve_stochastically import sample_systems_main
from syntheticstellarpopconvolve.general_functions import (
    calculate_digitized_sfr_rates,
    handle_custom_scaling_or_conversion,
)
from syntheticstellarpopconvolve.post_convolution_hook_routines import (
    handle_post_convolution_function,
)


def convolve_events_integration_post_convolution_hook_wrapper(
    config,
    job_dict,
    sfr_dict,
    data_dict,
    convolution_instruction,
    result_dict,
):
    """
    Function to wrap the post-convolution function call for event-convolution by integration.

    rules:
    - additional data can be added to the result_dict
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
    num_systems_before = len(result_dict[list(result_dict.keys())[0]])

    #############
    # call hook
    handle_post_convolution_function(
        config=config,
        job_dict=job_dict,
        sfr_dict=sfr_dict,
        data_dict=data_dict,
        convolution_instruction=convolution_instruction,
        result_dict=result_dict,
        name=name,
    )

    #############
    # check output
    num_systems_after = len(result_dict[list(result_dict.keys())[0]])

    #
    if num_systems_before != num_systems_after:
        raise ValueError(
            "post-convolution function for event-convolution by integration has changed the number of systems stored in the output dict. Due to current data structure decisions this is not supported. Please make sure that the number of systems before and after calling this function stays equal."
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

    #
    return config, data_dict, convolution_instruction


def event_convolution_function(
    bin_center, job_dict, config, convolution_instruction, data_dict
):
    """
    Function for the multiprocessing worker to convolve event-based data.

    TODO: implement here the call to sampling-based convolution method with event-based data
    """

    if convolution_instruction["convolution_type"] == "integrate":
        sfr_dict = job_dict["sfr_dict"]

        #
        convolution_time_bin_center = bin_center
        job_dict["convolution_time_bin_center"] = (
            convolution_time_bin_center  # TODO: putting this here isnt the cleanest solution. should be set earlier
        )

        #
        config["logger"].debug(
            "Convolving event-based data {} for bin_center {} using integration-based convolution".format(
                convolution_instruction["input_data_name"], bin_center
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

        #
        convolution_result = {"yield": convolved_rate_array}

        ######
        # Handle post-convolution function
        convolve_events_integration_post_convolution_hook_wrapper(
            config=config,
            job_dict=job_dict,
            sfr_dict=sfr_dict,
            data_dict=data_dict,
            convolution_instruction=convolution_instruction,
            result_dict=convolution_result,
        )

        return {"convolution_result": convolution_result}

    elif convolution_instruction["convolution_type"] == "sample":
        #
        starformation_bin_center = bin_center

        #
        config["logger"].debug(
            "Convolving event-based data {} for bin_center {} using sampling-based convolution".format(
                convolution_instruction["input_data_name"], starformation_bin_center
            )
        )

        # unpack
        sfr_dict = job_dict["sfr_dict"]
        lookback_time_index = job_dict["bin_number"]

        # make sure that this is all checked better at the start
        include_metallicity = False
        if "metallicity_weighted_starformation_rate_array" in sfr_dict:
            include_metallicity = True

        # TODO: allow for sampling with redshift as well
        sampled_data_dict = sample_systems_main(
            config=config,
            sfr_dict=sfr_dict,
            job_dict=job_dict,
            data_dict=data_dict,
            convolution_instruction=convolution_instruction,
            star_formation_rate_in_lookback_time_bin=sfr_dict[
                "starformation_rate_array"
            ][lookback_time_index],
            lookback_time_bin_lower_edge=sfr_dict["lookback_time_bin_edges"][
                lookback_time_index
            ],
            lookback_time_bin_size=sfr_dict["lookback_time_bin_sizes"][
                lookback_time_index
            ],
            include_metallicity=include_metallicity,
        )

        return sampled_data_dict
    else:
        raise ValueError(
            "Convolution type '{}' not supported".format(
                convolution_instruction["convolution_type"]
            )
        )
