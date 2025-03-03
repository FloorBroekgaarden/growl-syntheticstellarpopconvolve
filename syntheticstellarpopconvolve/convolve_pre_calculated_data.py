"""
File that contains main functions related to convolution of pre-calculated data.
"""

from syntheticstellarpopconvolve.calculate_starformation_rate import (
    calculate_starformation,
)
from syntheticstellarpopconvolve.convolution_by_integration import (
    convolution_by_integration_post_convolution_hook_wrapper,
)
from syntheticstellarpopconvolve.convolution_by_sampling import (
    convolution_by_sampling_post_convolution_hook_wrapper,
    sample_systems,
)
from syntheticstellarpopconvolve.general_functions import (
    get_normalized_yield_unit,
    has_unit,
)


def convolve_pre_calculated_data(
    config,
    sfr_dict,
    data_dict,
    time_bin_info_dict,
    convolution_instruction,
    #
    persistent_data,
    previous_convolution_results,
):
    """
    Main function to convolve pre-calculated data.
    """

    #########
    # Handle some choice support

    # We don't support binned data and redshift based time-types yet
    if (
        convolution_instruction["contains_binned_data"]
        and config["time_type"] == "redshift"
    ):
        raise ValueError(
            "Convolving binned data with redshift-based time is currently not supported"
        )

    #########
    # get SFR
    starformation = calculate_starformation(
        config=config,
        convolution_instruction=convolution_instruction,
        data_dict=data_dict,
        sfr_dict=sfr_dict,
        time_bin_info_dict=time_bin_info_dict,
    )

    ##########
    # Calculate actual yield
    # - take star formation values
    # - multiply by normalized yield
    # - (opt) multiply by convolution time-bin width
    normalized_yield = data_dict["normalized_yield"]
    normalized_yield_unit = get_normalized_yield_unit(config, convolution_instruction)
    yield_array = starformation * normalized_yield * normalized_yield_unit

    # Handle multiplication by convolution time-bin size
    # TODO: consider putting this in a separate function
    if config["multiply_by_convolution_time_binsize"]:
        if config["time_type"] == "redshift":
            raise ValueError(
                "Multiplication of yield by convolution time binsizes is not supported currently"
            )

        # TODO: if convolution direction is forward then convolution bin is SFR bin. double check if the user doesnt do this twice.
        yield_array = yield_array * time_bin_info_dict["bin_size"]

        config["logger"].info(
            "Multiplying the yield by convolution-time binsize {} to {}".format(
                time_bin_info_dict["bin_size"], yield_array
            )
        )

    #########
    # Wrap as convolution results
    convolution_results = {"yield": yield_array}

    #########
    # handle choice for sampling actual systems or just use i
    if convolution_instruction["convolution_type"] == "sample":

        ##################
        # check whether the yield is dimensionless

        # force into cgs (basically to ensure that Gyr/yr is seen as dimensionless with a scale)
        yield_array = yield_array.cgs

        # it has to be dimensionless, otherwise its not really a count.
        if has_unit(yield_array, fail_on_dimensionless=True):
            raise ValueError(
                "Combined formation yield (unit: {}) has to be dimensionless for convolution by sampling. The total star formation in bin ({}) times the normalized yield ({}) should not have a unit anymore.".format(
                    yield_array.unit.to_string(),
                    starformation.unit.to_string(),
                    normalized_yield_unit.unit.to_string(),
                )
            )

        # handle sampling
        # TODO: add persistent data and previous conv results?
        convolution_results = sample_systems(
            yield_array=yield_array,
            data_dict=data_dict,
            lookback_time_bin_size=time_bin_info_dict["bin_size"],
            lookback_time_bin_lower_edge=time_bin_info_dict["bin_edge_lower"],
            convolution_instruction=convolution_instruction,
            config=config,
        )

        # ############
        # # Assign random formation times (of system)
        # TODO: activate again
        # sampled_formation_lookback_times = (
        #     np.random.random(size=len(combined_sampled_indices)) * lookback_time_bin_size
        # ) + lookback_time_bin_lower_edge

        # # add to data_dict
        # data_dict_sampled_systems["formation_lookback_times"] = (
        #     sampled_formation_lookback_times
        # )

        # ######
        # # Add event lookback time. If the user provides delay-times for the systems/events,
        # # we determine the event times and (by default) filter out anything that happens in the future.
        # if convolution_instruction["assign_event_lookback_time"]:
        #     convolution_results = add_event_lookback_time_and_filter(
        #         config=config,
        #         data_dict=data_dict,
        #         convolution_instruction=convolution_instruction,
        #         sampled_data_dict=convolution_results,
        #     )

        # handle postconvolution
        convolution_results = convolution_by_sampling_post_convolution_hook_wrapper(
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

    else:
        # handle postconvolution
        convolution_results = convolution_by_integration_post_convolution_hook_wrapper(
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

    #############
    # delete the normalized yield
    if isinstance(convolution_results, dict):
        if "normalized_yield" in convolution_results:
            del convolution_results["normalized_yield"]
    else:
        for convolution_result in convolution_results:
            if "normalized_yield" in convolution_results:
                del convolution_result["normalized_yield"]

    return {"convolution_results": convolution_results}
