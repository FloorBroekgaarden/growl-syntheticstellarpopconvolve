"""
Functions for on-the-fly convolutions.

This is mostly experimental, but in short:
- based on the total mass formed into stars in a given target convolution bin, and potentially the metallicity distribution,
- the user can provide a call to a population-synthesis code that evolves a population on the fly.

The user is responsible for all the conversions and reweighting here.

what do we need:
- on-the-fly function to convolve

"""

from syntheticstellarpopconvolve.post_convolution_hook_routines import (
    extract_arguments,
    handle_post_convolution_function,
)


def convolve_on_the_fly_post_convolution_hook_wrapper(
    config,
    sfr_dict,
    convolution_instruction,
    time_bin_info_dict,
    convolution_results,
):
    """
    Function to wrap the post-convolution function call for event-convolution by integration.

    rules:
    - additional data can be added to the convolution_results
    - the number of systems has to be equal to before the post-convolution function.
    """

    #
    name = "convolve on-the-fly"

    #
    config["logger"].warning(
        "Handling post-convolution function hook call for {}".format(name)
    )

    #############
    # call hook
    convolution_results = handle_post_convolution_function(
        config=config,
        sfr_dict=sfr_dict,
        data_dict={},
        time_bin_info_dict=time_bin_info_dict,
        convolution_instruction=convolution_instruction,
        convolution_results=convolution_results,
        name=name,
    )

    return convolution_results


def handle_call_on_the_fly_function(
    config, time_bin_info_dict, sfr_dict, convolution_instruction
):
    """
    Function to call an external evolution code to perform on-the-fly evolution
    """

    ######
    # Get quantities
    bin_number = time_bin_info_dict["bin_number"]
    bin_size = time_bin_info_dict["bin_number"]
    bin_lower_edge = time_bin_info_dict["bin_lower_edge"]
    sfr = sfr_dict["starformation_rate_array"][bin_number]
    total_star_formation_in_bin = (
        sfr * bin_size
    )  # TODO: consider forcing this to be a time-like unit

    #
    config["logger"].warning(
        "Lower time bin {} upper time bin {} total mass formed {}".format(
            bin_lower_edge,
            bin_lower_edge + bin_size,
            total_star_formation_in_bin,
        )
    )

    #
    metallicity_weighted_sfr = (
        sfr_dict["metallicity_weighted_starformation_rate_array"][:, bin_number]
        if sfr_dict["include_metallicity_info"]
        else None
    )
    # metallicity_edges = (
    #     sfr_dict["metallicity_bin_edges"]
    #     if sfr_dict["include_metallicity_info"]
    #     else None
    # )

    ######
    # Call user-provided function including sfr info
    on_the_fly_function = convolution_instruction.get("on_the_fly_function", None)

    if on_the_fly_function is not None:

        # Construct what parameters are available for the extra function
        available_parameters = {
            # Standard info
            "config": config,
            "sfr_dict": sfr_dict,
            "time_bin_info_dict": time_bin_info_dict,
            "convolution_instruction": convolution_instruction,
            # Explicit info
            "total_star_formation_in_bin": total_star_formation_in_bin,
            "metallicity_weighted_sfr": metallicity_weighted_sfr
            ** convolution_instruction.get("on_the_fly_function_extra_parameters", {}),
        }

        # Extract the correct things from the available parameters
        on_the_fly_function_args = extract_arguments(
            func=on_the_fly_function,
            arg_dict=available_parameters,
        )

        # Enforce that certain arguments are present:
        if "total_star_formation_in_bin" not in on_the_fly_function_args:
            raise ValueError(
                "`total_star_formation_in_bin` is a required argument in the `on_the_fly_function` call."
            )

        if sfr_dict["include_metallicity_info"]:
            if "metallicity_weighted_sfr" not in on_the_fly_function_args:
                raise ValueError(
                    "`total_star_formation_in_bin` is a required argument in the `on_the_fly_function` call when including metallicity information in the starformation rate dict"
                )

        #
        config["logger"].debug(
            "Handling `on_the_fly_function` function call using function {} and arguments {}".format(
                convolution_instruction["on_the_fly_function"].__name__,
                on_the_fly_function,
            )
        )

        # Call post-convolution function
        convolution_results = on_the_fly_function(**on_the_fly_function_args)

        return convolution_results

    raise ValueError(
        "Can't perform on-the-fly convolution if no `on_the_fly_function` is provided. Please add a function to the `on_the_fly_function` field in the `convolution_instruction` dict"
    )


def convolve_on_the_fly(
    config,
    sfr_dict,
    convolution_instruction,
    data_dict,
    time_bin_info_dict,
):
    """ """

    #
    config["logger"].warning(
        "Performing on-the-fly convolution at bin-center {}".format(
            time_bin_info_dict["bin_center"]
        )
    )

    ######
    #
    convolution_results = handle_call_on_the_fly_function(
        config=config,
        time_bin_info_dict=time_bin_info_dict,
        sfr_dict=sfr_dict,
        convolution_instruction=convolution_instruction,
    )

    ######
    # Handle post-convolution function
    convolution_results = convolve_on_the_fly_post_convolution_hook_wrapper(
        config=config,
        sfr_dict=sfr_dict,
        convolution_instruction=convolution_instruction,
        convolution_results=convolution_results,
    )

    return convolution_results
