"""Routines for stochastic convolution


initial idea with simple situation

sfr [Msun /yr], global
fixed Z

partially grid-like

with a SFR evaluated in lookback time in bins, with t_l,i lookback times and dt_l,i binsize and edges t_l,i-0.5, t_l,i+0.5

in a given bin we have the total mass formed in stars sfr(t_l=t_l,i) * dt_l,i = m_tot,i

now, we have some systems of interest (e.g. dwd), gained through pop-synth simulations. these have a.o. the property normalized yield, i..e number per formed solar mass

Y_j [Msun]

total number of system j sampled;
Y_j * M_tot,i = N_j

if N_j > 1:
- take X systems where X = floor(N_j)
- N_j-x is then < 1
- take random number from uniform dist, P. if P < N_j-x: accept, else not

then we have a bunch of systems (which can include the same system)
but in that array, assign random lookback time between the bin edges

assign radnom position

- this sampling stategy can be multiprocssed easily (lookback time bins)
- can also easily be extended to include metallicity
- naturally handles unequal yield per systems

Notes:
- this method does not turn things around like the others do. We start
at a given lookback time bin for all systems. We sample a set of
systems based on the total starformation within that lookback time
bin, and the normalized yields of the systems. We then assign a birth
lookback time to the systems (taken randomly between the bin edges)
"""

import numpy as np

from syntheticstellarpopconvolve.general_functions import is_mass_unit
from syntheticstellarpopconvolve.post_convolution_hook_routines import (
    handle_post_convolution_function,
)


def convolve_events_by_sampling_post_convolution_hook_wrapper(
    config,
    sfr_dict,
    data_dict,
    time_bin_info_dict,
    convolution_instruction,
    convolution_results,
    #
    persistent_data=None,
    previous_convolution_results=None,
):
    """
    Function to wrap the post-convolution function call for event-convolution by sampling.

    rules:
    - additional data can be added to the result_dict
    - the number of systems can lower than before the call
    """

    #
    name = "convolve-events by sampling"

    #
    config["logger"].warning(
        "Handling post-convolution function hook call for {}".format(name)
    )

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

    return convolution_results


def select_dict_entries_with_new_indices(sampled_data_dict, new_indices):
    """
    Function to select dict entires with new indices
    """

    sampled_data_dict = {
        data_key: sampled_data_dict[data_key][new_indices]
        for data_key in sampled_data_dict.keys()
        if not data_key == "name"
    }

    return sampled_data_dict


def handle_sorting(convolution_results):
    """
    Function to handle sorting
    """

    if isinstance(convolution_results, dict):

        # Sort on indices
        sorted_indices = convolution_results["indices"].argsort()

        # re-select
        convolution_results = select_dict_entries_with_new_indices(
            sampled_data_dict=convolution_results, new_indices=sorted_indices
        )
    else:
        for convolution_result in convolution_results:
            # Sort on indices
            sorted_indices = convolution_result["indices"].argsort()

            # re-select
            select_dict_entries_with_new_indices(
                sampled_data_dict=convolution_result, new_indices=sorted_indices
            )

    return convolution_results


def add_event_lookback_time_and_filter(
    config, data_dict, convolution_instruction, sampled_data_dict
):
    """
    Function to add the event lookback time to the data
    """

    if "delay_time" not in data_dict.keys():
        return sampled_data_dict

    #
    config["logger"].warning("Adding event lookback time.")

    # extract data
    event_delay_times = data_dict["delay_time"]

    # select the event delay-times of the actual sampled systems
    event_delay_times_of_sampled_systems = event_delay_times[
        sampled_data_dict["indices"]
    ]

    # calculate event lookback times
    event_lookback_times = (
        sampled_data_dict["formation_lookback_times"]
        - event_delay_times_of_sampled_systems
    )

    # store in dict
    sampled_data_dict["event_lookback_times"] = event_lookback_times

    # filter out future events
    if convolution_instruction.get("filter_future_events", True):

        local_indices = np.arange(len(event_lookback_times))

        # calculate local indices that include only the events occuring in the past
        past_event_local_indices = local_indices[event_lookback_times > 0]
        future_event_local_indices = local_indices[event_lookback_times < 0]

        #
        config["logger"].warning(
            "Filtering out {} systems that would occur in the future. {} systems are left, and happen in the past".format(
                len(future_event_local_indices), len(past_event_local_indices)
            )
        )

        # updated sampled data dict to include only past-events
        sampled_data_dict = select_dict_entries_with_new_indices(
            sampled_data_dict=sampled_data_dict,
            new_indices=past_event_local_indices,
        )

    return sampled_data_dict


def calculate_total_star_formation_in_bin(
    config, sfr_dict, data_dict, time_bin_info_dict
):
    """
    Function to calculate the total starformation occuring in a particular time
    bin

    if metallicity information is not required, this yields a scalar value

    if it is required, this yields a vector with values matching
    `total_star_formation_mass * (dP/dZ_{j})*dZ_{j}` where Z_{j} is the
    metallicity-bin in which the system falls
    """

    # Unpack
    lookback_time_bin_size = time_bin_info_dict["bin_size"]
    lookback_time_bin_lower_edge = time_bin_info_dict["bin_edge_lower"]
    star_formation_rate_in_lookback_time_bin = sfr_dict["starformation_rate_array"][
        time_bin_info_dict["bin_number"]
    ]

    #
    total_star_formation_in_lookback_time_bin = (
        star_formation_rate_in_lookback_time_bin * lookback_time_bin_size
    )

    # Check if the total star formation is a mass-type value
    if not is_mass_unit(total_star_formation_in_lookback_time_bin):
        raise ValueError(
            "The total star formation in current bin ({}) is not of a mass-type unit. Something wrong with either the sfr ({}) or the time-bin size ({})".format(
                total_star_formation_in_lookback_time_bin,
                star_formation_rate_in_lookback_time_bin,
                lookback_time_bin_size,
            )
        )

    config["logger"].warning(
        "Lower time bin {} upper time bin {} total mass formed {}".format(
            lookback_time_bin_lower_edge,
            lookback_time_bin_lower_edge + lookback_time_bin_size,
            total_star_formation_in_lookback_time_bin,
        )
    )

    ############
    # if we want to include metallicity then for each system we weigh
    # the total starformation rate by a fraction determined by the
    # metallicity bin they fall in

    # make sure that this is all checked better at the start
    if "metallicity_weighted_starformation_rate_array" in sfr_dict:
        config["logger"].warning(
            "Convolution sampling using metallicity-weighted SFR rate array {}".format(
                sfr_dict["metallicity_weighted_starformation_rate_array"][
                    time_bin_info_dict["bin_number"], :
                ]
            )
        )

        # get the indices in the metallicity bins that the system fall into
        metallicity_indices = (
            np.digitize(
                data_dict["metallicity"],
                bins=config["padded_metallicity_bin_edges"],
                right=False,
            )
            - 1
        )

        # Using the metallicity-indices and the time-bin index, select the sfr for each bin (system)
        total_star_formation_in_lookback_time_bin = sfr_dict[
            "padded_metallicity_weighted_starformation_rate_array"
        ][metallicity_indices, time_bin_info_dict["bin_number"] + 1]

    return total_star_formation_in_lookback_time_bin


def sample_systems(
    total_star_formation_in_bin,
    lookback_time_bin_size,
    lookback_time_bin_lower_edge,
    data_dict,
    config,
):
    """
    General function to handle sampling a set of systems based on
    normalized yields and a total mass of stars formed
    """

    ###########
    #
    config["logger"].warning(
        "Convolving through sampling. Using a total of {}".format(
            total_star_formation_in_bin
        )
    )

    ############
    # calculate the formation yield of all the systems
    formation_yield = (
        total_star_formation_in_bin
        * data_dict["normalized_yield"]
        * config["normalized_yield_unit"]
    )

    #
    local_indices = np.arange(len(data_dict["normalized_yield"]))

    ############
    # first sample systems that have a should form at least one time, but only the down-rounded number of times
    integer_formations = np.array(np.floor(formation_yield), dtype=int)
    integer_sampled_formation_indices = np.repeat(local_indices, integer_formations)

    ############
    # then sample using the remainder (all parts with number between 0 and 1)

    # select the remainder
    fractional_formations = formation_yield - integer_formations

    # take a random set to sample the fractional formations
    random_chance = np.random.random(fractional_formations.shape)

    # Sample the indices
    fractional_sampled_formation_indices = local_indices[
        random_chance < fractional_formations
    ]

    ############
    # Combine the sampled indice
    combined_sampled_indices = np.concatenate(
        [integer_sampled_formation_indices, fractional_sampled_formation_indices]
    )

    ############
    # Make a copy of the data dict and select everything using the combined indices
    data_dict_sampled_systems = select_dict_entries_with_new_indices(
        sampled_data_dict=data_dict,
        new_indices=combined_sampled_indices,
    )

    ############
    # Assign random formation times (of system)
    sampled_formation_lookback_times = (
        np.random.random(size=len(combined_sampled_indices)) * lookback_time_bin_size
    ) + lookback_time_bin_lower_edge

    # add to data_dict
    data_dict_sampled_systems["formation_lookback_times"] = (
        sampled_formation_lookback_times
    )

    ############
    #
    config["logger"].warning(
        "Sampled {} systems.".format(len(combined_sampled_indices))
    )

    return data_dict_sampled_systems


def convolve_events_by_sampling(
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
    Function to handle convolution of events by sampling

    This function uses forward convolution, and 'star-formation-time' as the time-bin.

    NOTE: currently only works for lookback-time based sfr
    NOTE: This function does not really do anything useful atm
    """

    if time_bin_info_dict["time_type"] == "redshift":
        raise ValueError(
            "Convolution by sampling for redshift time-types is not supported currently"
        )

    #
    config["logger"].debug(
        "Convolving event-based data {}->{} for {} bin_center {} using sampling-based convolution".format(
            convolution_instruction["input_data_name"],
            convolution_instruction["output_data_name"],
            time_bin_info_dict["bin_type"],
            time_bin_info_dict["bin_center"],
        )
    )

    ##############
    #
    total_star_formation_in_lookback_time_bin = calculate_total_star_formation_in_bin(
        config=config,
        sfr_dict=sfr_dict,
        data_dict=data_dict,
        time_bin_info_dict=time_bin_info_dict,
    )

    # add indices to dict
    data_dict["indices"] = np.arange(len(data_dict["normalized_yield"]))

    #######
    # Generate the samples
    convolution_results = sample_systems(
        total_star_formation_in_bin=total_star_formation_in_lookback_time_bin,
        data_dict=data_dict,
        lookback_time_bin_size=time_bin_info_dict["bin_size"],
        lookback_time_bin_lower_edge=time_bin_info_dict["bin_edge_lower"],
        config=config,
    )

    ######
    # Add event lookback time. If the user provides delay-times for the systems/events,
    # we determine the event times and (by default) filter out anything that happens in the future.
    convolution_results = add_event_lookback_time_and_filter(
        config=config,
        data_dict=data_dict,
        convolution_instruction=convolution_instruction,
        sampled_data_dict=convolution_results,
    )

    ######
    # Handle post-convolution function
    convolution_results = convolve_events_by_sampling_post_convolution_hook_wrapper(
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

    ######
    # Handle sorting on indices
    convolution_results = handle_sorting(convolution_results=convolution_results)

    ###########
    # wrap up

    # delete the normalized yield
    if isinstance(convolution_results, dict):
        del convolution_results["normalized_yield"]
    else:
        for convolution_result in convolution_results:
            del convolution_result["normalized_yield"]

    return {"convolution_results": convolution_results}
