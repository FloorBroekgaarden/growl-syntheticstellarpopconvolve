# """Routines for convolution-by-sampling

# initial idea with simple situation

# sfr [Msun /yr], global
# fixed Z

# partially grid-like

# with a SFR evaluated in lookback time in bins, with t_l,i lookback times and dt_l,i binsize and edges t_l,i-0.5, t_l,i+0.5

# in a given bin we have the total mass formed in stars sfr(t_l=t_l,i) * dt_l,i = m_tot,i

# now, we have some systems of interest (e.g. dwd), gained through pop-synth simulations. these have a.o. the property normalized yield, i..e number per formed solar mass

# Y_j [Msun]

# total number of system j sampled;
# Y_j * M_tot,i = N_j

# if N_j > 1:
# - take X systems where X = floor(N_j)
# - N_j-x is then < 1
# - take random number from uniform dist, P. if P < N_j-x: accept, else not

# then we have a bunch of systems (which can include the same system)
# but in that array, assign random lookback time between the bin edges

# assign radnom position

# - this sampling stategy can be multiprocssed easily (lookback time bins)
# - can also easily be extended to include metallicity
# - naturally handles unequal yield per systems

# Notes:
# - this method does not turn things around like the others do. We start
# at a given lookback time bin for all systems. We sample a set of
# systems based on the total starformation within that lookback time
# bin, and the normalized yields of the systems. We then assign a birth
# lookback time to the systems (taken randomly between the bin edges)
# """

# import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve.general_functions import (
    get_normalized_yield_unit,
    has_unit,
)
from syntheticstellarpopconvolve.post_convolution_hook_routines import (
    handle_post_convolution_function,
)


def convolution_by_sampling_post_convolution_hook_wrapper(
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
    name = "convolution by sampling"

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


# def handle_sorting(convolution_results):
#     """
#     Function to handle sorting
#     """

#     if isinstance(convolution_results, dict):

#         # Sort on indices
#         sorted_indices = convolution_results["indices"].argsort()

#         # re-select
#         convolution_results = select_dict_entries_with_new_indices(
#             sampled_data_dict=convolution_results, new_indices=sorted_indices
#         )
#     else:
#         for convolution_result in convolution_results:
#             # Sort on indices
#             sorted_indices = convolution_result["indices"].argsort()

#             # re-select
#             select_dict_entries_with_new_indices(
#                 sampled_data_dict=convolution_result, new_indices=sorted_indices
#             )

#     return convolution_results


# def add_event_lookback_time_and_filter(
#     config, data_dict, convolution_instruction, sampled_data_dict
# ):
#     """
#     Function to add the event lookback time to the data

#     TODO: ensure same units
#     """

#     if "delay_time" not in data_dict.keys():
#         return sampled_data_dict

#     #
#     config["logger"].warning("Adding event lookback time.")

#     # extract data
#     event_delay_times = data_dict["delay_time"].to(u.yr)

#     # select the event delay-times of the actual sampled systems
#     event_delay_times_of_sampled_systems = event_delay_times[
#         sampled_data_dict["indices"]
#     ]

#     #
#     formation_lookback_times = sampled_data_dict["formation_lookback_times"].to(u.yr)

#     # calculate event lookback times
#     event_lookback_times = (
#         formation_lookback_times - event_delay_times_of_sampled_systems
#     )

#     # store in dict
#     sampled_data_dict["event_lookback_times"] = event_lookback_times.to(u.yr)

#     # filter out future events
#     if convolution_instruction["filter_future_events"]:

#         local_indices = np.arange(len(event_lookback_times))

#         # calculate local indices that include only the events occuring in the past
#         past_event_local_indices = local_indices[event_lookback_times > 0]
#         future_event_local_indices = local_indices[event_lookback_times < 0]

#         #
#         config["logger"].warning(
#             "Filtering out {} systems that would occur in the future. {} systems are left, and happen in the past".format(
#                 len(future_event_local_indices), len(past_event_local_indices)
#             )
#         )

#         # updated sampled data dict to include only past-events
#         sampled_data_dict = select_dict_entries_with_new_indices(
#             sampled_data_dict=sampled_data_dict,
#             new_indices=past_event_local_indices,
#         )

#     return sampled_data_dict


def sample_systems(
    total_star_formation_in_bin,
    lookback_time_bin_size,
    lookback_time_bin_lower_edge,
    data_dict,
    config,
    convolution_instruction,
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
    formation_yield = total_star_formation_in_bin * data_dict["normalized_yield"]

    # Extract normalized yield unit
    normalized_yield_unit = get_normalized_yield_unit(config, convolution_instruction)

    # Multiply by normalized yield unit
    formation_yield = formation_yield * normalized_yield_unit

    # force into cgs
    formation_yield = formation_yield.cgs

    # it has to be dimensionless, otherwise its not really a count.
    if has_unit(formation_yield, fail_on_dimensionless=True):
        raise ValueError(
            "Combined formation yield (unit: {}) has to be dimensionless for convolution by sampling. The total star formation in bin ({}) times the normalized yield ({}) should not have a unit anymore.".format(
                formation_yield.unit.to_string(),
                total_star_formation_in_bin.unit.to_string(),
                normalized_yield_unit.unit.to_string(),
            )
        )

    ############
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
    #
    config["logger"].warning(
        "Sampled {} systems.".format(len(combined_sampled_indices))
    )

    return data_dict_sampled_systems


# def convolution_by_sampling(
#     config,
#     sfr_dict,
#     data_dict,
#     time_bin_info_dict,
#     convolution_instruction,
#     #
#     persistent_data=None,
#     previous_convolution_results=None,
# ):
#     """
#     Function to handle convolution of events by sampling

#     This function uses forward convolution, and 'star-formation-time' as the time-bin.

#     NOTE: currently only works for lookback-time based sfr
#     """

#     if time_bin_info_dict["time_type"] == "redshift":
#         raise ValueError(
#             "Convolution by sampling for redshift time-types is not supported currently"
#         )

#     #
#     config["logger"].debug(
#         "Convolving event-based data {}->{} for {} bin_center {} using sampling-based convolution".format(
#             convolution_instruction["input_data_name"],
#             convolution_instruction["output_data_name"],
#             time_bin_info_dict["bin_type"],
#             time_bin_info_dict["bin_center"],
#         )
#     )


#     ##############
#     #
#     total_star_formation_in_lookback_time_bin = calculate_total_star_formation_in_bin(
#         config=config,
#         sfr_dict=sfr_dict,
#         data_dict=data_dict,
#         time_bin_info_dict=time_bin_info_dict,
#     )

#     # add indices to dict
#     data_dict["indices"] = np.arange(len(data_dict["normalized_yield"]))

#     #######
#     # Generate the samples
#     # TODO: handle support for backward sampling here too.
#     convolution_results = sample_systems(
#         total_star_formation_in_bin=total_star_formation_in_lookback_time_bin,
#         data_dict=data_dict,
#         lookback_time_bin_size=time_bin_info_dict["bin_size"],
#         lookback_time_bin_lower_edge=time_bin_info_dict["bin_edge_lower"],
#         convolution_instruction=convolution_instruction,
#         config=config,
#     )


#     # TODO: handle support for backward sampling. in that case the event lookback time is not relevant and we only should add the birth lookback time

#     ######
#     # Add event lookback time. If the user provides delay-times for the systems/events,
#     # we determine the event times and (by default) filter out anything that happens in the future.
#     if convolution_instruction["assign_event_lookback_time"]:
#         convolution_results = add_event_lookback_time_and_filter(
#             config=config,
#             data_dict=data_dict,
#             convolution_instruction=convolution_instruction,
#             sampled_data_dict=convolution_results,
#         )

#     ######
#     # Handle post-convolution function
#     convolution_results = convolution_by_sampling_post_convolution_hook_wrapper(
#         config=config,
#         sfr_dict=sfr_dict,
#         data_dict=data_dict,
#         time_bin_info_dict=time_bin_info_dict,
#         convolution_instruction=convolution_instruction,
#         convolution_results=convolution_results,
#         #
#         persistent_data=persistent_data,
#         previous_convolution_results=previous_convolution_results,
#     )

#     ######
#     # Handle sorting on indices
#     convolution_results = handle_sorting(convolution_results=convolution_results)

#     ###########
#     # wrap up

#     # delete the normalized yield
#     if isinstance(convolution_results, dict):
#         del convolution_results["normalized_yield"]
#     else:
#         for convolution_result in convolution_results:
#             del convolution_result["normalized_yield"]

#     return {"convolution_results": convolution_results}
