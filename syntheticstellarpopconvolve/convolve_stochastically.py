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

TODO: consider putting the configuration in through the convolution instruction rather than the global config.
TODO: allow calcualting the event lookback time and filtering of the events that occur in the future
TODO: implement post-convolution method that allows us to filter the data based on something (like whether they are within some frequency range)
"""

import time
import uuid

import astropy.units as u
import numpy as np

from syntheticstellarpopconvolve.post_convolution_hook_routines import (
    handle_post_convolution_function,
)


def convolve_events_sampling_post_convolution_hook_wrapper(
    config,
    job_dict,
    sfr_dict,
    data_dict,
    convolution_instruction,
    result_dict,
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
    handle_post_convolution_function(
        config=config,
        job_dict=job_dict,
        sfr_dict=sfr_dict,
        data_dict=data_dict,
        convolution_instruction=convolution_instruction,
        result_dict=result_dict,
        name=name,
    )


def select_dict_entries_with_new_indices(sampled_data_dict, new_indices):
    """
    Function to select dict entires with new indices
    """

    sampled_data_dict = {
        data_key: sampled_data_dict[data_key][new_indices]
        for data_key in sampled_data_dict.keys()
    }

    return sampled_data_dict


def handle_sorting(sampled_data_dict):
    """
    Function to handle sorting
    """

    # Sort on indices
    sorted_indices = sampled_data_dict["indices"].argsort()

    # re-select
    sampled_data_dict = select_dict_entries_with_new_indices(
        sampled_data_dict=sampled_data_dict, new_indices=sorted_indices
    )

    return sampled_data_dict


def add_event_lookback_time(
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
        * config["yield_rate_unit"]
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


def sample_systems_main(
    config,
    sfr_dict,
    job_dict,
    convolution_instruction,
    data_dict,
    star_formation_rate_in_lookback_time_bin,
    lookback_time_bin_size,
    lookback_time_bin_lower_edge,
    include_metallicity,
):
    """
    Function that handles sampling systems at a particular lookback time.

    if `include_metallicity` is True we include metallicity and metallicity-dependent starformation in the sampling
    """

    #
    config["logger"].warning(
        "Main sampling through convolution. Will sample systems according to their normalized yield and the total mass formed in stars."
    )

    #
    total_star_formation_in_lookback_time_bin = (
        star_formation_rate_in_lookback_time_bin * lookback_time_bin_size
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
    if include_metallicity:

        if metallicity_bins is None:
            raise ValueError("Please provide metallicity bins")
        config["logger"].warning("Convolution sampling using metallicity distributions")

        # get the indices in the metallicity bins that the elements fall into
        metallicity_indices = (
            np.digitize(
                data_dict["metallicity"],
                bins=config["padded_metallicity_bin_edges"],
                right=False,
            )
            - 1
        )

        # calculate the star formation per system due to them forming with differnt metallicities. This turns 'total_star_formation_in_lookback_time_bin' into an array.
        total_star_formation_in_lookback_time_bin = (
            sfr_dict["padded_metallicity_distribution_array"][metallicity_indices]
            * total_star_formation_in_lookback_time_bin
        )
    else:
        config["logger"].warning("Convolution sampling using metallicity distributions")

    # add indices to dict
    data_dict["indices"] = np.arange(len(data_dict["normalized_yield"]))

    #######
    # Generate the samples
    sampled_data_dict = sample_systems(
        total_star_formation_in_bin=total_star_formation_in_lookback_time_bin,
        data_dict=data_dict,
        lookback_time_bin_size=lookback_time_bin_size,
        lookback_time_bin_lower_edge=lookback_time_bin_lower_edge,
        config=config,
    )

    ######
    # Add event lookback time
    sampled_data_dict = add_event_lookback_time(
        config=config,
        data_dict=data_dict,
        convolution_instruction=convolution_instruction,
        sampled_data_dict=sampled_data_dict,
    )

    ######
    # Handle post-convolution function
    convolve_events_sampling_post_convolution_hook_wrapper(
        config=config,
        job_dict=job_dict,
        sfr_dict=sfr_dict,
        data_dict=data_dict,
        convolution_instruction=convolution_instruction,
        result_dict=sampled_data_dict,
    )

    ######
    # Handle sorting on indices
    sampled_data_dict = handle_sorting(sampled_data_dict=sampled_data_dict)

    ###########
    # wrap up

    # delete the normalized yield
    del sampled_data_dict["normalized_yield"]

    return {"convolution_result": sampled_data_dict}


if __name__ == "__main__":

    import copy
    import json
    import os

    from syntheticstellarpopconvolve import convolve, default_convolution_config
    from syntheticstellarpopconvolve.general_functions import temp_dir

    TMP_DIR = temp_dir("code", "convolve_stochastically", clean_path=True)

    import h5py
    import pandas as pd

    ##################
    # Testing method without metallicity distribution

    time_start = time.time()

    #
    lookback_time_index = 5
    scale_factor = 5e-9
    size = 100

    # have some starformation array
    lookback_time_bin_edges = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)
    starformation_rate_array = (
        0.25 * np.ones(lookback_time_bin_edges.shape[0] - 1) * u.Msun / u.yr
    )  # example of a constant star-formation rate. this could be anything of course.
    # print(starformation_array)

    bin_sizes = np.diff(lookback_time_bin_edges)
    # print(bin_sizes)

    #
    total_star_formation_at_lookback_times = starformation_rate_array * bin_sizes
    # print(total_star_formation_at_lookback_times)

    #
    normalized_yield_array = scale_factor * np.random.random(size=size)
    # print("normalized_yield_array", normalized_yield_array)

    #
    data_dict = {}
    data_dict["normalized_yield_array"] = normalized_yield_array
    data_dict["IDs"] = np.array([uuid.uuid4().hex for _ in range(size)])
    # print(data_dict)

    # #
    # sample_systems_main(
    #     total_star_formation_in_lookback_time_bin=total_star_formation_at_lookback_times[
    #         lookback_time_index
    #     ],
    #     data_dict=data_dict,
    #     lookback_time_bin_lower_edge=lookback_time_bin_edges[lookback_time_index],
    #     lookback_time_bin_size=bin_sizes[lookback_time_index],
    #     metallicity_distribution_at_lookback_time=None,
    #     metallicity_bins=None,
    # )

    ########################
    # use proper setup for convolution

    # create file
    input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5.h5")
    output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5.h5")
    input_hdf5_file = h5py.File(input_hdf5_filename, "w")

    # Create groups main
    input_hdf5_file.create_group("input_data")
    input_hdf5_file.create_group("config")

    # add group for events
    input_hdf5_file.create_group("input_data/events")

    # Write population config to file
    input_hdf5_file.create_dataset("config/population", data=json.dumps({}))

    # close
    input_hdf5_file.close()

    # load into pd
    df = pd.DataFrame.from_dict(data_dict)

    # store the data frame in the hdf5file
    df.to_hdf(input_hdf5_filename, key="input_data/events/stochastic_example")

    #
    convolution_config = copy.copy(default_convolution_config)
    convolution_config["input_filename"] = input_hdf5_filename
    convolution_config["output_filename"] = output_hdf5_filename
    convolution_config["tmp_dir"] = TMP_DIR
    convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
        TMP_DIR, "interpolator_dict.p"
    )
    convolution_config["multiply_by_time_binsize"] = False

    ###
    # convolution instructions
    convolution_config["convolution_instructions"] = [
        {
            "input_data_type": "event",
            "convolution_type": "sample",
            "input_data_name": "stochastic_example",
            "output_data_name": "stochastic_example",
            "ignore_metallicity": True,
            "data_column_dict": {
                # required
                "IDs": "IDs",
                "yield_rate": "normalized_yield_array",
                # # optional*
                # 'metallicity': 'metallicity',
            },
        },
    ]

    #
    convolution_config["time_type"] = "lookback_time"
    convolution_config["convolution_lookback_time_bin_edges"] = (
        np.arange(2, 4, 0.5) * u.Gyr
    )

    # construct the sfr-dict (NOTE: this uses absolute SFR, not metallicity dependent)
    sfr_dict = {}
    sfr_dict["lookback_time_bin_edges"] = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)
    sfr_dict["starformation_rate_array"] = (
        0.25 * np.ones(sfr_dict["lookback_time_bin_edges"].shape[0] - 1) * u.Msun / u.yr
    )  # example of a constant star-formation rate. this could be anything of course.

    # store
    convolution_config["SFR_info"] = sfr_dict

    input_hdf5_file = h5py.File(input_hdf5_filename, "r")

    # convolve
    convolve(config=convolution_config)

    print("finished convolution")
    # Show some of the content
    with h5py.File(convolution_config["output_filename"], "r") as output_hdf5_file:
        # print(output_hdf5_file["output_data/"].keys())
        # print(output_hdf5_file["output_data/event/"].keys())
        # print(output_hdf5_file["output_data/event/stochastic_example/"].keys())
        # print(
        #     output_hdf5_file[
        #         "output_data/event/stochastic_example/stochastic_example"
        #     ].keys()
        # )
        # print(
        #     output_hdf5_file[
        #         "output_data/event/stochastic_example/stochastic_example/convolved_array"
        #     ].keys()
        # )

        # print(
        #     output_hdf5_file[
        #         "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
        #     ].keys()
        # )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["IDs"][()]
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["formation_lookback_times"][()]
        )

    # import astropy.units as u
    # import legwork as lw
    # import numpy as np

    # N = 100
    # m_1 = np.random.uniform(1, 10, N) * u.Msun
    # m_2 = np.random.rand(N) * m_1
    # dist = np.random.uniform(1, 30, N) * u.kpc
    # f_orb_i = 10**(np.random.uniform(-5, -2, N)) * u.Hz
    # ecc_i = np.random.rand(N)

    # sources = lw.source.Source(m_1=m_1, m_2=m_2, ecc=ecc_i, f_orb=f_orb_i, dist=dist,
    #                            interpolate_g=N > 1000)

    # t_evol = np.random.uniform(0.1, 1, N) * u.Myr

    # sources.evolve_sources(t_evol)

    # print(sources.f_orb, sources.ecc)

    quit()

    ##################
    # Testing method with metallicity distribution

    #
    data_dict["metallicity"] = np.random.random(size=size)

    # print("data_dict['metallicity']", data_dict['metallicity'])

    metallicity_distribution_at_lookback_time = np.array([0.25, 0.25, 0.25, 0.25])
    metallicity_bins = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # time_start_convolution = time.time()

    # #
    # sampled_data_dict = sample_systems_main(
    #     total_star_formation_in_lookback_time_bin=total_star_formation_at_lookback_times[
    #         lookback_time_index
    #     ],
    #     data_dict=data_dict,
    #     lookback_time_bin_lower_edge=lookback_time_bin_edges[lookback_time_index],
    #     lookback_time_bin_size=bin_sizes[lookback_time_index],
    #     metallicity_distribution_at_lookback_time=metallicity_distribution_at_lookback_time,
    #     metallicity_bins=metallicity_bins,
    # )

    # time_end_convolution = time.time()

    # print("Total time: {:.2E}".format(time_end_convolution - time_start))
    # print(
    #     "Total time convolution: {:.2E}".format(
    #         time_end_convolution - time_start_convolution
    #     )
    # )
    # print(
    #     "Fractional time convolution: {:.2E}".format(
    #         (time_end_convolution - time_start_convolution)
    #         / (time_end_convolution - time_start)
    #     )
    # )

    # print(sampled_data_dict)

    #################
    # formal convolve with metallicity

    # create file
    input_hdf5_filename = os.path.join(TMP_DIR, "input_hdf5.h5")
    output_hdf5_filename = os.path.join(TMP_DIR, "output_hdf5.h5")
    input_hdf5_file = h5py.File(input_hdf5_filename, "w")

    # Create groups main
    input_hdf5_file.create_group("input_data")
    input_hdf5_file.create_group("config")

    # add group for events
    input_hdf5_file.create_group("input_data/events")

    # Write population config to file
    input_hdf5_file.create_dataset("config/population", data=json.dumps({}))

    # close
    input_hdf5_file.close()

    # load into pd
    df = pd.DataFrame.from_dict(data_dict)

    # store the data frame in the hdf5file
    df.to_hdf(input_hdf5_filename, key="input_data/events/stochastic_example")

    #
    convolution_config = copy.copy(default_convolution_config)
    convolution_config["input_filename"] = input_hdf5_filename
    convolution_config["output_filename"] = output_hdf5_filename
    convolution_config["tmp_dir"] = TMP_DIR
    convolution_config["redshift_interpolator_data_output_filename"] = os.path.join(
        TMP_DIR, "interpolator_dict.p"
    )
    convolution_config["multiply_by_time_binsize"] = False

    ###
    # convolution instructions
    convolution_config["convolution_instructions"] = [
        {
            "input_data_type": "event",
            "convolution_type": "sample",
            "input_data_name": "stochastic_example",
            "output_data_name": "stochastic_example",
            "ignore_metallicity": True,
            "data_column_dict": {
                # required
                "IDs": "IDs",
                "yield_rate": "normalized_yield_array",
                "metallicity": "metallicity",
                # # optional*
                # 'metallicity': 'metallicity',
            },
        },
    ]

    #
    convolution_config["time_type"] = "lookback_time"
    convolution_config["convolution_lookback_time_bin_edges"] = (
        np.arange(2, 4, 0.5) * u.Gyr
    )

    # construct the sfr-dict (NOTE: this uses absolute SFR, not metallicity dependent)
    sfr_dict = {}
    sfr_dict["lookback_time_bin_edges"] = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)
    sfr_dict["starformation_rate_array"] = (
        0.25 * np.ones(sfr_dict["lookback_time_bin_edges"].shape[0] - 1) * u.Msun / u.yr
    )  # example of a constant star-formation rate. this could be anything of course.

    sfr_dict["metallicity_weighted_starformation_rate_array"] = (
        sfr_dict["starformation_rate_array"][:, np.newaxis]
        * metallicity_distribution_at_lookback_time[np.newaxis, :]
    )
    sfr_dict["metallicity_bin_edges"] = metallicity_bins

    # store
    convolution_config["SFR_info"] = sfr_dict

    input_hdf5_file = h5py.File(input_hdf5_filename, "r")

    # convolve
    convolve(config=convolution_config)

    print("finished convolution")
    # Show some of the content
    with h5py.File(convolution_config["output_filename"], "r") as output_hdf5_file:
        print(output_hdf5_file["output_data/"].keys())
        print(output_hdf5_file["output_data/event/"].keys())
        print(output_hdf5_file["output_data/event/stochastic_example/"].keys())
        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example"
            ].keys()
        )
        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array"
            ].keys()
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ].keys()
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["IDs"][()]
        )

        print(
            output_hdf5_file[
                "output_data/event/stochastic_example/stochastic_example/convolved_array/2.25 Gyr"
            ]["formation_lookback_times"][()]
        )
