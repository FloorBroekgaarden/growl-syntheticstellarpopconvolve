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
at a given lookback time bin for all systems. We sample a set of systems based on the total starformation within that lookback time bin, and the normalized yields of the systems. We then assign a birth lookback time to the systems (taken randomly between the bin edges)
"""

import astropy.units as u
import numpy as np

#
lookback_time_index = 5
scale_factor = 1e-8
size = 10

# have some starformation array
lookback_time_bin_edges = (np.arange(0, 10, 1) * u.Gyr).to(u.yr)
starformation_array = (
    0.25 * np.ones(lookback_time_bin_edges.shape[0] - 1) * u.Msun / u.yr
)  # example of a constant star-formation rate. this could be anything of course.
print(starformation_array)

bin_sizes = np.diff(lookback_time_bin_edges)
print(bin_sizes)

#
total_star_formation_at_lookback_times = starformation_array * bin_sizes
print(total_star_formation_at_lookback_times)

#
normalized_yield_array = scale_factor * np.random.random(size=size) * (1 / u.Msun)
print("normalized_yield_array", normalized_yield_array)

data_dict = {}
data_dict["normalized_yield_array"] = normalized_yield_array
data_dict["IDs"] = np.arange(len(normalized_yield_array))


def sample_systems(
    total_star_formation_in_bin,
    lookback_time_bin_size,
    lookback_time_bin_lower_edge,
    data_dict,
):
    """
    General function to handle sampling a set of systems based on
    normalized yields and a total mass of stars formed

    # TODO: we work with indices now, which might not work when we slice and dice.
    # TODO: or we should make sure the indices map back to IDs
    """

    # calculate the formation yield of all the systems
    formation_yield = total_star_formation_in_bin * data_dict["normalized_yield_array"]
    print("formation_yield", formation_yield)

    # select those that have > 1:
    integer_formations = np.array(np.floor(formation_yield), dtype=int)
    print("integer_formations", integer_formations)

    # select the remainder
    fractional_formations = formation_yield - integer_formations
    print("fractional_formations", fractional_formations)

    # take a random set to sample the fractional formations
    random_chance = np.random.random(fractional_formations.shape)
    print("random_chance", random_chance)

    fractional_formations_sampled = random_chance < fractional_formations
    print("fractional_formations_sampled", fractional_formations_sampled)

    #
    integer_formation_IDs = np.repeat(data_dict["IDs"], integer_formations)
    print("integer_formation_indices", integer_formation_IDs)

    fractional_formation_IDs = data_dict["IDs"][fractional_formations_sampled]
    print("fractional_formation_indices", fractional_formation_IDs)

    combined_IDs = np.concatenate([integer_formation_IDs, fractional_formation_IDs])
    print("combined_indices", combined_IDs)

    # Assign random formation times (of system)
    sampled_formation_lookback_times = (
        np.random.random(size=size) * lookback_time_bin_size
    ) + lookback_time_bin_lower_edge
    print("sampled_formation_lookback_times", sampled_formation_lookback_times)

    return combined_IDs, sampled_formation_lookback_times


def sample_systems_main(
    total_star_formation_in_lookback_time_bin,
    lookback_time_bin_size,
    lookback_time_bin_lower_edge,
    data_dict,
    metallicity_distribution_at_lookback_time=None,
    metallicity_bins=None,
):
    """
    Function that handles sampling systems.

    Currently only supports sampling without metallicity dependence

    TODO: add support for metallicity specific sampling
    NOTE: data_dict should contain IDs of some sort
    """

    ############
    # Method 1: no metallicity dependence
    if metallicity_distribution_at_lookback_time is None:

        combined_IDs, sampled_formation_lookback_times = sample_systems(
            total_star_formation_in_bin=total_star_formation_in_lookback_time_bin,
            data_dict=data_dict,
            lookback_time_bin_size=lookback_time_bin_size,
            lookback_time_bin_lower_edge=lookback_time_bin_lower_edge,
        )

        print(combined_IDs, sampled_formation_lookback_times)
    ############
    # Method 2: metallicity dependence
    else:

        # TODO: loop over the metallicity bins
        # TODO: calculate the total mass formed in that metallicity bin
        # TODO: query the data dict for all systems in the current metallicity bin
        # TODO: create a data dict for this particular metallicity
        # TODO: store/append to combined array
        # TODO: we work with indices now, which might not work when we slice and dice.
        # TODO: or we should make sure the indices map back to IDs
        raise ValueError("Sampling with metallicity distribution is not supported yet")


sample_systems_main(
    total_star_formation_in_lookback_time_bin=total_star_formation_at_lookback_times[
        lookback_time_index
    ],
    data_dict=data_dict,
    lookback_time_bin_lower_edge=lookback_time_bin_edges[lookback_time_index],
    lookback_time_bin_size=bin_sizes[lookback_time_index],
    metallicity_distribution_at_lookback_time=None,
    metallicity_bins=None,
)
