"""
Routines for stochastic convolution


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

- this method does not turn things around like the others do. We start at a given lookback time bin for all systems, and we let the systems be born

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

# calculate the formation yield of all the systems
formation_yield = (
    total_star_formation_at_lookback_times[lookback_time_index] * normalized_yield_array
)
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

# Get the indices
index_array = np.arange(len(formation_yield))

integer_formation_indices = np.repeat(index_array, integer_formations)
print("integer_formation_indices", integer_formation_indices)

fractional_formation_indices = index_array[fractional_formations_sampled]
print("fractional_formation_indices", fractional_formation_indices)

combined_indices = np.concatenate(
    [integer_formation_indices, fractional_formation_indices]
)
print("combined_indices", combined_indices)

# Assign random formation times (of system)
sampled_formation_lookback_times = (
    np.random.random(size=size) * bin_sizes[lookback_time_index]
) + lookback_time_bin_edges[lookback_time_index]
print("sampled_formation_lookback_times", sampled_formation_lookback_times)

# TODO: also assign in
# TODO: glue together and store
