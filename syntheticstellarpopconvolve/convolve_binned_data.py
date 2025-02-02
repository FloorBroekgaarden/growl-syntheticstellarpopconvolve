"""
Functions to support convolution of binned data.

Binned data is a new version of ensemble-based data, but inflated. Because
that data usually is binned, and the time-bin spans some range rather than a
distinct point in time, we should provide support for that.

Routine that handles calculating the overlap of a time-bin series with the starformation rate bins
- calculates distances from edges
- calculates overlap fraction of sfr bins
- calculates fraction of time

TODO: allow using CDF to re-scale
"""

import math

import matplotlib.pyplot as plt
import numpy as np


def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def calculate_overlap_fractions(
    shifted_left_time_bin_edge,
    shifted_right_time_bin_edge,
    sfr_bin_sizes,
    sfr_bin_edges,
):
    """
    Function to calculate the overlap
    """

    ##############
    # calculate distances
    left_distances = sfr_bin_edges[1:] - shifted_left_time_bin_edge
    right_distances = shifted_right_time_bin_edge - sfr_bin_edges[:-1]
    # print('left_distances', left_distances)
    # print('right_distances', right_distances)

    ##############
    # mask by negatives
    left_distances[left_distances < 0] = 0
    right_distances[right_distances < 0] = 0

    # print("Masked negatives")
    # print('left_distances', left_distances)
    # print('right_distances', right_distances)

    ##############
    # Construct the combined overlap array
    combined_overlap_array = sfr_bin_sizes.astype(float)
    combined_overlap_array[left_distances == 0] = 0
    combined_overlap_array[right_distances == 0] = 0
    combined_overlap_array[np.nonzero(left_distances)[0][0]] = left_distances[
        np.nonzero(left_distances)[0][0]
    ]
    combined_overlap_array[np.nonzero(right_distances)[0][-1]] = right_distances[
        np.nonzero(right_distances)[0][-1]
    ]  #
    # print("Time-bin {} overlap fraction with SFR_bins:\n\t{}".format(time_bin_i, combined_overlap_array))

    ##############
    # normalize to fraction of the sfr bin
    normalized_combined_overlap_array = combined_overlap_array / sfr_bin_sizes
    # print("Time-bin {} normalized overlap fraction with SFR_bins:\n\t{}".format(time_bin_i, normalized_combined_overlap_array))

    ##############
    # get fraction of time-bin
    time_bin_fraction = combined_overlap_array / time_bin_size_i
    # print("Time-bin {} time bin fraction\n\t{}".format(time_bin_i, time_bin_fraction))

    ##############
    # calcualte cumulative fraction to allow re-weighting with in-bin expected distribution
    cumulative_time_bin_fraction = np.cumsum(time_bin_fraction)

    ##############
    # non-zero sfr-bins. NOTE: these are the indices we should loop over
    non_zero_overlap_with_sfr_bins = np.nonzero(normalized_combined_overlap_array)[0]

    return {
        "combined_overlap_array": combined_overlap_array,
        "normalized_combined_overlap_array": normalized_combined_overlap_array,
        "time_bin_fraction": time_bin_fraction,
        "cumulative_time_bin_fraction": cumulative_time_bin_fraction,
        "non_zero_overlap_with_sfr_bins": non_zero_overlap_with_sfr_bins,
    }


if __name__ == "__main__":

    #
    shift = 2.6

    sfr_bin_edges = np.arange(0, 20, 2)
    sfr_bin_sizes = np.diff(sfr_bin_edges)

    time_bin_edges = np.arange(0, 100, 5)
    time_bin_sizes = np.diff(time_bin_edges)

    #
    left_time_bin_edges = time_bin_edges[:-1]
    right_time_bin_edges = time_bin_edges[1:]

    shifted_left_time_bin_edges = left_time_bin_edges + shift
    shifted_right_time_bin_edges = right_time_bin_edges + shift

    print("sfr_bin_edges", sfr_bin_edges)
    print("shifted_left_time_bin_edges", shifted_left_time_bin_edges)
    print("shifted_right_time_bin_edges", shifted_right_time_bin_edges)

    ##########
    # Loop over the data time-bins
    for time_bin_i, (
        time_bin_size_i,
        shifted_left_time_bin_edge,
        shifted_right_time_bin_edge,
    ) in enumerate(
        list(
            zip(
                time_bin_sizes,
                shifted_left_time_bin_edges,
                shifted_right_time_bin_edges,
            )
        )[:1]
    ):

        print("time bin", time_bin_i)
        print("time bin size", time_bin_size_i)
        print("shifted_left_time_bin_edge", shifted_left_time_bin_edge)
        print("shifted_right_time_bin_edge", shifted_right_time_bin_edge)

        #
        overlap_fractions = calculate_overlap_fractions(
            shifted_left_time_bin_edge=shifted_left_time_bin_edge,
            shifted_right_time_bin_edge=shifted_right_time_bin_edge,
            sfr_bin_sizes=sfr_bin_sizes,
            sfr_bin_edges=sfr_bin_edges,
        )

        # ##
        # #
        # results = (
        #     np.zeros()
        # )  # TODO: make this the same shape as normalized_yield column

        # # TODO: select data indices that coincides with the current data time-bin
        # data_indices_for_current_time_bin

        # ##
        # # loop over
        # for overlap_sfr_bin_index in overlap_fractions[
        #     "non_zero_overlap_with_sfr_bins"
        # ]:

        #     # TODO:
        #     sfr_bin_index = overlap_sfr_bin_index * np.ones(
        #         data_indices_for_current_time_bin
        #     )

        #     # TODO: calculate sfr in those bins (incl metallicity)

        quit()

    plt.plot(sfr_bin_edges, np.ones(sfr_bin_edges.shape), "bo")
    plt.plot(
        shifted_left_time_bin_edges,
        2 * np.ones(shifted_left_time_bin_edges.shape),
        "ro",
    )
    plt.plot(
        shifted_right_time_bin_edges,
        2 * np.ones(shifted_right_time_bin_edges.shape),
        "go",
    )
    plt.ylim(0, 5)
    plt.show()
