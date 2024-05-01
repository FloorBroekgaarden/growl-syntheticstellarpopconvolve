"""
Functions to handle calculating the star formation and meta;licity distributions

TODO: Force these functions to return a certain astropy unit
TODO: go through all papers and find what unit they use
TODO: add https://arxiv.org/abs/2111.13704 (MW models)
TODO: add https://ui.adsabs.harvard.edu/abs/2020ApJ...898...71B/abstract (MW models)
"""

import astropy.units as u
import numpy as np

# def generate_metallicity_sfr_array(
#     config, star_formation_rate_time_distribution_bin_centers, metallicity_centers
# ):
#     """
#     Function that generates the 2d array containing

#     - Time array
#     - available metallicity array

#     First it sets up the empty array

#     Then we loop over all the values of time, generate the metallicity weighting and sfr values and put them into the array

#     TODO: extract arguments to functions to pass things better
#     TODO: clean this function and make the components
#     """

#     ##############
#     # Get the star formation rate: this is an array containing
#     SFR_args = {"config": config, **config["star_formation_rate_distribution_args"]}
#     if config["time_type"] == "lookback_time":
#         SFR_args["lookback_times"] = star_formation_rate_time_distribution_bin_centers
#     elif config["time_type"] == "redshift":
#         SFR_args["redshifts"] = star_formation_rate_time_distribution_bin_centers

#     #
#     starformation_array = config["star_formation_rate_distribution_function"](
#         **SFR_args
#     )

#     ##############
#     # Get the metallicity distribution
#     # TODO: make optional that this is not done at all (if Z_function is None)
#     Z_args = {
#         "config": config,
#         "metallicity_centers": metallicity_centers,
#         **config["metallicity_distribution_args"],
#     }
#     if config["time_type"] == "lookback_time":
#         Z_args["lookback_times"] = star_formation_rate_time_distribution_bin_centers
#     elif config["time_type"] == "redshift":
#         Z_args["redshifts"] = star_formation_rate_time_distribution_bin_centers

#     #
#     metallicity_distribution_array = config["metallicity_distribution_function"](
#         **Z_args
#     )

#     #############
#     # Construct the combined array.

#     # Multiply by sfr:
#     metallicity_weighted_starformation_array = (
#         starformation_array * metallicity_distribution_array.T
#     ).T

#     # TODO: remove this at this location. Modify it at the convolution step
#     # We need to add two empty columns here to make sure the digitise does not multiply the wrong one
#     metallicity_weighted_starformation_array = np.insert(
#         metallicity_weighted_starformation_array, 0, 0, axis=0
#     )
#     metallicity_weighted_starformation_array = np.insert(
#         metallicity_weighted_starformation_array,
#         metallicity_weighted_starformation_array.shape[0],
#         0,
#         axis=0,
#     )

#     #
#     starformation_array = np.insert(starformation_array, 0, 0, axis=0)
#     starformation_array = np.insert(
#         starformation_array,
#         starformation_array.shape[0],
#         0,
#         axis=0,
#     )

#     #
#     return (
#         metallicity_weighted_starformation_array,
#         metallicity_distribution_array,
#         starformation_array,
#     )


def madau_dickinson_sfr(redshifts, a, b, c, d):
    """
    Cosmological star formation rate density from Madau & Dickinson (https://ui.adsabs.harvard.edu/abs/2014ARA%26A..52..415M/abstract)
        {'a': 0.015, 'b': 2.7, 'c': 2.9, 'd': 5.6}
    as a function of redshift.

    Used in Neijsel et al 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3740N/abstract)
    """

    rate = a * np.power(1 + redshifts, b) / (1 + np.power((1 + redshifts) / c, d))

    return rate * (u.Msun / (u.yr * (u.Mpc**3)))


def starformation_rate_distribution_vanSon2023(redshifts):
    """
    Cosmological star formation rate density used in van Son 2021. Based on Madau & Dickinson SFR.
    """

    return madau_dickinson_sfr(redshifts=redshifts, a=0.02, b=1.48, c=4.45, d=5.90)


def mor19_sfr(config, lookback_time):
    """
    Star-formation rate as a function of time (years) since the birth of the milky way, based on Gaia DR2 from Mor et al. 2019 (https://ui.adsabs.harvard.edu/abs/2019A%26A...624L...1M/abstract)

    Input: time since birth of the Galaxy / years
    """

    # print(lookback_time)

    # lookback_time_in_Gyr = lookback_time.to(u.Gyr)

    # hence age in Gyr
    age_Gyr = config["cosmology"].age(0) - lookback_time

    # data only goes back 10Gyr
    lower_age_limit = config["cosmology"].age(0) - 10 * u.Gyr

    # if its further back we just set it to 0
    age_Gyr[age_Gyr < lower_age_limit] = 0

    age_Gyr = age_Gyr.value

    # rough fit to Mor et al. (2019) : Msun/Gyr/pc^2
    sfr = (0.7946) * np.exp((0.2641) * age_Gyr) + (7.3643) * np.exp(
        -((age_Gyr - (2.5566)) ** 2) / (3.036)
    )

    # convert to Msun/year assuming (as Mor+ do)
    # 1Msun/year now == 1.58Msun/Gyr/pc^2 (data point for now)
    sfr *= 1.0 / 1.58

    return sfr * u.Msun / u.yr
