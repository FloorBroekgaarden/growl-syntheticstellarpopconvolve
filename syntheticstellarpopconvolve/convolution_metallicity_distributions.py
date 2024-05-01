"""
Metallicity distribution function from COMPAS
"""

import numpy as np
from scipy.stats import norm as NormDist


def compas_metallicity_distribution(
    config,
    redshifts,
    metallicity_centers,
    mu0=0.035,
    muz=-0.23,
    sigma_0=0.39,
    sigma_z=0.0,
    alpha=0.0,
):
    """
    Calculate the distribution of metallicities at different redshifts using a log skew normal distribution
    the log-normal distribution is a special case of this log skew normal distribution distribution, and is retrieved by setting
    the skewness to zero (alpha = 0).
    Based on the method in Neijssel+19. Default values of mu0=0.035, muz=-0.23, sigma_0=0.39, sigma_z=0.0, alpha =0.0,
    retrieve the dP/dZ distribution used in Neijssel+19

    NOTE: This assumes that metallicities in COMPAS are drawn from a flat in log distribution!

    Args:
        max_redshift       --> [float]          max redshift for calculation
        redshift_step      --> [float]          step used in redshift calculation
        min_logZ_COMPAS    --> [float]          Minimum logZ value that COMPAS samples
        max_logZ_COMPAS    --> [float]          Maximum logZ value that COMPAS samples

        mu0    =  0.035    --> [float]           location (mean in normal) at redshift 0
        muz    = -0.25    --> [float]           redshift scaling/evolution of the location
        sigma_0 = 0.39     --> [float]          Scale (variance in normal) at redshift 0
        sigma_z = 0.00     --> [float]          redshift scaling of the scale (variance in normal)
        alpha   = 0.00    --> [float]          shape (skewness, alpha = 0 retrieves normal dist)

        min_logZ           --> [float]          Minimum logZ at which to calculate dPdlogZ (influences normalization)
        max_logZ           --> [float]          Maximum logZ at which to calculate dPdlogZ (influences normalization)
        step_logZ          --> [float]          Size of logZ steps to take in finding a Z range

    Returns:
        dPdlogZ            --> [2D float array] Probability of getting a particular logZ at a certain redshift
        metallicities      --> [list of floats] Metallicities at which dPdlogZ is evaluated
        p_draw_metallicity --> float            Probability of drawing a certain metallicity in COMPAS (float because assuming uniform)

    TODO: break this apart and use other functions to fix.
    """

    # extract the stuff
    metallicity_distribution_min_value = config["metallicity_distribution_min_value"]
    metallicity_distribution_max_value = config["metallicity_distribution_max_value"]
    metallicity_distribution_resolution = config["metallicity_distribution_resolution"]

    # for compas we convert to log
    min_logZ = np.log(metallicity_distribution_min_value)
    max_logZ = np.log(metallicity_distribution_max_value)

    #
    step_logZ = (max_logZ - min_logZ) / metallicity_distribution_resolution

    ##################################
    # create a range of metallicities (thex-values, or random variables)
    log_metallicities = np.arange(min_logZ, max_logZ + step_logZ, step_logZ)
    metallicities = np.exp(log_metallicities)

    ##################################
    # Log-Linear redshift dependence of sigma
    sigma = sigma_0 * 10 ** (sigma_z * redshifts)

    ##################################
    # Follow Langer & Norman 2007? in assuming that mean metallicities evolve in z as:
    mean_metallicities = mu0 * 10 ** (muz * redshifts)

    # Now we re-write the expected value of ou log-skew-normal to retrieve mu
    beta = alpha / (np.sqrt(1 + (alpha) ** 2))
    PHI = NormDist.cdf(beta * sigma)
    mu_metallicities = np.log(
        mean_metallicities / 2.0 * 1.0 / (np.exp(0.5 * sigma**2) * PHI)
    )

    ##################################
    # probabilities of log-skew-normal (without the factor of 1/Z since this is dp/dlogZ not dp/dZ)
    dPdlogZ = (
        2.0
        / (sigma[:, np.newaxis])
        * NormDist.pdf(
            (log_metallicities - mu_metallicities[:, np.newaxis]) / sigma[:, np.newaxis]
        )
        * NormDist.cdf(
            alpha
            * (log_metallicities - mu_metallicities[:, np.newaxis])
            / sigma[:, np.newaxis]
        )
    )

    ##################################
    # normalise the distribution over al metallicities
    norm = dPdlogZ.sum(axis=-1) * step_logZ
    dPdlogZ = dPdlogZ / norm[:, np.newaxis]

    ##################################
    # Select the metallicities that we have
    dPdLogZ_for_sampled_metallicities = dPdlogZ[
        :, np.digitize(metallicity_centers, metallicities) - 1
    ]

    ##################################
    # Calculate the dlogZ (stepsizes) values, adding one to the end.
    dlogZ_sampled = np.diff(np.log(config["convolution_metallicity_bin_edges"]))

    ##################################
    # Calculate dP/dlogZ * dlogZ
    dP = dPdLogZ_for_sampled_metallicities * dlogZ_sampled

    #
    return dP


def mean_metallicity(z, z0, alpha):
    """
    Function for mean metallicity
    """

    return z0 * np.power(10, alpha * z)


def mean_mu(z, z0, alpha, sigma):
    """
    Function to calculate mu
    """

    return np.log(mean_metallicity(z, z0, alpha)) - np.power(sigma, 2) / 2


def metallicity_distribution_lognormal(Z, z, z0, alpha, sigma):
    """
    Function to calculate the metallicity distribution on a grid of redshifts and metallicity according to a lognormal distribution.

    See Neijsel et al 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3740N/abstract) section 4.

    Function returns $dP(z)/dZ$
    """

    return (1 / (Z * sigma * np.power(2 * np.pi, 0.5))) * np.exp(
        -np.power(np.log(Z) - mean_mu(z, z0, alpha, sigma), 2)
        / (2 * np.power(sigma, 2))
    )


# def metallicity_distribution_Neijsel19(Z, z, z0, alpha, sigma):
#     """
#     Function to calculate the metallicity distribution fraction at a given redshift according to Neijsel et al 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3740N/abstract)
#     """


#             # # metallicity distribution settings for Neijssel 2019
#             # 'mu0': 0.035,
#             # 'muz': -0.23,
#             # 'sigma_0': 0.39,
#             # 'sigma_z': 0.0
#             # 'alpha': 0.0,


#     metallicity_distribution_lognormal(Z=Z, z=z, z0, alpha, sigma)


def metallicity_distribution_vanSon2022(metallicities, redshifts):
    """
    Function to calculate the metallicity distribution fraction as a function of redshift according to van Son et al. 2022

    TODO: bind everything in here
    """

    #     # metallicity distribution settings for van Son 2021
    # "mu0": 0.025,
    # "muz": -0.05,
    # "sigma_0": 1.125,
    # "sigma_z": 0.05,
    # "alpha": -1.77,

    pass


def metallicity_distribution_dummy(constant=1):
    """
    Dummy metallicity distribution, based on the same functional form as neijsel et al 2019 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.3740N/abstract)
    """

    # override redshift to have it not change

    return constant
