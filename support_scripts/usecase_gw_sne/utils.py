import os

import deepdish as dd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

KAPPA_DEFAULT = 2.9
REDSHIFT_DEFAULT = 0.2


#######################
# primary mass
def get_data_powerlaw_peak_primary_mass(data_root, redshift, limits, kappa):
    """
    Routine to get the data for the powerlaw peak estimates
    """

    # Get file and set limits
    mass_PP_path = os.path.join(
        data_root, "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5"
    )

    # Create mass grid
    mass_1 = np.linspace(2, 100, 1000)
    mass_ratio = np.linspace(0.1, 1, 500)

    # load in the traces.
    # Each entry in lines is p(m1 | Lambda_i) or p(q | Lambda_i)
    # where Lambda_i is a single draw from the hyperposterior
    # The ppd is a 2D object defined in m1 and q
    with open(mass_PP_path, "r") as _data:
        _data = dd.io.load(mass_PP_path)
        lines = _data["lines"]
        ppd = _data["ppd"]

    # Set redshift scaling multiplication factor
    redshift_scaling_factor = (1 + redshift) ** (kappa)

    # marginalize over q to get the ppd in terms of m1 only
    mass_1_ppd = np.trapz(ppd, mass_ratio, axis=0) * redshift_scaling_factor
    CI_down = (
        np.percentile(lines["mass_1"], limits[0], axis=0) * redshift_scaling_factor
    )
    CI_up = np.percentile(lines["mass_1"], limits[1], axis=0) * redshift_scaling_factor

    return_dict = {
        "mass_1": mass_1,
        "mass_1_ppd": mass_1_ppd,
        "mass_1_lines": lines["mass_1"],
        "CI_up": CI_up,
        "CI_down": CI_down,
    }

    return return_dict


def add_primary_mass_distribution_to_figure(
    fig, ax, mass_1, mass_1_ppd, CI_down, CI_up, label=None, fill_between_kwargs=None
):
    """
    Function to add the distribution to the figure
    """

    if fill_between_kwargs is None:
        fill_between_kwargs = {}

    # plot the PPD as a solid line
    ax.semilogy(
        mass_1,
        mass_1_ppd,
        label=label,
        alpha=0.75,
        **fill_between_kwargs,
    )

    # plot the CIs as a filled interval
    ax.fill_between(
        mass_1,
        CI_down,
        CI_up,
        alpha=0.5,
        # label=label,
        **fill_between_kwargs,
    )

    return fig, ax


def add_text_to_primary_mass_distribution_plot(fig, ax, redshift, kappa):
    """
    Function to add text to the primary mass distribution plot
    """

    # Set redshift scaling multiplication factor
    redshift_scaling_factor = (1 + redshift) ** (kappa)

    # with plt.xkcd():
    ax.text(
        20,
        0.5 * 10**-1 * redshift_scaling_factor,
        "GWTC-3\nPL+Peak model",
        # "GWTC-3\nPL+Peak model\n(z={} \kappa={})".format(redshift, kappa),
        rotation=-30,
        horizontalalignment="right",
        verticalalignment="center",
        fontsize=20,
    )

    #
    a = patches.FancyArrowPatch(
        (20, 0.5 * 10**-1 * redshift_scaling_factor),
        (22, 0.65 * 10**-1 * redshift_scaling_factor),
        connectionstyle="arc3,rad=.4",
        arrowstyle=patches.ArrowStyle.Fancy(head_length=8, head_width=12, tail_width=2),
        color="k",
    )
    ax.add_patch(a)

    return fig, ax


def add_confidence_interval_powerlaw_peak_primary_mass(
    fig,
    ax,
    data_root,
    add_text=False,
    label=None,
    fill_between_kwargs=None,
    limits=[5, 95],
    redshift=REDSHIFT_DEFAULT,
    kappa=KAPPA_DEFAULT,
):
    """
    Function to add the confidence interval for the powerlaw + peak model of the GWTC03b data release for the primary mass distribution

    data_root has to contain the file "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5"
    """

    #
    if fill_between_kwargs is None:
        fill_between_kwargs = {}

    # get data
    data_powerlaw_peak_primary_mass = get_data_powerlaw_peak_primary_mass(
        data_root=data_root, redshift=redshift, limits=limits, kappa=kappa
    )

    # unpack
    mass_1 = data_powerlaw_peak_primary_mass["mass_1"]
    mass_1_ppd = data_powerlaw_peak_primary_mass["mass_1_ppd"]
    CI_up = data_powerlaw_peak_primary_mass["CI_up"]
    CI_down = data_powerlaw_peak_primary_mass["CI_down"]

    fig, ax = add_primary_mass_distribution_to_figure(
        fig=fig,
        ax=ax,
        mass_1=mass_1,
        mass_1_ppd=mass_1_ppd,
        CI_down=CI_down,
        CI_up=CI_up,
        label=label,
        fill_between_kwargs=fill_between_kwargs,
    )

    # Add text to plot
    if add_text:
        fig, ax = add_text_to_primary_mass_distribution_plot(
            fig=fig, ax=ax, redshift=redshift, kappa=kappa
        )

    return fig, ax


# # Add confidence interval of observations
# if add_LVK_observations:
#     fig, axes_list[0] = add_confidence_interval_powerlaw_peak_primary_mass(
#         fig=fig,
#         ax=axes_list[0],
#         data_root=os.path.join(os.environ["DATAFILES_ROOT"], "GW"),
#         fill_between_kwargs=plot_settings.get(
#             "observations_fill_between_kwargs", {}
#         ),
#         add_text=plot_settings.get("add_GW_text", False),
#         redshift=redshift_value,
#     )
