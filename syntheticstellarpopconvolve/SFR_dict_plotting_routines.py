def load_mpl_rc():
    import matplotlib as mpl

    # https://matplotlib.org/users/customizing.html
    mpl.rc(
        "axes",
        labelweight="normal",
        linewidth=2,
        labelsize=30,
        grid=True,
        titlesize=40,
        facecolor="white",
    )

    mpl.rc("savefig", dpi=100)

    mpl.rc("lines", linewidth=4, color="g", markeredgewidth=2)

    mpl.rc(
        "ytick",
        **{
            "labelsize": 30,
            "color": "k",
            "left": True,
            "right": True,
            "major.size": 12,
            "major.width": 2,
            "minor.size": 6,
            "minor.width": 2,
            "major.pad": 12,
            "minor.visible": True,
            "direction": "inout",
        },
    )

    mpl.rc(
        "xtick",
        **{
            "labelsize": 30,
            "top": True,
            "bottom": True,
            "major.size": 12,
            "major.width": 2,
            "minor.size": 6,
            "minor.width": 2,
            "major.pad": 12,
            "minor.visible": True,
            "direction": "inout",
        },
    )

    mpl.rc("legend", frameon=False, fontsize=30, title_fontsize=30)

    mpl.rc("contour", negative_linestyle="solid")

    mpl.rc(
        "figure",
        figsize=[16, 16],
        titlesize=30,
        dpi=100,
        facecolor="white",
        edgecolor="white",
        frameon=True,
        max_open_warning=10,
        # autolayout=True
    )

    mpl.rc(
        "legend",
        fontsize=20,
        handlelength=2,
        loc="best",
        fancybox=False,
        numpoints=2,
        framealpha=None,
        scatterpoints=3,
        edgecolor="inherit",
    )

    mpl.rc("savefig", dpi="figure", facecolor="white", edgecolor="white")

    mpl.rc("grid", color="b0b0b0", alpha=0.5)

    mpl.rc("image", cmap="viridis")

    mpl.rc(
        "font",
        # weight='bold',
        serif="Palatino",
        size=20,
    )

    mpl.rc("errorbar", capsize=2)

    mpl.rc("mathtext", default="sf")
