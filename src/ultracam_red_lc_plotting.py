from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse as ap

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 35,
    "axes.labelsize": 35,
    "axes.titlesize": 35,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

def arg_parse(argv=None):
    p = ap.ArgumentParser(description=__doc__,
                          formatter_class=ap.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True,
                   help="FITS file with light curves")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = arg_parse()
    filename = args.file.expanduser().resolve()

    bands = ["us", "gs", "rs"]
    labels = {"us": r"$u_s$", "gs": r"$g_s$", "rs": r"$r_s$"}
    colors = {"us": "tab:blue", "gs": "tab:green", "rs": "tab:red"}

    curves = {}

    with fits.open(filename) as hdul:
        extensions = {
            hdu.header["FILTER"].strip(): hdu
            for hdu in hdul[1:]
            if "FILTER" in hdu.header
        }

        for band in bands:
            if band not in extensions:
                raise ValueError(f"Missing {band} extension in {filename}")

            data = extensions[band].data

            time = np.asarray(data["BMJD(TDB)"], dtype=float)
            flux = np.asarray(data["Flux"], dtype=float) * 1000
            error = np.asarray(data["Flux_err"], dtype=float) * 1000
            weight = np.asarray(data["Weight"], dtype=float)

            good = (
                np.isfinite(time)
                & np.isfinite(flux)
                & np.isfinite(error)
                & (error >= 0)
                & (weight > 0)
            )

            if not np.any(good):
                raise ValueError(f"No usable measurements in {band}")

            order = np.argsort(time[good])
            curves[band] = (
                time[good][order],
                flux[good][order],
                error[good][order],
            )

    reference = int(np.floor(min(curves[b][0].min() for b in bands)))

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=True,
        layout="constrained",
    )

    for ax, band in zip(axes, bands):
        time, flux, error = curves[band]

        ax.errorbar(
            time - reference,
            flux,
            yerr=error,
            fmt=".",
            linestyle="none",
            color=colors[band],
            markersize=3,
            elinewidth=0.6,
            capsize=0,
            alpha=0.8,
            rasterized=True,
        )

        ax.text(
            0.97, 0.88,
            labels[band],
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=colors[band],
        )

        ax.tick_params(which="both", top=True, right=True)
        ax.margins(x=0.02)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig.supylabel("Flux density (mJy)")
    axes[-1].set_xlabel(f"BMJD(TDB) − {reference} (days)")

    for extension in ["png", "pdf"]:
        output = filename.with_name(
            f"{filename.stem}_lightcurves.{extension}"
        )
        fig.savefig(output, dpi=300, bbox_inches="tight")

    plt.close(fig)