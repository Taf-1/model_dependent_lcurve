from pathlib import Path
import numpy as np
from astropy.io import ascii
from scipy.interpolate import interp1d, LinearNDInterpolator

TRACKS = Path(__file__).resolve().parent / "data" / "cooling_tracks"

he = ascii.read(TRACKS / "He_tracks_thick.dat")
co = ascii.read(TRACKS / "CO_tracks.dat")
one = ascii.read(TRACKS / "ONe_tracks.dat")
baraffe = ascii.read(TRACKS / "Baraffe" / "baraffe.dat")
mass, radius = np.loadtxt(TRACKS / "MdwarfMRrel.dat", unpack=True)
std_mass, std_radius = np.loadtxt(TRACKS / "Mdwarf_stds.dat", unpack=True)

wd_interpolator = {
    "He": LinearNDInterpolator(list(zip(he["Mass"], he["Teff"])),
                               list(he["Radius"]), rescale=True),
    "CO": LinearNDInterpolator(list(zip(co["M"], co["Teff"])),
                               list(co["R"]), rescale=True),
    "ONe": LinearNDInterpolator(list(zip(one["M"], one["Teff"])),
                                list(one["R"]), rescale=True),
}

ms_interpolator = {
    "baraffe": LinearNDInterpolator(
        list(zip(baraffe["M/Ms"], 10 ** (baraffe["logt(yr)"] - 9))),
        list(baraffe["R/Rs"]), rescale=True),
    "empirical": interp1d(mass, radius, bounds_error=False, fill_value=np.nan),
    "std": interp1d(std_mass, std_radius, bounds_error=False, fill_value=np.nan),
}


def get_radius(m: float, teff: float | None = None, star_type: str = "CO",
               relation: str = "empirical", age: float = 5.0,
               factor: float = 1.0) -> float:
    """Radius in R_sun.

    WD: star_type is one of He, CO, ONe and teff is required.
    MS: star_type='MS'; `relation` picks the table, `factor` scales the result
    (used to marginalise over M-dwarf radius inflation).
    """
    if star_type == "MS":
        if relation == "baraffe":
            r = float(ms_interpolator["baraffe"](m, age))
        else:
            r = float(ms_interpolator[relation](m))
    else:
        if teff is None:
            raise ValueError("teff is required for a white dwarf radius")
        r = float(wd_interpolator[star_type](m, teff))

    if not np.isfinite(r):
        raise ValueError(
            f"{star_type} mass-radius relation undefined at m={m:.4f}"
            + (f", teff={teff:.0f}" if teff is not None else "")
        )
    return r * factor