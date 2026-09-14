from __future__ import annotations
import functools
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import G
from astropy.io import ascii
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import brentq

DATA = Path(__file__).resolve().parent / "data"
BAND_SUFFIX = {"u'": "us", "g'": "gs", "r'": "rs", "i'": "is", "z'": "zs"}


def log_g(m: float, r: float) -> float:
    return float(np.log10((G * m * u.M_sun / (r * u.R_sun) ** 2).to_value(u.cm / u.s ** 2)))


def separation(m1: float, m2: float, period: float) -> float:
    acubed = G * (period * u.d) ** 2 * ((m1 + m2) * u.M_sun) / (4 * np.pi ** 2)
    return float((acubed ** (1 / 3)).to_value(u.R_sun))


def xl1(q: float) -> float:
    def f(x):
        return 1.0 / x ** 2 - q / (1.0 - x) ** 2 - (1.0 + q) * x + q
    return float(brentq(f, 1e-8, 1.0 - 1e-8, xtol=1e-14))


def roche_lobe_va(q: float) -> float:
    q13 = q ** (1 / 3)
    q23 = q13 * q13
    return float(0.49 * q23 / (0.6 * q23 + np.log(1.0 + q13)))


def rva_to_rl1(q: float, rva: float) -> float:
    r_lobe_va = roche_lobe_va(q)
    r_lobe_l1 = 1.0 - xl1(q)
    f = min(rva / r_lobe_va, 1.0)
    ratio = 1.0 + f * (r_lobe_va / r_lobe_l1 - 1.0)
    return float(rva / ratio)


def fill_factor(q: float, r2_l1: float) -> float:
    return float(r2_l1 / (1.0 - xl1(q)))


@functools.lru_cache(maxsize=None)
def _tbb_interp(star_type: str, band: str, model: str, instrument: str):
    prefix = "Twd_to_Tbb" if star_type == "WD" else "Tms_to_Tbb"
    tab = ascii.read(DATA / "blackbody_temps" / f"{prefix}_{model}_{instrument}.dat")
    points = np.column_stack([np.asarray(tab["Teff"], float), np.asarray(tab["log(g)"], float)])
    values = np.asarray(tab[f"T_BB_{BAND_SUFFIX[band]}"], float)
    return LinearNDInterpolator(points, values, rescale=True), NearestNDInterpolator(points, values)


def get_tbb(teff: float, logg: float, band: str, star_type: str = "WD",
            model: str = "Koester", instrument: str = "ucam") -> float:
    lin, near = _tbb_interp(star_type, band, model, instrument)
    val = lin(teff, logg)
    if np.isnan(val):
        val = near(teff, logg)
    return float(val)


@functools.lru_cache(maxsize=None)
def _ldc_interp(star_type: str, band: str, instrument: str):
    tag = "DA" if star_type == "WD" else "MS"
    tab = ascii.read(DATA / "ld_coeffs" / f"{tag}_LDCs_{instrument}_{BAND_SUFFIX[band]}.dat")
    points = np.column_stack([np.asarray(tab["Teff"], float), np.asarray(tab["log(g)"], float)])
    values = np.column_stack([np.asarray(tab[c], float) for c in ("a1", "a2", "a3", "a4")])
    return LinearNDInterpolator(points, values, rescale=True), NearestNDInterpolator(points, values)


def get_ldcs(teff: float, logg: float, band: str, star_type: str = "WD",
             instrument: str = "ucam") -> tuple[float, float, float, float]:
    lin, near = _ldc_interp(star_type, band, instrument)
    val = np.atleast_2d(lin(teff, logg))[0]
    if np.any(np.isnan(val)):
        val = np.atleast_2d(near(teff, logg))[0]
    return tuple(float(v) for v in val)


@functools.lru_cache(maxsize=None)
def _gdc_interp(star_type: str, band: str):
    tag = "DA" if star_type == "WD" else "MS"
    path = DATA / "gravity_darkening_coeffs" / f"{tag}_GDCs_{BAND_SUFFIX[band]}.dat"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run physics.cache_wd_gdcs() once to build it.")
    tab = ascii.read(path)
    points = np.column_stack([np.asarray(tab["Teff"], float), np.asarray(tab["log(g)"], float)])
    values = np.asarray(tab["y"], float)
    return LinearNDInterpolator(points, values, rescale=True), NearestNDInterpolator(points, values)


def get_gdc(teff: float, logg: float, band: str, star_type: str = "WD") -> float:
    lin, near = _gdc_interp(star_type, band)
    val = lin(teff, logg)
    if np.isnan(val):
        val = near(teff, logg)
    return float(val)


def cache_wd_gdcs(bands: tuple[str, ...] = ("u'", "g'", "r'")) -> None:
    from astroquery.vizier import Vizier
    Vizier.VIZIER_SERVER = "vizier.cfa.harvard.edu"
    Vizier.ROW_LIMIT = 500000
    cat = Vizier.get_catalogs("J/A+A/634/A93/tabley")[0]
    for band in bands:
        sub = cat[(cat["Filter"] == band) & (cat["Mod"] == "DA")]
        out = Table({"Teff": np.asarray(sub["Teff"], float),
                     "log(g)": np.asarray(sub["logg"], float),
                     "y": np.asarray(sub["y1"], float) + np.asarray(sub["y2"], float)})
        out.write(DATA / "gravity_darkening_coeffs" / f"DA_GDCs_{BAND_SUFFIX[band]}.dat",
                  format="ascii", overwrite=True)


@functools.lru_cache(maxsize=None)
def pivot_wavelength(band: str, instrument: str = "ucam") -> float:
    wave, trans = np.loadtxt(DATA / "filter_profiles" / f"{instrument}_{BAND_SUFFIX[band]}.txt",
                             usecols=(0, 1), unpack=True)
    lam = np.sqrt(np.trapz(wave * trans, wave) / np.trapz(trans / wave, wave))
    return float(lam / 10.0)


WD_MASS_RANGE = {"He": (0.150, 0.500), "CO": (0.200, 1.200), "ONe": (1.060, 1.350)}


def mass_from_radius(r: float, teff: float, star_type: str = "CO") -> float:
    from m_r_tracks import get_radius
    m_lo, m_hi = WD_MASS_RANGE[star_type]

    def radius(m):
        try:
            return get_radius(m, teff, star_type=star_type)
        except ValueError:
            return np.nan

    grid = np.linspace(m_lo, m_hi, 200)
    rad = np.array([radius(m) for m in grid])
    ok = np.isfinite(rad)
    if ok.sum() < 2:
        raise ValueError(f"{star_type} track has no coverage at Teff={teff:.0f}")
    grid, rad = grid[ok], rad[ok]

    if not rad.min() <= r <= rad.max():
        raise ValueError(f"R={r:.5f} at Teff={teff:.0f} is outside the {star_type} track "
                         f"(M {grid.min():.3f}-{grid.max():.3f} gives R {rad.min():.5f}-{rad.max():.5f})")

    i = int(np.argmin(np.abs(rad - r)))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, grid.size - 1)]
    if (radius(lo) - r) * (radius(hi) - r) > 0.0:
        return float(grid[i])
    return float(brentq(lambda m: radius(m) - r, lo, hi, xtol=1e-6))


def logg_from_radius(r: float, teff: float, star_type: str = "CO") -> float:
    return log_g(mass_from_radius(r, teff, star_type), r)


def ms_mass_from_radius(r: float, relation: str = "empirical") -> float:
    from m_r_tracks import TRACKS
    if relation != "empirical":
        raise ValueError("only the empirical M-dwarf relation can be inverted directly")
    m, rad = np.loadtxt(TRACKS / "MdwarfMRrel.dat", unpack=True)
    if not rad.min() <= r <= rad.max():
        raise ValueError(f"R={r:.4f} outside the M-dwarf relation (R {rad.min():.4f}-{rad.max():.4f})")
    return float(np.interp(r, rad, m))