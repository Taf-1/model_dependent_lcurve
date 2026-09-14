import logging
import os
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from astropy.table import Table
import astropy.io.ascii as ascii

DATA = Path(__file__).resolve().parent / "data"

class Adjust_Mod_Files:
    BAND_INDEX_MAP = {"u'": 1, "g'": 2, "r'": 3}
    FILT_SUFFIX = {"u'": "us", "g'": "gs", "r'": "rs"}

    def __init__(self, logger: logging.Logger, wd_temp: float, wd_logg: float, comp_temp: float, comp_logg: float, filt: str, tar_name: str, sec_type: str = "MS") -> None:
        self.logger = logger
        self.wd_temp = wd_temp
        self.wd_logg = wd_logg
        self.comp_temp = comp_temp
        self.comp_logg = comp_logg
        self.sec_type = sec_type
        if filt not in self.BAND_INDEX_MAP:
            self.logger.error(f"Invalid filter: {filt}. Must be one of {list(self.BAND_INDEX_MAP)}")
            raise ValueError(f"Invalid filter: {filt}. Must be one of {list(self.BAND_INDEX_MAP)}")
        self.filt = filt
        self.tar_name = tar_name
        self.band_index = self.BAND_INDEX_MAP[filt]
        self.suffix = self.FILT_SUFFIX[filt]

    @staticmethod
    def interp(logger: logging.Logger, points: np.ndarray, values: np.ndarray, query: np.ndarray, label: str) -> np.ndarray:
        val = np.atleast_2d(LinearNDInterpolator(points, values, rescale=True)(query))[0]
        if np.any(np.isnan(val)):
            val = np.atleast_2d(NearestNDInterpolator(points, values)(query))[0]
            logger.debug(f"{label}: outside convex hull, using nearest neighbour")
        return val

    @staticmethod
    def itp(logger: logging.Logger, filt_data: Table, logg: float, temp: float, coef: str) -> tuple[float, float]:
        points = np.column_stack([np.array(filt_data['logg'], dtype=float),
                                  np.array(filt_data['Teff'], dtype=float)])
        values = np.column_stack([np.array(filt_data['y1'], dtype=float),
                                  np.array(filt_data['y2'], dtype=float)])
        y1, y2 = Adjust_Mod_Files.interp(logger, points, values, np.array([[logg, temp]]), coef)
        logger.debug(f"Interpolated to get the {coef} coefficients via the Claret 2-term law")
        return float(y1), float(y2)

    def limb_darkening(self, star_type: str) -> tuple[float, float, float, float]:
        tag = "DA" if star_type == "WD" else "MS"
        temp, logg = (self.wd_temp, self.wd_logg) if star_type == "WD" else (self.comp_temp, self.comp_logg)
        tab = Table.read(DATA / "ld_coeffs" / f"{tag}_LDCs_ucam_{self.suffix}.dat", format='ascii')
        points = np.column_stack([np.array(tab['Teff'], dtype=float),
                                  np.array(tab['log(g)'], dtype=float)])
        values = np.column_stack([np.array(tab[c], dtype=float) for c in ('a1', 'a2', 'a3', 'a4')])
        a1, a2, a3, a4 = self.interp(self.logger, points, values, np.array([[temp, logg]]), f"{tag} ldc")
        return float(a1), float(a2), float(a3), float(a4)

    def wd_limb_darkening(self) -> tuple[float, float, float, float]:
        return self.limb_darkening("WD")

    def comp_limb_darkening(self) -> tuple[float, float, float, float]:
        return self.limb_darkening(self.sec_type)

    def wd_gravity_darkening(self) -> float:
        path = DATA / "gravity_darkening_coeffs" / f"DA_GDCs_{self.suffix}.dat"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run utils.cache_wd_gdcs() once to build it."
            )
        gdcs = ascii.read(path)
        points = np.column_stack([np.array(gdcs['Teff'], dtype=float),
                                  np.array(gdcs['log(g)'], dtype=float)])
        values = np.array(gdcs['y'], dtype=float)
        y = self.interp(self.logger, points, values,
                        np.array([[self.wd_temp, self.wd_logg]]), 'WD gdc')
        return float(np.atleast_1d(y)[0])

    def comp_gravity_darkening(self) -> float:
        tag = "DA" if self.sec_type == "WD" else "MS"
        gdcs = ascii.read(DATA / "gravity_darkening_coeffs" / f"{tag}_GDCs_{self.suffix}.dat")
        points = np.column_stack([np.array(gdcs['Teff'], dtype=float),
                                  np.array(gdcs['log(g)'], dtype=float)])
        values = np.array(gdcs['y'], dtype=float)
        y = self.interp(self.logger, points, values, np.array([[self.comp_temp, self.comp_logg]]), f'{tag} gdc')
        return float(np.atleast_1d(y)[0])

    @staticmethod
    def load_speedyfit_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
        wave, flux = np.loadtxt(path, skiprows=1, usecols=(0, 1), unpack=True)
        return wave, flux

    def transmission(self) -> tuple[np.ndarray, np.ndarray]:
        filt = DATA / "filter_profiles" / f"ucam_{self.suffix}.txt"
        wave, trans = np.loadtxt(filt, usecols=(0, 1), unpack=True)
        self.logger.debug(f"Loaded the wavelength and transmission for filter {filt}")
        return wave, trans

    def beam_factor(self) -> tuple[float, float]:
        SED_spec = [DATA / "SED_data" / f"{self.tar_name}_primary.txt",
                    DATA / "SED_data" / f"{self.tar_name}_companion.txt"]
        wave_filt, trans_filt = self.transmission()
        beam_facts = []
        for spec in SED_spec:
            if not os.path.isfile(spec):
                self.logger.error(f"Missing SED spectrum file: {spec}")
                raise FileNotFoundError(f"Missing SED spectrum file: {spec}")
            wave_spec, flux_spec = self.load_speedyfit_spectrum(spec)
            T = np.interp(wave_spec, wave_filt, trans_filt, left=0.0, right=0.0)
            c_Apers = 2.99792458e18
            nu = c_Apers / wave_spec
            F_nu = flux_spec * wave_spec ** 2 / c_Apers
            order = np.argsort(nu)
            alpha = np.empty_like(wave_spec)
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha[order] = np.gradient(np.log(F_nu[order]), np.log(nu[order]))
            B = 3.0 - alpha
            w = T * wave_spec * flux_spec
            inband = w > 0
            B_mean = np.trapz((w * B)[inband], wave_spec[inband]) / np.trapz(w[inband], wave_spec[inband])
            beam_facts.append(float(B_mean))
        self.logger.debug(f"Calculated photon-weighted <3 - alpha> for both primary and companion: {beam_facts}")
        return beam_facts[0], beam_facts[1]

    def pivot_wave(self) -> float:
        wave, T = self.transmission()
        lambda_pivot = np.sqrt(np.trapz(wave * T, wave) / np.trapz(T / wave, wave))
        self.logger.debug(f"Calculated a pivot wavelength of {lambda_pivot / 10}")
        return float(lambda_pivot / 10)