from __future__ import annotations
import logging
from typing import Sequence
import numpy as np
import lcurve
from ldc_gdc_coeffs import Adjust_Mod_Files

class Rust_LCURVE:
    BAND_INDEX_MAP = {"u'": 1, "g'": 2, "r'": 3}
    DISC_COMPS = (
        "rdisc1",
        "height_disc",
        "beta_disc",
        "temp_disc",
        "texp_disc",
        "lin_limb_disc",
        "quad_limb_disc",
    )
    SPOT_COMPS = (
        "radius_spot",
        "length_spot",
        "height_spot",
        "expon_spot",
        "epow_spot",
        "angle_spot",
        "yaw_spot",
        "temp_spot",
        "tilt_spot",
        "cfrac_spot",
    )
    def __init__(self, logger: logging.Logger, mod_file: str, data_file: str, loggs: Sequence[float], filt: str, target_name: str,
                    disc: bool, spot: bool, secondary_eclipse: bool) -> None:
        self.logger = logger
        self.mod_file = mod_file
        self.data_file = data_file
        self.target_name = target_name
        self.binary_model = lcurve.BinaryModel.from_file(mod_file)
        self.prim_logg, self.sec_logg = loggs[0], loggs[1]
        if filt not in self.BAND_INDEX_MAP:
            self.logger.error(
                f"Invalid filter: {filt}. Must be one of {list(self.BAND_INDEX_MAP)}"
            )
            raise ValueError(
                f"Invalid filter: {filt}. Must be one of {list(self.BAND_INDEX_MAP)}"
            )
        self.filt = filt
        self.band_index = self.BAND_INDEX_MAP[filt]
        self.disc = disc
        self.spot = spot
        self.secondary_eclipse = secondary_eclipse

    @staticmethod
    def read_mod_file(config: str, name: str) -> str:
        with open(config) as f:
            for line in f:
                p = line.split()
                if len(p) >= 3 and p[0] == name and p[1] == "=":
                    return p[2]
        raise KeyError(name)

    @staticmethod
    def _fixed(value: float) -> dict:
        return {"value": value, "vary": False, "defined": True}

    def adjust_mod_config(self) -> lcurve.BinaryModel:
        self.logger.debug(f"Adjusting model parameters from: {self.mod_file}")
        model = self.binary_model.model
        t1, t2 = model.t1.value, model.t2.value
        adj_mod = Adjust_Mod_Files(
            self.logger, float(t1), self.prim_logg, float(t2), self.sec_logg,
            self.filt, self.target_name,
        )
        a1, a2, a3, a4 = adj_mod.wd_limb_darkening()
        b1, b2, b3, b4 = adj_mod.comp_limb_darkening()
        gdc1 = adj_mod.wd_gravity_darkening()
        gdc2 = adj_mod.comp_gravity_darkening()
        pivot_wave = adj_mod.pivot_wave()
        beam_factor1, beam_factor2 = adj_mod.beam_factor()
        updates: dict = {
            "ldc1_1": self._fixed(a1),
            "ldc1_2": self._fixed(a2),
            "ldc1_3": self._fixed(a3),
            "ldc1_4": self._fixed(a4),
            "ldc2_1": self._fixed(b1),
            "ldc2_2": self._fixed(b2),
            "ldc2_3": self._fixed(b3),
            "ldc2_4": self._fixed(b4),
            "gravity_dark1": self._fixed(gdc1),
            "gravity_dark2": self._fixed(gdc2),
            "beam_factor1": self._fixed(beam_factor1),
            "beam_factor2": self._fixed(beam_factor2),
            "wavelength": float(pivot_wave),
            "add_disc": bool(self.disc),
            "add_spot": bool(self.spot),
            "eclipse2": bool(self.secondary_eclipse),
        }
        if not model.velocity_scale.defined:
            v0 = model.velocity_scale.value
            if v0 <= 0.0:
                raise ValueError(f"{self.mod_file}: velocity_scale must be > 0 (validate() rejects 0)")
            updates["velocity_scale"] = {"value": v0, "vary": True, "defined": True}
        if self.disc:
            updates.update({comp: {"vary": True} for comp in self.DISC_COMPS})
        if self.spot:
            updates.update({comp: {"vary": True} for comp in self.SPOT_COMPS})
        self.binary_model.update(updates)
        self.logger.debug(
            f"Adjusted model for {self.target_name} in filter {self.filt} "
            f"(disc={self.disc}, spot={self.spot}, "
            f"secondary_eclipse={self.secondary_eclipse})"
        )
        return self.binary_model