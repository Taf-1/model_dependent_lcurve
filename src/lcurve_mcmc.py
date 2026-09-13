import argparse
import logging
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
import emcee
import numpy as np
from ruamel.yaml import YAML
import lcurve
import utils as ph
from m_r_tracks import get_radius
from lcurve_model import Rust_LCURVE
from lc_logger import lcurve_logging

class Lcurve_MCMC:

    BLOB_NAMES = ("logg1", "logg2", "r1", "r2", "a", "ffac", "rva2")

    def __init__(self, logger: logging.Logger, config: dict, build_model_files: bool = False) -> None:
        self.logger = logger
        self.config = config
        self.target_name = config["run_name"]
        self.template_mod = config["model_file"]
        self.period = float(config["period"])
        self.instrument = config.get("instrument", "ucam")
        self.bands = list(config["light_curves"].keys())
        self.parameter_names = tuple(config["params"].keys())
        self.bounds = config["param_bounds"]
        if build_model_files:
            self.setup_model_files()
        self.beam_factors = {b: self.read_beam_factors(b) for b in self.bands}
        self.data = {b: self.load_data_file(p) for b, p in config["light_curves"].items()}
        self.binary_model = lcurve.BinaryModel.from_file(self.mod_path(self.bands[0]))
        self.set_core_parameters()
        self.geometry_dirty = True
        self.blobs = [np.nan] * len(self.BLOB_NAMES)

    def mod_path(self, band: str) -> str:
        tag = band.replace("'", "")
        return f"model_files/{self.target_name}_{tag}.mod"

    def setup_model_files(self) -> None:
        os.makedirs("model_files", exist_ok=True)
        loggs = self.config["seed_loggs"]
        opts = self.config.get("components", {})
        for band in self.bands:
            rl = Rust_LCURVE(self.logger, self.template_mod, self.config["light_curves"][band],
                             loggs, band, self.target_name,
                             disc=opts.get("disc", False), spot=opts.get("spot", False),
                             secondary_eclipse=opts.get("secondary_eclipse", False))
            rl.adjust_mod_config()
            rl.binary_model.model.write(self.mod_path(band))
            self.logger.info(f"Wrote {self.mod_path(band)}")

    def read_beam_factors(self, band: str) -> dict:
        path = self.mod_path(band)
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing - run with build_model_files=True first")
        m = lcurve.Model.from_file(path)
        return {"beam_factor1": m.beam_factor1.value, "beam_factor2": m.beam_factor2.value}

    def set_core_parameters(self) -> None:
        ms = self.config["model_settings"]
        self.binary_model.update({
            "period": self.period,
            "tperiod": self.period,
            "delta_phase": float(ms["delta_phase"]),
            "nlat1f": int(ms["primary_fine_resolution"]),
            "nlat2f": int(ms["secondary_fine_resolution"]),
            "nlat1c": int(ms["primary_coarse_resolution"]),
            "nlat2c": int(ms["secondary_coarse_resolution"]),
            "npole": bool(ms["true_north_pole"]),
            "roche1": bool(ms["primary_roche"]),
            "roche2": bool(ms["secondary_roche"]),
            "eclipse1": bool(ms["primary_eclipse"]),
            "eclipse2": bool(ms["secondary_eclipse"]),
        })

    def load_data_file(self, path: str) -> dict:
        self.logger.debug(f"Inspecting data file: {path}")
        try:
            columns = np.loadtxt(path, unpack=True)
        except Exception as e:
            self.logger.error(f"Error reading data file: {e}")
            raise
        if len(columns) != 6:
            raise ValueError(f"{path}: expected 6 columns "
                             f"(time, t_exp, flux, f_err, weight, n_div), got {len(columns)}")
        keys = ("time", "t_exp", "flux", "f_err", "weight", "n_div")
        out = {k: np.ascontiguousarray(c, dtype=np.float64) for k, c in zip(keys, columns)}
        self.logger.debug(f"Loaded {out['time'].size} points from {path}")
        return out

    def set_parameter_vector(self, params) -> None:
        for name, value in zip(self.parameter_names, params):
            setattr(self, name, float(value))
        self.geometry_dirty = True

    def in_bounds(self) -> bool:
        for name in self.parameter_names:
            lo, hi = self.bounds[name]
            if not (lo < getattr(self, name) < hi):
                return False
        return True

    def geometry(self) -> dict:
        q = self.m2 / self.m1
        self.r1 = get_radius(self.m1, self.t1, star_type=self.config["wd_core_comp"])
        self.r2 = get_radius(self.m2, star_type="MS", relation=self.config["secondary_mr"])
        self.logg1 = ph.log_g(self.m1, self.r1)
        self.logg2 = ph.log_g(self.m2, self.r2)
        self.a = ph.separation(self.m1, self.m2, self.period)
        r1_a = self.r1 / self.a
        r2_a = ph.rva_to_rl1(q, self.r2 / self.a)
        self.ffac = ph.fill_factor(q, r2_a)
        phase1 = float(np.arcsin(r1_a + r2_a) / (2 * np.pi) + 0.001)
        return {"q": q, "iangle": self.incl, "r1": r1_a, "r2": r2_a,
                "t0": self.t0, "phase1": phase1, "phase2": 0.5 - phase1}

    def continuum(self, band: str) -> dict:
        a1, a2, a3, a4 = ph.get_ldcs(self.t1, self.logg1, band, "WD", self.instrument)
        b1, b2, b3, b4 = ph.get_ldcs(self.t2, self.logg2, band, "MS", self.instrument)
        pars = {
            "t1": ph.get_tbb(self.t1, self.logg1, band, "WD", self.config["wd_model"], self.instrument),
            "t2": ph.get_tbb(self.t2, self.logg2, band, "MS", self.config["ms_model"], self.instrument),
            "wavelength": ph.pivot_wavelength(band, self.instrument),
            "gravity_dark1": ph.get_gdc(self.t1, self.logg1, band, "WD"),
            "gravity_dark2": ph.get_gdc(self.t2, self.logg2, band, "MS"),
            "ldc1_1": a1, "ldc1_2": a2, "ldc1_3": a3, "ldc1_4": a4,
            "ldc2_1": b1, "ldc2_2": b2, "ldc2_3": b3, "ldc2_4": b4,
        }
        pars.update(self.beam_factors[band])
        return pars

    def get_value(self, band: str):
        pars = {}
        if self.geometry_dirty:
            pars.update(self.geometry())
            self.geometry_dirty = False
        pars.update(self.continuum(band))
        self.binary_model.update(pars)
        d = self.data[band]
        lc = self.binary_model.compute_light_curve(d["time"], d["t_exp"], d["n_div"],
                                                   d["flux"], d["f_err"], d["weight"])
        self.blobs = [self.logg1, self.logg2, self.r1, self.r2, self.a, self.ffac, lc.rva2]
        return lc

    def log_prior(self) -> float:
        if not self.in_bounds():
            return -np.inf
        val = 0.0
        for name, spec in self.config.get("priors", {}).items():
            if name not in self.parameter_names:
                continue
            kind, *args = spec
            x = getattr(self, name)
            if kind == "gauss":
                mu, sigma = args
                val += -0.5 * ((x - mu) / sigma) ** 2
            elif kind == "uniform":
                lo, hi = args
                if not (lo < x < hi):
                    return -np.inf
        return val

    def log_probability(self, params) -> tuple:
        nan_blobs = [np.nan] * len(self.BLOB_NAMES)
        self.set_parameter_vector(params)
        lp = self.log_prior()
        if not np.isfinite(lp):
            return -np.inf, *nan_blobs
        chisq = 0.0
        try:
            for band in self.bands:
                chisq += self.get_value(band).chi2
        except BaseException as err:
            if isinstance(err, (KeyboardInterrupt, SystemExit)):
                raise
            return -np.inf, *nan_blobs
        return lp - 0.5 * chisq, *self.blobs

_MODEL: Lcurve_MCMC | None = None

def init_worker(config: dict) -> None:
    global _MODEL
    _MODEL = Lcurve_MCMC(logging.getLogger("lcurve_mcmc.worker"), config)

def log_probability(params):
    return _MODEL.log_probability(params)

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit eclipse photometry with lcurve")
    parser.add_argument("--conf", "-c", required=True)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--test", "-t", action="store_true")
    parser.add_argument("--fit", "-f", action="store_true")
    args = parser.parse_args()
    return args

def main() -> None:
    args = arg_parse()
    yaml = YAML(typ="safe")
    with open(args.conf) as f:
        config = yaml.load(f)
    run = config["run_settings"]
    logger = lcurve_logging("lcurve_mcmc", f"logs/{config['run_name']}.log").setup_logger()

    if args.setup:
        Lcurve_MCMC(logger, config, build_model_files=True)
        return

    names = list(config["params"].keys())
    p_start = np.array(list(config["params"].values()), dtype=float)
    ndim = len(p_start)

    if args.test:
        init_worker(config)
        lnp, *blobs = log_probability(p_start)
        logger.info(f"ln_prob = {lnp}")
        logger.info(str(dict(zip(Lcurve_MCMC.BLOB_NAMES, blobs))))
        return

    if not args.fit:
        parser.error("pass --setup, --test or --fit")

    nwalkers = run["walkers"]
    nburn = run["burnin"]
    nprod = run["production"]
    nthreads = run["n_cores"]

    run_name = config["run_name"]
    folder = os.path.join("MCMC_runs", run_name)
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(args.conf, os.path.join(folder, f"{run_name}.yaml"))

    backend = emcee.backends.HDFBackend(os.path.join(folder, f"{run_name}.h5"))
    if backend.iteration == 0:
        backend.reset(nwalkers, ndim)
        scatter = 1e-3 * np.abs(p_start)
        if "t0" in names:
            scatter[names.index("t0")] = 1e-6
        p0 = p_start + scatter * np.random.randn(nwalkers, ndim)
    else:
        logger.info(f"Resuming from step {backend.iteration}")
        p0, nburn = None, 0
        nprod -= backend.iteration

    dtype = [(n, float) for n in Lcurve_MCMC.BLOB_NAMES]

    with Pool(nthreads, initializer=init_worker, initargs=(config,)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,
                                        backend=backend, blobs_dtype=dtype)
        if nburn > 0:
            state = sampler.run_mcmc(p0, nburn, progress=True)
            sampler.reset()
            p0 = state
        if nprod > 0:
            sampler.run_mcmc(p0, nprod, progress=True)

    logger.info(f"Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    try:
        logger.info(f"Autocorrelation time: {sampler.get_autocorr_time()}")
    except emcee.autocorr.AutocorrError as err:
        logger.warning(f"Chain too short for a reliable autocorr time: {err}")

if __name__ == "__main__":
    main()