from __future__ import annotations
import argparse
import logging
import os
import shutil
from multiprocessing import get_context
from pathlib import Path
import emcee
import numpy as np
from ruamel.yaml import YAML
import lcurve
import utils as ph
from m_r_tracks import get_radius
from lcurve_model import Rust_LCURVE
from lc_logger import lcurve_logging

os.nice(19) # lower priority for CPU-intensive tasks

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
        self.primary_model = config.get("primary_model", "WD")
        self.secondary_model = config.get("secondary_model", "MS")
        self.secondary_wd_core_comp = config.get("secondary_wd_core_comp", config["wd_core_comp"])
        self.secondary_wd_model = config.get("secondary_wd_model", config.get("wd_model", "Koester"))
        # sdB primaries use WD atmosphere grids (hot subdwarfs have WD-like spectra); r1 is a free param
        # For donors, donor_atm controls which atmosphere tables to use (MS for CV donors, WD for AM CVn)
        # sdB secondaries use WD atmosphere grids; r2 is a free param
        if self.secondary_model == "donor":
            self._sec_atm = config.get("donor_atm", "MS")
        elif self.secondary_model in ("WD", "sdB"):
            self._sec_atm = "WD"
        else:
            self._sec_atm = "MS"
        self.seeds = self.sed_seeds(config)
        if build_model_files:
            self.setup_model_files()
        self.beam_factors = {b: self.read_beam_factors(b) for b in self.bands}
        self.data = {b: self.load_data_file(p) for b, p in config["light_curves"].items()}
        self.binary_model = lcurve.BinaryModel.from_file(self.mod_path(self.bands[0]))
        self.set_core_parameters()
        self.geometry_dirty = True
        self.blobs = [np.nan] * len(self.BLOB_NAMES)

    @staticmethod
    def sed_seeds(config: dict) -> dict:
        sed = config["sed"]
        secondary_model = config.get("secondary_model", "MS")
        m1 = sed["m1"] if "m1" in sed else ph.mass_from_radius(sed["r1"], sed["t1"], config["wd_core_comp"])
        if "m2" in sed:
            m2 = sed["m2"]
        elif secondary_model == "WD":
            sec_core = config.get("secondary_wd_core_comp", config["wd_core_comp"])
            m2 = ph.mass_from_radius(sed["r2"], sed["t2"], sec_core)
        elif secondary_model == "donor":
            raise ValueError("secondary_model=donor requires 'm2' to be specified in the sed block "
                             "(radius is set by Roche lobe geometry, not a mass-radius relation)")
        else:
            m2 = ph.ms_mass_from_radius(sed["r2"], config["secondary_mr"])
        # For Roche-lobe-filling donors, r2 at the seed point is set by orbital geometry
        if secondary_model == "donor":
            a_seed = ph.separation(m1, m2, float(config["period"]))
            r2_seed = ph.roche_lobe_va(m2 / m1) * a_seed
        else:
            r2_seed = sed["r2"]
        return {"m1": m1, "m2": m2, "t1": sed["t1"], "t2": sed["t2"],
                "r1": sed["r1"], "r2": r2_seed,
                "logg1": ph.log_g(m1, sed["r1"]), "logg2": ph.log_g(m2, r2_seed)}

    def mod_path(self, band: str) -> str:
        tag = band.replace("'", "")
        return f"model_files/{self.target_name}_{tag}.mod"

    def setup_model_files(self) -> None:
        os.makedirs("model_files", exist_ok=True)
        loggs = [self.seeds["logg1"], self.seeds["logg2"]]
        opts = self.config.get("components", {})
        for band in self.bands:
            rl = Rust_LCURVE(self.logger, self.template_mod, self.config["light_curves"][band],
                             loggs, band, self.target_name,
                             disc=opts.get("disc", False), spot=opts.get("spot", False),
                             secondary_eclipse=opts.get("secondary_eclipse", False),
                             secondary_model=self._sec_atm)
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
        self.a = ph.separation(self.m1, self.m2, self.period)
        if self.primary_model == "sdB":
            pass  # self.r1 is a free parameter; set_parameter_vector already applied it
        else:
            self.r1 = get_radius(self.m1, self.t1, star_type=self.config["wd_core_comp"])
        if self.secondary_model == "WD":
            self.r2 = get_radius(self.m2, self.t2, star_type=self.secondary_wd_core_comp)
        elif self.secondary_model == "donor":
            self.r2 = ph.roche_lobe_va(q) * self.a * (1.0 - 1e-6)
        elif self.secondary_model == "sdB":
            pass  # self.r2 is a free parameter; set_parameter_vector already applied it
        else:
            self.r2 = get_radius(self.m2, star_type="MS", relation=self.config["secondary_mr"])
        self.logg1 = ph.log_g(self.m1, self.r1)
        self.logg2 = ph.log_g(self.m2, self.r2)
        r1_a = self.r1 / self.a
        r2_vol_a = self.r2 / self.a          # volume radius (sphere equivalent)
        # lcurve wants the L1-facing radius for the secondary Roche lobe mesh
        r2_a = ph.rva_to_rl1(q, r2_vol_a)
        self.ffac = ph.fill_factor(q, r2_a)
        # phase1 must use volume radius so the eclipse window matches the actual
        # data eclipse contacts; L1-facing radius would extend the window past OOT
        phase1 = float(np.arcsin(r1_a + r2_vol_a) / (2 * np.pi) + 0.001)
        # K1+K2 = 2π a sin(i) / P; keeps absolute flux scale consistent with physical masses
        velocity_scale = (2 * np.pi * self.a * 6.957e10 * np.sin(np.radians(self.incl))
                          / (self.period * 86400.0)) / 1e5  # km/s
        return {"q": q, "iangle": self.incl, "r1": r1_a, "r2": r2_a,
                "t0": self.t0, "phase1": phase1, "phase2": 0.5 - phase1,
                "velocity_scale": velocity_scale}

    def continuum(self, band: str) -> dict:
        a1, a2, a3, a4 = ph.get_ldcs(self.t1, self.logg1, band, "WD", self.instrument)
        if self._sec_atm == "WD":
            b1, b2, b3, b4 = ph.get_ldcs(self.t2, self.logg2, band, "WD", self.instrument)
            t2_bb = ph.get_tbb(self.t2, self.logg2, band, "WD", self.secondary_wd_model, self.instrument)
            gdc2 = ph.get_gdc(self.t2, self.logg2, band, "WD")
        else:
            b1, b2, b3, b4 = ph.get_ldcs(self.t2, self.logg2, band, "MS", self.instrument)
            t2_bb = ph.get_tbb(self.t2, self.logg2, band, "MS", self.config["ms_model"], self.instrument)
            gdc2 = ph.get_gdc(self.t2, self.logg2, band, "MS")
        pars = {
            "t1": ph.get_tbb(self.t1, self.logg1, band, "WD", self.config["wd_model"], self.instrument),
            "t2": t2_bb,
            "wavelength": ph.pivot_wavelength(band, self.instrument),
            "gravity_dark1": ph.get_gdc(self.t1, self.logg1, band, "WD"),
            "gravity_dark2": gdc2,
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
        if 'disc_temp' in self.parameter_names:
            pars['temp_disc'] = float(self.disc_temp)
        if 'rdisc2' in self.parameter_names:
            pars['rdisc2'] = float(self.rdisc2)
        if 'angle_spot' in self.parameter_names:
            pars['angle_spot'] = float(self.angle_spot)
        self.binary_model.update(pars)
        d = self.data[band]
        lc = self.binary_model.compute_light_curve(d["time"], d["t_exp"], d["n_div"],
                                                   d["flux"], d["f_err"], d["weight"])
        self.blobs = [self.logg1, self.logg2, self.r1, self.r2, self.a, self.ffac, lc.rva2]
        return lc

    @property
    def lightcurves(self) -> dict:
        return self.data

    def model(self, band: str, params) -> tuple:
        self.set_parameter_vector(params)
        lc = self.get_value(band)
        d = self.data[band]
        return d["time"], lc.total, d["flux"], d["f_err"]

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
            # invalid geometry comes back as a Rust PanicException, not an Exception
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


def _chi2_total(params) -> float:
    """Total chi-squared across all bands for scipy minimization."""
    _MODEL.set_parameter_vector(params)
    if not _MODEL.in_bounds():
        return np.inf
    chisq = 0.0
    try:
        for band in _MODEL.bands:
            chisq += _MODEL.get_value(band).chi2
    except BaseException as err:
        if isinstance(err, (KeyboardInterrupt, SystemExit)):
            raise
        return np.inf
    return chisq


def _residuals_flat(params) -> np.ndarray:
    """Concatenated sqrt(weight)*(O-C)/sigma residuals for Levenberg-Marquardt."""
    _MODEL.set_parameter_vector(params)
    n_total = sum(d["flux"].size for d in _MODEL.data.values())
    if not _MODEL.in_bounds():
        return np.full(n_total, 1e6)
    parts = []
    try:
        for band in _MODEL.bands:
            lc = _MODEL.get_value(band)
            d = _MODEL.data[band]
            parts.append(np.sqrt(d["weight"]) * (d["flux"] - lc.total) / d["f_err"])
    except BaseException as err:
        if isinstance(err, (KeyboardInterrupt, SystemExit)):
            raise
        return np.full(n_total, 1e6)
    return np.concatenate(parts)


def _binned_oc_ptp(model: Lcurve_MCMC, params, n_bins: int = 20) -> dict:
    """Peak-to-peak amplitude of phase-binned O-C residuals (in units of sigma) per band."""
    model.set_parameter_vector(params)
    result = {}
    for band in model.bands:
        lc = model.get_value(band)
        d = model.data[band]
        resid = (d["flux"] - lc.total) / d["f_err"]
        phase = ((d["time"] - model.t0) / model.period) % 1.0
        order = np.argsort(phase)
        ph_s, res_s = phase[order], resid[order]
        idx = np.digitize(ph_s, np.linspace(0, 1, n_bins + 1)) - 1
        means = np.array([res_s[idx == i].mean() if np.any(idx == i) else np.nan
                          for i in range(n_bins)])
        finite = means[np.isfinite(means)]
        result[band] = float(np.ptp(finite)) if finite.size > 1 else np.nan
    return result


def main() -> None:

    parser = argparse.ArgumentParser(description="Fit eclipse photometry with lcurve")
    parser.add_argument("--conf", "-c", required=True)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--test", "-t", action="store_true")
    parser.add_argument("--minimize", "-m", action="store_true",
                        help="run scipy optimizer only (no MCMC); same as --fit runs it automatically")
    parser.add_argument("--no-minimize", action="store_true",
                        help="skip pre-MCMC minimization and start MCMC directly from seed parameters")
    parser.add_argument("--method", choices=["nelder-mead", "lm"], default="nelder-mead",
                        help="optimizer method: nelder-mead (simplex) or lm (Levenberg-Marquardt)")
    parser.add_argument("--fit", "-f", action="store_true")
    parser.add_argument("--plot", "-p", action="store_true")
    parser.add_argument("--nwalkers", type=int)
    parser.add_argument("--nburn", type=int)
    parser.add_argument("--nprod", type=int)
    parser.add_argument("--nthreads", type=int)
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.conf) as f:
        config = yaml.load(f)
    run = config["run_settings"]
    logger = lcurve_logging("lcurve_mcmc", f"logs/{config['run_name']}.log").setup_logger()

    if args.setup:
        Lcurve_MCMC(logger, config, build_model_files=True)
        return

    names = list(config["params"].keys())
    seeds = Lcurve_MCMC.sed_seeds(config)
    p_start = np.array([config["params"][n] if config["params"][n] is not None else seeds[n]
                        for n in names], dtype=float)
    ndim = len(p_start)
    logger.info("seeds: " + ", ".join(f"{n}={v:.5g}" for n, v in zip(names, p_start)))

    if args.test:
        init_worker(config)
        lnp, *blobs = log_probability(p_start)
        logger.info(f"ln_prob = {lnp}")
        logger.info(str(dict(zip(Lcurve_MCMC.BLOB_NAMES, blobs))))
        return

    run_minimize = args.minimize or (args.fit and not args.no_minimize)
    if run_minimize:
        from scipy.optimize import minimize as sp_minimize, least_squares
        init_worker(config)
        logger.info(f"Running {args.method} minimization from seed parameters")
        # Optionally mask OOT data so minimizer focuses on the eclipse region.
        min_phase_lim = run.get("minimize_phase_lim", None)
        saved_weights = {}
        if min_phase_lim is not None:
            t0_seed = float(config["params"]["t0"])
            logger.info(f"  Masking |phase| > {min_phase_lim} for minimization (eclipse-only)")
            for band in _MODEL.bands:
                d = _MODEL.data[band]
                ph = ((d["time"] - t0_seed) / _MODEL.period) % 1.0
                ph[ph > 0.5] -= 1.0
                saved_weights[band] = d["weight"].copy()
                d["weight"] = d["weight"] * (np.abs(ph) <= min_phase_lim).astype(float)
        if args.method == "lm":
            lb = np.array([config["param_bounds"][n][0] for n in names])
            ub = np.array([config["param_bounds"][n][1] for n in names])
            result = least_squares(_residuals_flat, p_start, method="trf",
                                   bounds=(lb, ub),
                                   ftol=1e-4, xtol=1e-4, gtol=0.0, max_nfev=20000,
                                   diff_step=1e-3, x_scale='jac')
            best_params = result.x
            best_chisq = 2.0 * result.cost
            success = result.success
        else:
            result = sp_minimize(_chi2_total, p_start, method="Nelder-Mead",
                                 options={"xatol": 1e-5, "fatol": 0.01,
                                          "maxiter": 20000, "adaptive": True})
            best_params = result.x
            best_chisq = result.fun
            success = result.success
        for band, w in saved_weights.items():
            _MODEL.data[band]["weight"] = w
        logger.info(f"Minimization {'converged' if success else 'hit iteration limit'}: "
                    f"chi2={best_chisq:.3f}  (nfev={result.nfev})")
        logger.info("Best-fit parameters:")
        for n, v in zip(names, best_params):
            logger.info(f"  {n} = {v:.10g}")
        oc = _binned_oc_ptp(_MODEL, best_params)
        logger.info("Phase-binned O-C peak-to-peak (sigma) — flat means <~1:")
        for band, ptp in oc.items():
            logger.info(f"  {band}: {ptp:.3f}")
        p_start = best_params
        if args.minimize and not args.fit:
            import plotting
            run_name = config["run_name"]
            folder = os.path.join("MCMC_runs", run_name)
            os.makedirs(folder, exist_ok=True)
            phase_lim = run.get("phase_lim", 0.05)
            lc_path = os.path.join(folder, f"{run_name}_minimize_LC.pdf")
            plotting.plot_LC(_MODEL, best_params, show=False, save=True, name=lc_path, phase_lim=phase_lim)
            logger.info(f"Saved minimizer LC plot → {lc_path}")
            return

    if args.plot:
        import plotting
        run_name = config["run_name"]
        folder = os.path.join("MCMC_runs", run_name)
        run_yaml = os.path.join(folder, f"{run_name}.yaml")
        if os.path.exists(run_yaml):
            with open(run_yaml) as _f:
                config = yaml.load(_f)
            run = config["run_settings"]
            names = list(config["params"].keys())
        backend = emcee.backends.HDFBackend(os.path.join(folder, f"{run_name}.h5"), read_only=True)
        chain = backend.get_chain()           # (nsteps, nwalkers, ndim)
        flat_chain = backend.get_chain(flat=True)  # (nsteps*nwalkers, ndim)
        lnprob = backend.get_log_prob()            # (nsteps, nwalkers)
        lnprob_flat = backend.get_log_prob(flat=True)  # (nsteps*nwalkers,)

        # Sigma-clip outlier walkers from flat chain and traces using MAD.
        # Compute statistics on finite samples only so -inf values don't bias the threshold.
        finite_mask = np.isfinite(lnprob_flat)
        finite_lnp = lnprob_flat[finite_mask]
        median_lnp = np.median(finite_lnp)
        mad_lnp = np.median(np.abs(finite_lnp - median_lnp))
        clip_thresh = median_lnp - 5.0 * mad_lnp * 1.4826
        good_flat = finite_mask & (lnprob_flat > clip_thresh)
        flat_chain_clipped = flat_chain[good_flat]
        logger.info(f"Sigma-clip: kept {good_flat.sum()}/{len(good_flat)} samples "
                    f"({100*good_flat.mean():.1f}%, threshold lnp>{clip_thresh:.1f})")

        # For traces: remove walkers whose median finite lnprob is below threshold.
        # nanmedian handles -inf values gracefully.
        lnprob_finite = np.where(np.isfinite(lnprob), lnprob, np.nan)
        walker_med_lnp = np.nanmedian(lnprob_finite, axis=0)   # (nwalkers,)
        good_walkers = walker_med_lnp > clip_thresh
        chain_clipped = chain[:, good_walkers, :]
        logger.info(f"Sigma-clip traces: kept {good_walkers.sum()}/{len(good_walkers)} walkers")

        trace_path = os.path.join(folder, f"{run_name}_traces.pdf")
        plotting.plot_traces(chain_clipped, names, name=trace_path)
        logger.info(f"Saved trace plot → {trace_path}")

        corner_path = os.path.join(folder, f"{run_name}_corner.pdf")
        plotting.plot_CP(flat_chain_clipped, names,
                         composition=config.get("wd_core_comp", "CO"),
                         name=corner_path)
        logger.info(f"Saved corner plot → {corner_path}")

        median_params = np.median(flat_chain_clipped, axis=0)
        init_worker(config)
        lc_path = os.path.join(folder, f"{run_name}_LC.pdf")
        phase_lim = run.get("phase_lim", 0.05)
        plotting.plot_LC(_MODEL, median_params, show=False, save=True, name=lc_path, phase_lim=phase_lim)
        logger.info(f"Saved light curve plot → {lc_path}")
        return

    if not args.fit:
        parser.error("pass --setup, --test, --minimize, --fit or --plot")

    nwalkers = args.nwalkers or run["walkers"]
    nburn = args.nburn if args.nburn is not None else run["burnin"]
    nprod = args.nprod or run["production"]
    nthreads = args.nthreads or run["n_cores"]

    run_name = config["run_name"]
    folder = os.path.join("MCMC_runs", run_name)
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(args.conf, os.path.join(folder, f"{run_name}.yaml"))

    backend = emcee.backends.HDFBackend(os.path.join(folder, f"{run_name}.h5"))
    try:
        current_iter = backend.iteration
    except (FileNotFoundError, OSError):
        current_iter = 0

    if current_iter == 0:
        backend.reset(nwalkers, ndim)
        scatter_conf = run.get("init_scatter", {})
        if isinstance(scatter_conf, (int, float)):
            default_frac = float(scatter_conf)
            param_overrides = {}
        else:
            default_frac = float(scatter_conf.get("default", 0.03))
            param_overrides = {k: float(v) for k, v in scatter_conf.items() if k != "default"}
        scatter = default_frac * np.abs(p_start)
        for pname, val in param_overrides.items():
            if pname in names:
                scatter[names.index(pname)] = val
        p0 = p_start + scatter * np.random.randn(nwalkers, ndim)
    else:
        logger.info(f"Resuming from step {current_iter}")
        p0, nburn = None, 0
        nprod -= current_iter

    dtype = [(n, float) for n in Lcurve_MCMC.BLOB_NAMES]

    with get_context("spawn").Pool(nthreads, initializer=init_worker, initargs=(config,)) as pool:
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