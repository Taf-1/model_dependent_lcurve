import argparse as ap
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
from cam_cal.observation import Observation
from lc_logger import lcurve_logging

def arg_parse(argv=None):
    p = ap.ArgumentParser(description=__doc__,
                          formatter_class=ap.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--science", required=True, help="Target directory/name")
    p.add_argument("--run", nargs="+", required=True, help="Science run names")
    p.add_argument("--atm_run", nargs="+", required=True,
                   help="One atmosphere group, or one per science run; commas join logs")
    p.add_argument("--std_run", nargs="+", required=True,
                   help="One standard run, or one per science run")
    p.add_argument("--std", nargs="+", required=True,
                   help="One standard name, or one per science run (e.g. GD50)")
    args = p.parse_args(argv)
    count = len(args.run)
    for key in ("atm_run", "std_run", "std"):
        entries = getattr(args, key)
        if len(entries) == 1:
            setattr(args, key, entries * count)
        elif len(entries) != count:
            p.error(f"--{key} requires one entry or {count} entries, matching --run")
    return args

def log_path(base, name):
    name = name.strip()
    if not name:
        raise ValueError("Run names cannot be empty")
    path = Path(name if name.endswith(".log") else name + ".log")
    path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def make_jobs(args):
    base = args.dir / args.science
    jobs = []
    outputs = set()
    for run, atm_group, std_run, std_name in zip(
        args.run, args.atm_run, args.std_run, args.std
    ):
        science = log_path(base, run)
        atmosphere = tuple(log_path(base, item) for item in atm_group.split(","))
        standard = log_path(base, std_run)
        output_name = f"{args.science}_{science.stem}"
        output = science.parent / "reduced" / output_name / f"{output_name}.fits"
        if output in outputs:
            raise ValueError(f"Duplicate science output: {output}")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        outputs.add(output)
        jobs.append((science, atmosphere, standard, std_name, output_name, output))
    return jobs


def main(argv=None):
    args = arg_parse(argv)
    jobs = make_jobs(args)

    class UltracamObservation(Observation):
        def __init__(self):
            with patch("cam_cal.observation.Tk"):
                super().__init__("ultracam", filters=["us", "gs", "rs"])

        def fit_extinction(self, logfiles):
            self.add_observation(
                name="extinction", logfiles=[str(p) for p in logfiles], obs_type="atm"
            )
            self.get_atm_ex(plot=False)
            return deepcopy(self.atm_extinction)

        def fit_zeropoints(self, logfile, std_name):
            self.add_observation(
                name=std_name, logfiles=[str(logfile)], obs_type="std",
                cal_mags=std_name,
            )
            self.get_zeropoint()
            if not hasattr(self, "standard") or not hasattr(self, "std_run"):
                raise RuntimeError("No standard-star calibration accepted; stopping.")
            return deepcopy(self.zeropoint)

        def science_calibration(self, logfile, output_name):
            self.add_observation(
                name=output_name, logfiles=[str(logfile)], obs_type="science"
            )
            self.calibrate_science(output_name, eclipse=None, lcurve=True, show=False)

    logger = lcurve_logging(
        "ultracam_extract_lc", "ultracam_extract_lc.log"
    ).setup_logger()
    atmosphere_cache = {}
    standard_cache = {}

    for science, atmosphere, standard, std_name, output_name, output in jobs:
        obs = UltracamObservation()
        logger.info(f"Science: {science}; standard: {std_name}, {standard}")
        logger.info(f"Atmospheric logs: {', '.join(map(str, atmosphere))}")
        if atmosphere not in atmosphere_cache:
            atmosphere_cache[atmosphere] = obs.fit_extinction(atmosphere)
        obs.atm_extinction = deepcopy(atmosphere_cache[atmosphere])

        key = (atmosphere, standard, std_name)
        if key not in standard_cache:
            zp = obs.fit_zeropoints(standard, std_name)
            standard_cache[key] = (zp, obs.standard, obs.std_run)
        zp, obs.standard, obs.std_run = standard_cache[key]
        obs.zeropoint = deepcopy(zp)

        # These are the values actually adopted, including defaults if 'n'
        # was selected at the atmospheric-fit prompt.
        for band in obs.filters:
            message = (
                f"{band}: zeropoint = {obs.zeropoint['mean'][band]:.3f} "
                f"+/- {obs.zeropoint['err'][band]:.3f} mag; "
                f"adopted extinction = {obs.atm_extinction['mean'][band]:.3f} "
                f"+/- {obs.atm_extinction['err'][band]:.3f} mag/airmass"
            )
            logger.info(message)

        obs.science_calibration(science, output_name)
        logger.info(f"Science calibration completed: {output}")

if __name__ == "__main__":
    main()
