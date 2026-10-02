"""
check_eclipse.py  –  Phase-bin residuals inside the eclipse window and report
peak-to-peak.  Criterion: ptp ≤ 0.5 σ for all bands.

Reads best-fit parameters from the run log file automatically.

Run from src/:
    python check_eclipse.py --target ASASSN18cv_run024
    python check_eclipse.py --target DR3_600_run009
    python check_eclipse.py   # all known targets
"""
import argparse, logging, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ruamel.yaml import YAML
import lcurve_mcmc as mc

logging.basicConfig(level=logging.WARNING)

# ── eclipse window per target (|phase| < this inspected) ─────────────────────
ECLIPSE_WIN = {
    "ASASSN18cv_run023": 0.07,
    "ASASSN18cv_run024": 0.07,   # outer contact ≈ ±0.057; window just outside
    "DR3_600_run008":    0.06,
    "DR3_600_run009":    0.06,
}

N_BINS      = 10
THRESHOLD   = 0.5   # σ  (binned peak-to-peak criterion)
INFLATE_ERRORS = True   # add OOT rms in quadrature

yaml = YAML(typ="safe")

# ── regex for parsing log lines like:  "  m1 = 0.213949" ─────────────────────
_PARAM_RE = re.compile(r"INFO \|\s+(\w+)\s+=\s+([0-9eE+\-.]+)")


def read_best_fit_from_log(run_name: str):
    """Parse the most recent 'Best-fit parameters' block from the run log."""
    log_path = f"logs/{run_name}.log"
    if not os.path.exists(log_path):
        return None
    with open(log_path) as fh:
        lines = fh.readlines()

    # Find the last occurrence of the "Best-fit parameters:" header
    last_start = None
    for i, line in enumerate(lines):
        if "Best-fit parameters:" in line:
            last_start = i
    if last_start is None:
        return None

    params = {}
    for line in lines[last_start + 1:]:
        m = _PARAM_RE.search(line)
        if m:
            params[m.group(1)] = float(m.group(2))
        elif "INFO" in line and "|" in line:
            break   # non-param INFO line → end of block
    return params if params else None


def discover_targets() -> list[str]:
    """Return run names that have both a YAML config and a log with best-fit params."""
    targets = []
    for entry in sorted(os.listdir("MCMC_runs")):
        conf = f"MCMC_runs/{entry}/{entry}.yaml"
        log  = f"logs/{entry}.log"
        if os.path.isfile(conf) and os.path.isfile(log):
            targets.append(entry)
    return targets


def run_check(run_name: str, verbose: bool = True) -> bool:
    conf_path = f"MCMC_runs/{run_name}/{run_name}.yaml"
    if not os.path.exists(conf_path):
        print(f"  Config not found: {conf_path}")
        return False

    best = read_best_fit_from_log(run_name)
    if best is None:
        print(f"  No best-fit params found in logs/{run_name}.log")
        return False

    with open(conf_path) as fh:
        config = yaml.load(fh)

    phase_win = ECLIPSE_WIN.get(run_name, 0.08)

    print(f"\n{'='*68}")
    print(f"  {run_name}")
    print(f"{'='*68}")

    mc.init_worker(config)
    model = mc._MODEL

    seeds = config.get("params", {})
    param_vec = np.array([best.get(n, float(seeds[n])) for n in model.parameter_names])
    model.set_parameter_vector(param_vec)
    model.geometry_dirty = True

    period = model.period
    t0     = model.t0

    # report derived geometry
    model.geometry()
    print(f"  r1={model.r1:.5f} Rsun  r2={model.r2:.5f} Rsun  a={model.a:.4f} Rsun")
    r1a, r2a = model.r1 / model.a, model.r2 / model.a
    contact_half = np.degrees(np.arcsin(np.clip(r1a + r2a, 0, 1))) / 360.0
    print(f"  contact half-width ≈ {contact_half:.4f} phase  ({contact_half*period*1440:.2f} min)")

    model.geometry_dirty = True

    all_ok = True
    for band in model.bands:
        lc = model.get_value(band)
        d  = model.data[band]

        phase = ((d["time"] - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0

        ferr = d["f_err"].copy()
        if INFLATE_ERRORS:
            oot_mask = np.abs(phase) > 0.2
            if oot_mask.sum() > 10:
                oot_rms = d["flux"][oot_mask].std()
                ferr    = np.sqrt(ferr**2 + oot_rms**2)

        mask  = np.abs(phase) < phase_win
        ph_e  = phase[mask]
        res_e = (d["flux"][mask] - lc.total[mask]) / ferr[mask]

        edges = np.linspace(-phase_win, phase_win, N_BINS + 1)
        idx   = np.digitize(ph_e, edges) - 1
        means = np.array([
            res_e[idx == i].mean() if np.any(idx == i) else np.nan
            for i in range(N_BINS)
        ])
        finite = means[np.isfinite(means)]
        ptp    = float(np.ptp(finite)) if finite.size > 1 else np.nan
        ok     = ptp <= THRESHOLD

        chi2_ecl = float((res_e**2).sum())
        status   = "PASS ✓" if ok else f"FAIL ✗  (> {THRESHOLD:.1f} σ)"
        print(f"  {band:4s}  n={mask.sum():4d}  chi2_ecl={chi2_ecl:8.1f}"
              f"  ptp={ptp:.3f}σ  {status}")

        if verbose and not ok:
            ctrs = 0.5 * (edges[:-1] + edges[1:])
            for ctr, m in zip(ctrs, means):
                if np.isfinite(m):
                    bar  = "#" * int(abs(m) * 4 + 0.5)
                    sign = "+" if m >= 0 else "-"
                    print(f"      phase {ctr:+.3f}  {sign}{abs(m):.3f}σ  {bar}")

        all_ok = all_ok and ok

    status_line = "ALL BANDS PASS ✓" if all_ok else "NEEDS WORK ✗"
    print(f"\n  ──► {status_line}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", dest="targets",
                    help="which run(s) to check (default: all with logs)")
    args = ap.parse_args()
    targets = args.targets or discover_targets()

    results = {}
    for t in targets:
        results[t] = run_check(t)

    print("\n" + "="*68)
    print("  Summary")
    print("="*68)
    for t, ok in results.items():
        print(f"  {t:30s}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
