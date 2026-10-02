"""Quick phase-fold check: load raw data, fold, plot. No model needed."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

targets = [
    ("../targets/config_files/ASASSN18cv.yaml",  "ASASSN18cv"),
    ("../targets/config_files/ASASSN18jh.yaml",  "ASASSN18jh"),
    ("../targets/config_files/DR3_600.yaml",      "DR3_600"),
]

yaml = YAML(typ="safe")
fig, axes = plt.subplots(3, 3, figsize=(15, 9))

for row, (conf_path, label) in enumerate(targets):
    with open(conf_path) as f:
        cfg = yaml.load(f)
    period  = float(cfg["period"])
    t0_seed = float(cfg["params"]["t0"])
    run     = cfg["run_settings"]
    phase_lim = run.get("phase_lim", 0.05)

    bands = list(cfg["light_curves"].keys())
    for col, band in enumerate(bands):
        ax = axes[row][col]
        path = cfg["light_curves"][band]
        t, texp, flux, ferr, w, ndiv = np.loadtxt(path, unpack=True)

        phase = ((t - t0_seed) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        order = np.argsort(phase)
        ph, fl, fe = phase[order], flux[order], ferr[order]

        ax.errorbar(ph, fl, yerr=fe, fmt='.', ms=2, elinewidth=0.5,
                    color={'u': 'cornflowerblue', "u'": 'cornflowerblue',
                           'g': 'limegreen',      "g'": 'limegreen',
                           'r': 'orange',         "r'": 'orange'}.get(band.strip("'"), 'gray'),
                    alpha=0.6)
        ax.axvline(0, c='k', ls='--', lw=0.8)
        ax.set_xlim(-phase_lim, phase_lim)
        # show full range in small inset hint
        ax.set_title(f"{label} {band}", fontsize=8)
        if col == 0:
            ax.set_ylabel("Flux (mJy)", fontsize=7)
        if row == 2:
            ax.set_xlabel("Phase", fontsize=7)
        ax.tick_params(labelsize=7)

        # annotate with eclipse depth estimate
        in_ecl = np.abs(ph) < 0.02
        out_ecl = (np.abs(ph) > phase_lim * 0.5) & (np.abs(ph) < phase_lim)
        if in_ecl.sum() > 2 and out_ecl.sum() > 2:
            depth = np.median(fl[out_ecl]) - np.median(fl[in_ecl])
            noise = np.median(fe[out_ecl])
            ax.set_title(f"{label} {band}  depth={depth/noise:.1f}σ", fontsize=8)

fig.suptitle("Phase-folded raw data (no model)  |  dashed line = t0 seed", fontsize=10)
fig.tight_layout()
out = "check_fold.pdf"
fig.savefig(out, dpi=120)
print(f"Saved → {out}")

# Also print stats
print("\nPhase-fold stats:")
for conf_path, label in targets:
    with open(conf_path) as f:
        cfg = yaml.load(f)
    period  = float(cfg["period"])
    t0_seed = float(cfg["params"]["t0"])
    run     = cfg["run_settings"]
    phase_lim = run.get("phase_lim", 0.05)
    band = list(cfg["light_curves"].keys())[1]  # g-band
    path = cfg["light_curves"][band]
    t, texp, flux, ferr, w, ndiv = np.loadtxt(path, unpack=True)
    phase = ((t - t0_seed) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    in_ecl  = np.abs(phase) < 0.02
    out_ecl = (np.abs(phase) > phase_lim * 0.3) & (np.abs(phase) < phase_lim)
    n_in    = in_ecl.sum()
    n_out   = out_ecl.sum()
    if n_in > 0 and n_out > 0:
        med_in  = np.median(flux[in_ecl])
        med_out = np.median(flux[out_ecl])
        noise   = np.median(ferr[out_ecl])
        depth   = med_out - med_in
        print(f"  {label} ({band}): t0={t0_seed:.6f}  in-eclipse pts={n_in}  depth={depth:.4f}  noise={noise:.4f}  SNR={depth/noise:.1f}")
    else:
        print(f"  {label} ({band}): t0={t0_seed:.6f}  in-eclipse pts={n_in} (no eclipse coverage?)")
        print(f"    phase range: {phase.min():.3f} to {phase.max():.3f}")
        print(f"    t range: {t.min():.4f} to {t.max():.4f},  t0={t0_seed:.6f}")
