"""
check_eclipse_plot.py  —  Eclipse diagnostic plot for ASASSN18cv_run023.

Phase-folds eclipse data, overplots best-fit model, measures and compares:
  • Outer / inner contact half-widths  (ingress T1/T2, egress T3/T4)
  • Eclipse depth
  • Width at half-depth
for both the observed data and the model, and prints the differences.

Run from src/:
    python check_eclipse_plot.py
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ruamel.yaml import YAML
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import lcurve_mcmc as mc

# ── config ────────────────────────────────────────────────────────────────────
RUN_NAME   = "ASASSN18cv_run023"
CONF_PATH  = f"MCMC_runs/{RUN_NAME}/{RUN_NAME}.yaml"
LOG_PATH   = f"logs/{RUN_NAME}.log"
OUT_PDF    = f"MCMC_runs/{RUN_NAME}/{RUN_NAME}_eclipse_check.pdf"
PHASE_WIN  = 0.10          # half-width of plot window in phase
N_BINS     = 40            # bins for folded-data profile
DEPTH_FRAC = 0.10          # flux must drop by this fraction of depth to count as contact

BAND_COLOURS = {"u'": "cornflowerblue", "g'": "limegreen", "r'": "orange"}

yaml = YAML(typ="safe")
_PARAM_RE = re.compile(r"INFO \|\s+(\w+)\s+=\s+([0-9eE+\-.]+)")


# ── helpers ───────────────────────────────────────────────────────────────────

def read_best_fit(log_path):
    with open(log_path) as fh:
        lines = fh.readlines()
    last = None
    for i, line in enumerate(lines):
        if "Best-fit parameters:" in line:
            last = i
    if last is None:
        return {}
    params = {}
    for line in lines[last + 1:]:
        m = _PARAM_RE.search(line)
        if m:
            params[m.group(1)] = float(m.group(2))
        elif "INFO" in line and "|" in line:
            break
    return params


def model_contact_phases(r1, r2, a, incl_deg):
    """
    Exact contact half-widths from binary geometry.
    Returns (phi_out, phi_in) where phi_in is NaN if eclipse is grazing.

    Orbit geometry: projected separation d at phase φ satisfies
        d²(φ) = a²·sin²(2πφ) + b²·cos²(2πφ)
    where b = a·cos(i) is the impact parameter. Solving for φ at contact:
        sin²(2πφ) = [(r₁±r₂)² − b²] / (a² − b²)
    """
    incl  = np.radians(incl_deg)
    b     = a * np.cos(incl)          # impact parameter (Rsun)
    a2mb2 = a**2 - b**2               # denominator; > 0 whenever eclipse can occur

    if a2mb2 <= 0:
        return np.nan, np.nan

    # outer contact: d = r1 + r2
    num_out = (r1 + r2)**2 - b**2
    if num_out < 0:
        return np.nan, np.nan        # no eclipse
    phi_out = np.arcsin(np.sqrt(num_out / a2mb2)) / (2 * np.pi)

    # inner contact (total eclipse only): d = |r2 - r1|
    num_in = (r2 - r1)**2 - b**2
    phi_in = np.arcsin(np.sqrt(num_in / a2mb2)) / (2 * np.pi) if num_in > 0 else np.nan

    return phi_out, phi_in


def bin_profile(phase, flux, ferr, n_bins, phase_win):
    """Return bin centres, binned flux median, and binned flux error (stderr)."""
    edges = np.linspace(-phase_win, phase_win, n_bins + 1)
    ctrs  = 0.5 * (edges[:-1] + edges[1:])
    bflux = np.full(n_bins, np.nan)
    berr  = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase >= edges[i]) & (phase < edges[i + 1])
        if mask.sum() > 0:
            bflux[i] = np.median(flux[mask])
            berr[i]  = np.sqrt(np.sum(ferr[mask]**2)) / mask.sum()
    return ctrs, bflux, berr


def measure_data_eclipse(ctrs, bflux, oot_flux, depth):
    """
    Estimate outer and inner contact half-widths from binned data profile.
    outer contact: phase where flux drops below oot - DEPTH_FRAC * depth
    inner contact: phase where flux reaches oot - (1 - DEPTH_FRAC) * depth
    Returns (phi_out_ing, phi_in_ing, phi_in_egr, phi_out_egr, measured_depth)
    Half-widths are (phi_out_ing + phi_out_egr)/2 etc.
    """
    thresh_out = oot_flux - DEPTH_FRAC * depth
    thresh_in  = oot_flux - (1.0 - DEPTH_FRAC) * depth

    valid = np.isfinite(bflux)
    ph_v  = ctrs[valid]
    fl_v  = bflux[valid]

    # ingress side (negative phase, going right toward 0)
    ing = ph_v < 0
    if ing.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan, depth

    ph_ing = ph_v[ing]
    fl_ing = fl_v[ing]

    # outer ingress: rightmost phase where flux is still near OOT (moving inward)
    below_out = ph_ing[fl_ing < thresh_out]
    phi_out_ing = float(np.abs(below_out[0])) if len(below_out) > 0 else np.nan

    # inner ingress: rightmost phase where flux has reached minimum level
    below_in = ph_ing[fl_ing < thresh_in]
    phi_in_ing = float(np.abs(below_in[-1])) if len(below_in) > 0 else np.nan

    # egress side (positive phase)
    egr = ph_v > 0
    if egr.sum() < 2:
        return phi_out_ing, phi_in_ing, np.nan, np.nan, depth

    ph_egr = ph_v[egr]
    fl_egr = fl_v[egr]

    below_out_e = ph_egr[fl_egr < thresh_out]
    phi_out_egr = float(below_out_e[-1]) if len(below_out_e) > 0 else np.nan

    below_in_e = ph_egr[fl_egr < thresh_in]
    phi_in_egr = float(below_in_e[0]) if len(below_in_e) > 0 else np.nan

    return phi_out_ing, phi_in_ing, phi_in_egr, phi_out_egr, depth


# ── main ─────────────────────────────────────────────────────────────────────

with open(CONF_PATH) as fh:
    config = yaml.load(fh)

best = read_best_fit(LOG_PATH)
seeds = config.get("params", {})

mc.init_worker(config)
model = mc._MODEL
param_vec = np.array([best.get(n, float(seeds[n])) for n in model.parameter_names])
model.set_parameter_vector(param_vec)
model.geometry_dirty = True
model.geometry()

period = model.period
t0     = model.t0
r1, r2, a = model.r1, model.r2, model.a
incl      = model.incl

phi_out_model, phi_in_model = model_contact_phases(r1, r2, a, incl)
b = a * np.cos(np.radians(incl))
print(f"\n{'='*64}")
print(f"  {RUN_NAME}  |  best-fit geometry")
print(f"{'='*64}")
print(f"  r1={r1:.5f} Rsun  r2={r2:.5f} Rsun  a={a:.4f} Rsun")
print(f"  incl={incl:.3f}°   impact param b={b:.4f} Rsun")
print(f"  Eclipse type: {'TOTAL' if phi_in_model is not np.nan and not np.isnan(phi_in_model) else 'GRAZING'}")
print(f"  Model outer contact half-width φ_out = {phi_out_model:.4f} phase"
      f"  ({phi_out_model*period*1440:.2f} min)")
if not np.isnan(phi_in_model):
    print(f"  Model inner contact half-width φ_in  = {phi_in_model:.4f} phase"
          f"  ({phi_in_model*period*1440:.2f} min)")
else:
    print(f"  Model inner contact: N/A (grazing eclipse)")

bands = model.bands
fig = plt.figure(figsize=(14, 4 * len(bands)))
gs_outer = gridspec.GridSpec(len(bands), 1, hspace=0.45, figure=fig)

summary_rows = []

for bi, band in enumerate(bands):
    model.geometry_dirty = True
    lc = model.get_value(band)
    d  = model.data[band]

    phase = ((d["time"] - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    # OOT stats
    oot_mask = np.abs(phase) > 0.15
    oot_flux  = np.median(d["flux"][oot_mask]) if oot_mask.sum() > 5 else 1.0
    oot_model = np.median(lc.total[oot_mask]) if oot_mask.sum() > 5 else 1.0

    # normalise for display
    data_norm  = d["flux"]  / oot_flux
    model_norm = lc.total   / oot_flux
    err_norm   = d["f_err"] / oot_flux

    # eclipse region
    ecl_mask = np.abs(phase) < PHASE_WIN
    ph_e   = phase[ecl_mask]
    fl_e   = data_norm[ecl_mask]
    fe_e   = err_norm[ecl_mask]
    mod_e  = model_norm[ecl_mask]

    # bin data
    ctrs, bflux, berr = bin_profile(ph_e, fl_e, fe_e, N_BINS, PHASE_WIN)
    _, bmod,  _      = bin_profile(ph_e, mod_e, fe_e, N_BINS, PHASE_WIN)

    # data eclipse depth (from binned profile)
    eclipse_mask = np.abs(ctrs) < 0.03
    if np.any(np.isfinite(bflux[eclipse_mask])):
        data_depth  = 1.0 - np.nanmin(bflux[eclipse_mask])
        model_depth = 1.0 - np.nanmin(bmod[eclipse_mask])
    else:
        data_depth  = np.nan
        model_depth = np.nan

    # data contact phases
    phi_oi, phi_ii, phi_ie, phi_oe, _ = measure_data_eclipse(
        ctrs, bflux, 1.0, data_depth)

    phi_out_data = 0.5 * (phi_oi + phi_oe) if (phi_oi and phi_oe) else np.nan
    phi_in_data  = 0.5 * (phi_ii + phi_ie) if (phi_ii and phi_ie
                                                 and not np.isnan(phi_ii)
                                                 and not np.isnan(phi_ie)) else np.nan

    # ── subplot layout: LC on top, residuals below ──────────────────────────
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_outer[bi], height_ratios=[3, 1], hspace=0.05)
    ax_lc  = fig.add_subplot(gs_inner[0])
    ax_res = fig.add_subplot(gs_inner[1], sharex=ax_lc)

    col = BAND_COLOURS.get(band, "gray")

    # raw data (small points)
    ax_lc.errorbar(ph_e, fl_e, yerr=fe_e, fmt='.', ms=1.5, elinewidth=0.4,
                   color=col, alpha=0.35, zorder=1)
    # binned data
    ax_lc.errorbar(ctrs[np.isfinite(bflux)], bflux[np.isfinite(bflux)],
                   yerr=berr[np.isfinite(berr)],
                   fmt='o', ms=4, elinewidth=1.2, color=col,
                   markeredgecolor='k', markeredgewidth=0.4,
                   label="Data (binned)", zorder=3)
    # model
    order_m = np.argsort(ph_e)
    ax_lc.plot(ph_e[order_m], mod_e[order_m], 'k-', lw=1.4, label="Model", zorder=4)

    # model contact lines
    if not np.isnan(phi_out_model):
        for sign in [-1, 1]:
            ax_lc.axvline(sign * phi_out_model, color='k', ls='--', lw=0.8,
                          alpha=0.7, label="Model T1/T4" if sign == -1 else None)
    if not np.isnan(phi_in_model):
        for sign in [-1, 1]:
            ax_lc.axvline(sign * phi_in_model, color='k', ls=':', lw=0.8,
                          alpha=0.7, label="Model T2/T3" if sign == -1 else None)

    # data contact lines
    if not np.isnan(phi_out_data):
        for sign in [-1, 1]:
            ax_lc.axvline(sign * phi_out_data, color=col, ls='--', lw=1.0,
                          alpha=0.85, label="Data T1/T4" if sign == -1 else None)
    if not np.isnan(phi_in_data):
        for sign in [-1, 1]:
            ax_lc.axvline(sign * phi_in_data, color=col, ls=':', lw=1.0,
                          alpha=0.85, label="Data T2/T3" if sign == -1 else None)

    ax_lc.set_ylabel("Norm. flux", fontsize=9)
    ax_lc.set_xlim(-PHASE_WIN, PHASE_WIN)
    ax_lc.legend(fontsize=6.5, loc="lower center", ncol=4)
    ax_lc.set_title(f"{RUN_NAME}  {band}", fontsize=10)
    plt.setp(ax_lc.get_xticklabels(), visible=False)

    # residuals
    res = fl_e - mod_e
    ax_res.errorbar(ph_e, res, yerr=fe_e, fmt='.', ms=1.5,
                    elinewidth=0.4, color=col, alpha=0.4)
    bres = bflux - bmod
    ax_res.errorbar(ctrs[np.isfinite(bres)], bres[np.isfinite(bres)],
                    fmt='o', ms=3.5, color=col, markeredgecolor='k',
                    markeredgewidth=0.4, zorder=3)
    ax_res.axhline(0, c='k', lw=0.8)
    ax_res.set_ylabel("Residual", fontsize=8)
    ax_res.set_xlabel("Orbital phase", fontsize=9)
    ax_res.tick_params(labelsize=8)

    # ── annotation box ───────────────────────────────────────────────────────
    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if not np.isnan(v) else " N/A "

    d_phi_out = phi_out_data - phi_out_model if not np.isnan(phi_out_data) else np.nan
    d_phi_in  = (phi_in_data  - phi_in_model
                 if (not np.isnan(phi_in_data) and not np.isnan(phi_in_model))
                 else np.nan)
    d_depth   = data_depth - model_depth if not np.isnan(data_depth) else np.nan

    txt = (f"φ_out  model={_fmt(phi_out_model)}  data={_fmt(phi_out_data)}"
           f"  Δ={_fmt(d_phi_out, '+.4f')}\n"
           f"φ_in   model={_fmt(phi_in_model)}  data={_fmt(phi_in_data)}"
           f"  Δ={_fmt(d_phi_in, '+.4f')}\n"
           f"depth  model={_fmt(model_depth, '.3f')}  data={_fmt(data_depth, '.3f')}"
           f"  Δ={_fmt(d_depth, '+.3f')}")
    ax_lc.text(0.01, 0.03, txt, transform=ax_lc.transAxes,
               fontsize=6.5, family="monospace",
               va="bottom", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    summary_rows.append((band, phi_out_model, phi_out_data, d_phi_out,
                          phi_in_model, phi_in_data, d_phi_in,
                          model_depth, data_depth, d_depth))

fig.suptitle(f"{RUN_NAME}  —  Eclipse diagnostic  (black=model, colour=data)\n"
             f"dashed=outer contact (T1/T4), dotted=inner contact (T2/T3)",
             fontsize=10, y=1.01)
fig.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PDF}")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Band':5s} {'φ_out_mod':>10} {'φ_out_dat':>10} {'Δφ_out':>8}"
      f" {'φ_in_mod':>9} {'φ_in_dat':>9} {'Δφ_in':>7}"
      f" {'dep_mod':>8} {'dep_dat':>8} {'Δdepth':>8}")
print("-" * 95)
for row in summary_rows:
    band, pom, pod, dpо, pim, pid, dpi, dm, dd, dD = row
    def f(v): return f"{v:9.4f}" if not np.isnan(v) else "      N/A"
    def g(v): return f"{v:8.4f}" if not np.isnan(v) else "     N/A"
    def h(v): return f"{v:8.3f}" if not np.isnan(v) else "     N/A"
    print(f"{band:5s} {f(pom)} {f(pod)} {g(dpо)} {f(pim)} {f(pid)} {g(dpi)}"
          f" {h(dm)} {h(dd)} {h(dD)}")

print(f"\nInterpretation:")
for row in summary_rows:
    band = row[0]
    dpо, dpi, dD = row[3], row[6], row[9]
    msgs = []
    if not np.isnan(dpо):
        if dpо < -0.004:
            msgs.append(f"eclipse too WIDE by {abs(dpо)*period*1440:.1f} min "
                        f"→ r2 too large → reduce m2")
        elif dpо > 0.004:
            msgs.append(f"eclipse too NARROW by {dpо*period*1440:.1f} min "
                        f"→ r2 too small → increase m2")
    if not np.isnan(dD):
        if dD > 0.02:
            msgs.append(f"eclipse too SHALLOW by {dD:.3f} "
                        f"→ incl too low → increase incl")
        elif dD < -0.02:
            msgs.append(f"eclipse too DEEP by {abs(dD):.3f} "
                        f"→ incl too high → decrease incl")
    if msgs:
        print(f"  {band}: " + "; ".join(msgs))
    else:
        print(f"  {band}: contacts and depth within tolerance")
