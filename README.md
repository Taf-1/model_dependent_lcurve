# model_dependent_lcurve

Python wrapper around the [lcurve](https://github.com/trmrsh/cpp-lcurve) binary-star light-curve code for fitting ULTRACAM eclipse photometry when no spectroscopy is available. Physical parameters (masses, radii, temperatures) are coupled through white-dwarf cooling tracks and M-dwarf mass–radius relations so that every MCMC step produces a self-consistent binary model.

---

## Overview

The fitter samples over:
- **m1, m2** – masses (M☉); radii are derived from WD cooling tracks / M-dwarf MR relation
- **t1, t2** – effective temperatures (K)
- **incl** – orbital inclination (deg)
- **t0** – mid-eclipse time (BMJD)
- **parallax** – Gaia parallax (mas) with a Gaussian prior

Limb-darkening coefficients (4-term law), gravity-darkening coefficients, beam factors, and pivot wavelengths are all computed internally from the parameter values at every step – hence "model-dependent".

---

## Installation

### System Python (recommended – avoids the venv libpython issue)

```bash
# clone or unpack the repo
git clone <repo-url>
cd model_dependent_lcurve

# install lcurve first (Rust-backed, needs its own install)
pip install lcurve

# install remaining deps
pip install "numpy>=1.24" "scipy>=1.10" "astropy>=5.0" emcee h5py \
            "ruamel.yaml>=0.18" "astroquery>=0.4"
```

### From setup.py

```bash
pip install -e .
```

> **Python version**: 3.9 or later. The `.venv/` directory in this repo was built against Python 3.10 whose shared library is not available on this system — use the system Python 3.9 install instead.

---

## Data requirements

### Light-curve files

Six-column whitespace-delimited ASCII, one file per band:

| Column | Description |
|--------|-------------|
| time | BMJD (TDB) |
| t_exp | Exposure time (days) |
| flux | Calibrated flux (mJy or similar) |
| f_err | Flux uncertainty |
| weight | Per-point weight (1 = use, 0 = ignore) |
| n_div | Sub-exposure divisions for smearing correction |

### lcurve model template

A standard `.mod` file as used by `lcurve`. One template is provided at `targets/model_files/example.mod`.

### SED spectra (for `--setup` beam factors)

Two ASCII files per target, placed in `src/data/SED_data/`:
- `{run_name}_primary.txt` – WD SED from speedyfit (columns: wavelength Å, flux)
- `{run_name}_companion.txt` – companion SED

---

## Workflow

All commands are run from the **`src/`** directory.

```bash
cd src/
```

### 1. Setup – generate per-band model files

```bash
python3 lcurve_mcmc.py --conf /path/to/ASASSN18cv.yaml --setup
```

This reads the template `.mod`, sets limb/gravity-darkening, beam factors, and disc/spot switches for each band, and writes `model_files/{run_name}_{band}.mod`.

### 2. Test – single evaluation at the seed parameters

```bash
python3 lcurve_mcmc.py --conf /path/to/ASASSN18cv.yaml --test
```

Prints `ln_prob` and the derived geometry (log g, radii, separation, fill factor). Use this to verify the setup is sane before committing CPU time.

### 3. Fit – MCMC run

```bash
python3 lcurve_mcmc.py --conf /path/to/ASASSN18cv.yaml --fit \
        --nwalkers 100 --nburn 500 --nprod 5000 --nthreads 8
```

Results are saved to `MCMC_runs/{run_name}/{run_name}.h5` (emcee HDF5 backend). The run resumes automatically if the file already exists.

---

## Configuration file

See `targets/config_files/ASASSN18cv.yaml` for a worked example. Key sections:

```yaml
run_name: ASASSN18cv_run023    # used for output file names

light_curves:                  # absolute paths, one per band
  u': /abs/path/to/us_fc.dat
  g': /abs/path/to/gs_fc.dat
  r': /abs/path/to/rs_fc.dat

model_file: /abs/path/to/example.mod   # lcurve template

instrument: ucam               # ucam | hcam
period: 0.07543                # orbital period (days), held fixed
wd_core_comp: CO               # He | CO | ONe
wd_model: Koester              # Koester | Bergeron
ms_model: BT-SETTL             # BT-SETTL | BT-SETTL-CIFIST | PHOENIX-HiRes
secondary_mr: empirical        # empirical | baraffe

sed:                           # speedyfit seed values
  r1: 0.03   # R☉
  t1: 15000  # K
  r2: 0.11
  t2: 5400

params:                        # null → take from sed block
  m1: null
  m2: null
  t1: null
  t2: null
  incl: 75.0
  t0: 60994.294
  parallax: 2.0205

param_bounds:                  # hard prior walls
  m1: [0.2, 1.35]
  ...

priors:
  parallax: [gauss, 2.0205, 0.095]   # [gauss, mu, sigma] or [uniform, lo, hi]

run_settings:
  walkers: 100
  burnin: 500
  production: 5000
  n_cores: 8                   # max CPU cores for --fit

components:
  disc: true                   # include accretion disc in model
  spot: false
  secondary_eclipse: false
```

---

## Output

| Path | Contents |
|------|----------|
| `src/model_files/{run_name}_{band}.mod` | Per-band lcurve model files (from `--setup`) |
| `src/logs/{run_name}.log` | Rotating log file |
| `src/MCMC_runs/{run_name}/{run_name}.h5` | emcee chain (blobs: logg1, logg2, r1, r2, a, ffac, rva2) |
| `src/MCMC_runs/{run_name}/{run_name}.yaml` | Copy of the config used |

---

## Data tables bundled in `src/data/`

| Directory | Contents |
|-----------|----------|
| `ld_coeffs/` | 4-term limb-darkening coefficients for DA WDs and M-dwarfs (ucam, hcam, sdss) |
| `gravity_darkening_coeffs/` | GDCs for DA WDs and M-dwarfs |
| `blackbody_temps/` | T_eff → T_BB look-up tables for each atmosphere model and instrument |
| `filter_profiles/` | Transmission curves for ucam, hcam, sdss, panstarrs, 2MASS, WISE, Gaia |
| `cooling_tracks/` | WD cooling tracks (He, CO, ONe) and M-dwarf MR relations |
| `BedardTracks/` | Bédard+2020 WD evolutionary sequences |
| `PaneiTracks2007/` | Panei+2007 He-WD tracks (SDSS bands) |
| `SED_data/` | speedyfit SED spectra for beam-factor computation |
| `cam_cal/` | ULTRACAM/HiPERCAM calibration submodule |
