from cam_cal.observation import Observation

obs = Observation('ultracam')

# Add one or more runs on some relatively stable stars covering a decent airmass in order to fit the atmospheric extinction.
obs.add_observation(name='ZTFJ1341_atm', logfiles=['2021_01_22/run025_atm.log'], obs_type='atm')
obs.add_observation(name='ZTFJ1404_atm', logfiles=['2021_01_22/run027_atm.log'], obs_type='atm')

# Fit the atmospheric extinction.
obs.get_atm_ex()

# If the flux standard is not in the given tables then add its AB magnitudes in the correct filter system for UCAM/HCAM.
WD1225_006_mags = dict(mean={'us':15.357, 'gs':14.988, 'rs':15.022, 'is':15.135, 'zs':15.310},
                       err={'us':0.02, 'gs':0.02, 'rs':0.02, 'is':0.02, 'zs':0.02})

# Add the observation of the standard star.
obs.add_observation(name='WD1225+006', logfiles=['2021_01_22/run029_1.8.log'], obs_type='std', cal_mags=WD1225_006_mags)

# Calculate the zeropoint
obs.get_zeropoint()

# Add the run on the target.
obs.add_observation(name='ZTFJ1026', logfiles=['2021_01_22/run022_1.8.log'], obs_type='science')

# Flux calibrate the target using the Fitted atmospheric extinction and zeropoint.
obs.calibrate_science('ZTFJ1026', eclipse=1.5)