from cam_cal.fits import FITS
from argparse import ArgumentParser


parser = ArgumentParser(description=__doc__)
parser.add_argument('fname', type=str,
                    help='Name of fits file')

    # Parse arguments
args = parser.parse_args()
fname = args.fname

with FITS.open(fname) as f:
    f.to_lcurve()

