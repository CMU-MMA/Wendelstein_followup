# step1. download ls DR10 bricks for all observations.
python lsdown.py

# step2. flux calibration and derive fwhm, limiting magnitudes of science images.
python preproc.py

# step3. image subtraction with existing decals_* images
# NOTE: it is currently a beta version, not fully optimized yet.
python quicksub.py

# WARNING:
# 1. ./ligo_wst is an empty directory (only to demonstrate file structure), please replace by the real one..
# 2. Please change directory name by removing the space
#    ./ligo_wst/data/S230615az/Growth Fup >>> ./ligo_wst/data/S230615az/GrowthFup
# 3. please remove the empty directory: ./lsdr10
