# Clump Forward Modeling
## Author: Traci Johnson

Code to model sizes of clumps in gravitationally-lensed galaxies.

Description of original dataset and forward-modeling technique are published in the Astrophysical Journal, 843, p 78 (TLJ et al. 2017). Here is a link to the arXiv submission of this paper: https://arxiv.org/abs/1707.00707

#### File descriptions:
- clump\_forward\_modeling.py: Python2 script containing all the functions used by run\_mcmc.py
- run\_mcmc.py: Python2 script which digests the input file and creates the walkers needed for the MCMC.
- run\_mcmc: script for calling python from the command line and sending processes to the background
- clump_input.par: contains the initial parameters and names of files needed to initialize the MCMC
- dpl?2.fits: example deflection tensors in x and y, in units of arcseconds

Details of the files can be found in other markdown files.

#### Python Packages:
This code runs on Python2, but should be compatible with Python3 if the print statements are updated in the run\_mcmc.py script.
- numpy
- scipy
- astropy
- shapely
- emcee

Install these packages using "pip install [package]"


