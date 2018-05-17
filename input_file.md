# Input file description

The input file (clump_input.par) is designed to be a more user-friendly way of updating the clump parameters in the source plane and pointing to all the relevant files neeeded for the MCMC to run.

## Files and settings:
- deflect [file_x.fits] [file_y.fits]: these are the FITS files containing the deflection matrices, usually these are in units of arcseconds; these should be the same pixel scale and grid as the data files (do not have to be same size, just cover the image plane regions around the images being modeled)
- rms [float]: the rms noise in the data
- kernel [float]: the size of the 2d gaussian PSF (sigma) in units of pixels
- data [data.fits]: file containing the data that you want to fit the model to
- aperture [ap.fits]: file containing the pixel mask for computing the chi2 residuals in the MCMC (1 for pixels used in the computation, 0 elsewhere); doesn't have to be anything fancy
- offset [xpix#] [ypix#]: the offset between the data file and the deflection field, this should correspond to the pixel in the data file that is the pixel (1,1) for the deflection file
- resolution [int]: the resampling level of pixels for the source plane model, ex. if the magnification is 10, this should be 10 so that the pixels in the source plane are 1/10th the size of the image plane pixels

## Clumps and parameters:
You can add a clump to the model by starting with the line "clump" followed by its starting parameters. He're an example:

```
clump 1
  profile gaussian
  x_im    100
  y_im    200
  flux    1
  size    0.1
  e       0.5
  pa      45
  end
limit 1
  flux  1 0   5
  size  2 0.1 0.02
  e     1 0.0 0.8
  pa    0 0   180
  end
```

A few things to note: what are the parameters?
- profile: this can be either "gaussian" or "sersic"
- x_im, y_im: the image plane pixel coordinates for the center of the clump
- size: Size in IMAGE PLANE pixels of the clump. For a gaussian, this is sigma. For a Sersic, this is the half light radius.
- e: ellipticity, defined as (a-b)/(a+b)
- pa: position angle in degrees, measured CCW from the x-axis
If any of these values are not placed under "clump", then the values for that clump all get defaulted to 0. The default profile is gaussian.

The limit indicates the priors on each variable. The first number after the parameter file indicates whether or not the parameter is fixed (set to 0) and the parameter value is fixed at the starting value for the entire MCMC. If the parameter is fixed, then the second two numbers are ignored. A value of 1 indicates a uniform random distribution for the prior where the next two values are the lower and upper bounds of the distribution. A value of 2 indicates a normal distribution where the next two values are the mean and standard deviation. Any parameter can be set to free or fixed. If the parameter is not listed under "limit", then the parameter is automatically fixed.
