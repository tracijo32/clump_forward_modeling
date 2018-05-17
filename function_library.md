# Function Library

This file is a description of the functions listed in clump_forward_modeling.py.

- delens: takes x,y pixel values in the image plane and de-lenses them to the source plane
- getMagnification: returns the magnification of the pixel at x, y. currently only works if you have a magnification map.
- select_source_grid: returns mesh grid of the cropped portion of the source plane to paint the clumps onto. xx,yy are the x position and y position of each pixel
- orig_gaussian2d: **obsolete** my custom function for creating a 2d gaussian blob
- gaussian2d: code for creating a 2d gaussian blob using astropy's modeling repository
- sersic2dd: code for creating a 2d sersic blob using astropy's modeling repository
- mag2flux: computes the flux to a magnitude given a zeropoint. I can't remember why I needed this function.
- model_clumps: this is the code which takes all the parameters of all the clumps and creates an model of the source, returns a two dimensional image
- raytrace: **obsolete** old code for tracing pixel fluxes from source to image -- it is physically flawed because it doesn't conserve surface brightness, so don't use it!
- relens: opposite of delens, lenses x,y pixel values in the source plane to image plane; requires that you give it a region in the image plane to search for pixels (xa,ya)
- addFreePars: creates the x vector of free parameters that are called by the logProb function; needed for cooperation with emcee Ensemble Sampler
- makeParVector: basically the opposite of addFreePars, this takes the x vector of free parameters and places them in the array of parameter values for all the clumps
- drizzle_matrix: creates a sparse matrix mapping each image plane pixel to a source plane pixel. The values are the fraction of that image plane pixel area that overlaps with the area of a source plane pixel. Mostly zeros since most pixels don't overlap. It's best to save this matrix (pickle it or whatever) because it can take a few minutes to run the first time. Once you run it for a single cropping of the image plane and of the source plane, then you don't need to run it again. Dramatically improves speed of ray tracing. If you're running this from the terminal, there should be a progress bar that shows up.
- fo_drizzle: uses the output of drizzle_matrix to take the fluxes of all the pixels in the source plane model and trace them to the image plane. This does not do PSF convolution. This is the raw image.
- logProb_image_drz: this is the main function that is called by the emcee Ensemble Sampler. It takes a given set of free parameters x, creates a source plane model of those clumps, ray traces the source plane model to the image plane, convolves with PSF, and then computes the residual chi2. It returns the log of the probability, or -chi2.
- logProb_image: **obsolete** the original function back when I was using raytrace.
- createOutputs_drz: takes in the x vector with the best parameters and outputs the model ray traced to the image plane and creates a residual between data and model, so you can look at the result.
- createOutputs: **obsolete** created outputs back when I was using raytrace
- plotWalker: takes in the chain.p pickled object and plots the walker values at each step in the MCMC for each free parameter
- getLabels: writes labels for cornerplot
- cornerplot: takes in the flatchain.p and flatlnprobability.p pickled objects and creates a corner plot of all the parameters and covariances

Input Object: this is an object that when initialized reads in the input file (default: "clump_input.par") and saves all the values and priors for the parameters as well as save variables for settings

