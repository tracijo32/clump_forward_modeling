# date: 04/18/2018
# author: Traci Johnson
# Solves for the sizes of the clumps in the source plane by ray tracing
# a source plane model of clumps to image plane, convolving with instrument
# PSF, and then comparing with data.

# this code implements an MCMC (python 'emcee') to solve for clump parameters
# each clump is modeled as a 2d gaussian and has the following parameters:
#   - flux (amplitude)
#   - size (width)
#   - ellipticity
#   - position angle
#   - x location
#   - y position
# any combination of those parameters can be set fixed or free for any clump

# update 1/2018:
# have added a way to create a sersic profile for smooth components

###########################
## IMPORT external modules
###########################
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel,convolve_fft
from astropy.modeling.functional_models import Gaussian2D,Sersic2D
from scipy import interpolate,sparse
import emcee
import pickle
import sys
import timeit
from datetime import datetime
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import polygonize, unary_union
from astropy.utils.console import ProgressBar

###########################
# helper functions
###########################
def delens(ix,iy,deflectx,deflecty,dx=0,dy=0):
    # delenses a set of x and y coordinates in the image plane
    # dx,dy are offsets to origin if deflection matrices are cropped
    # ps=pixelscale of image

    x = ix-dx
    y = iy-dy

    n = len(ix)
    # load deflection matrices and convert from arcsec -> pix
    dplx = deflectx
    dply = deflecty

    srcx = np.zeros(n)
    srcy = np.zeros(n)

    # create interpolation functions for the deflection matrices
    xpixvals = np.arange(dplx.shape[0])
    ypixvals = np.arange(dply.shape[1])
    dplx_interp = interpolate.interp2d(xpixvals,ypixvals,dplx)
    dply_interp = interpolate.interp2d(xpixvals,ypixvals,dply)

    for i in range(n):
        deflectx = dplx_interp(x[i]-1,y[i]-1)[0]
        deflecty = dply_interp(x[i]-1,y[i]-1)[0]
        #deflectx = dplx_interp(x[i],y[i])[0]
        #deflecty = dply_interp(x[i],y[i])[0]
        srcx[i] = x[i] - deflectx
        srcy[i] = y[i] - deflecty
   
    srcx = srcx+dx
    srcy = srcy+dy
 
    return srcx,srcy

def getMagnification(ix,iy,magnification,dx=0,dy=0):
    # grab the magnification value at each image plane location (in pixels)
    # does fancy interpolation
    x = ix-dx
    y = iy-dy
   
    n = len(ix)

    xpixvals = np.arange(magnification.shape[0])
    ypixvals = np.arange(magnification.shape[1])
    mag_interp = interpolate.interp2d(xpixvals,ypixvals,magnification)

    mags = np.zeros(n)

    for i in range(n):
        mags[i] = mag_interp(x[i]-1,y[i]-1)[0]

    return mags

##############################################
# code needed to create output of lnprob_image
##############################################
def select_source_grid(dims,resolution=10):
    # selects the dimensions of the source plane grid
    xmin = dims[0]
    xmax = dims[1]
    ymin = dims[2]
    ymax = dims[3]

    # find range for an image with better resolution, as specified
    rx = (xmax-xmin)*resolution
    ry = (ymax-ymin)*resolution

    # make x and y coordinates of higher-rez image
    xx,yy = np.meshgrid(np.arange(rx),np.arange(ry))
    
    xx = xx.astype(float)/resolution+xmin
    yy = yy.astype(float)/resolution+ymin

    return xx,yy

def orig_gaussian2d(xx,yy,x0,y0,amp,width,e,pa):
    # creates a 1d gaussian (w/ ellipticity and pa) given the 
    # mesharrays xx and yy with the x and y values of the 2d array
    xp = (xx-x0)*np.cos(pa*np.pi/180)-(yy-y0)*np.sin(pa*np.pi/180)
    yp = (xx-x0)*np.sin(pa*np.pi/180)+(yy-y0)*np.cos(pa*np.pi/180)
    sigxp = width
    sigyp = width*np.sqrt((1-e)/(1+e))
    U = (xp/sigxp)**2+(yp/sigyp)**2
    return amp*np.exp(-U/2)

def gaussian2d(xx,yy,x0,y0,amp,width,e,pa):
    # creates a 2d gaussian (w/ ellipticity and pa) given the 
    # mesharrays xx and yy with the x and y values of the 2d array
    sigxp = width
    sigyp = width*np.sqrt((1-e)/(1+e))
    model = Gaussian2D(amplitude=amp, x_mean=x0, y_mean=y0, x_stddev=sigxp, 
                        y_stddev=sigyp, theta=np.pi*pa/180)
    return model(xx,yy)

def sersic2d(xx,yy,x0,y0,peak,reff,e,pa,n):
    # creates a 2d sersic profile, I = A*exp[-k*R**0.25]
    from scipy.special import gammaincinv
    bn = gammaincinv(2*n,0.5)
    amp = peak/np.exp(bn)
    
    model = Sersic2D(amplitude = amp, r_eff = reff, n=n, x_0=x0, y_0=y0,
               ellip=e, theta=np.pi*pa/180)#theta=np.pi*float(pa)/180)
    return model(xx,yy)

def mag2flux(m,mag0=20.0):
    return 10**(0.4*(mag0-m))

def model_clumps(xas,yas,x0,y0,flux,size,e,pa,n,ps_s=None,profile=None):
    # creates a model clumps brightness on a fine resolution source plane grid
    nclumps = len(x0)
    if profile is None:
        profile = ['gaussian']*nclumps
    

    # find range for an image with better resolution, as specified
    rx = (xas[1]-xas[0])/ps_s
    ry = (yas[1]-yas[0])/ps_s

    # make x and y coordinates of higher-rez image
    xx,yy = np.meshgrid(np.arange(rx),np.arange(ry))
    
    xx = xx.astype(float)*ps_s+xas[0]
    yy = yy.astype(float)*ps_s+yas[0]

    # superpose each gaussian onto the flux array
    f = xx*0
    for i in range(nclumps):
        if profile[i] == 'gaussian':
            f = f+gaussian2d(xx,yy,x0[i],y0[i],flux[i]*ps_s**2,size[i],e[i],pa[i])
        else:
            f = f+sersic2d(xx,yy,x0[i],y0[i],flux[i]*ps_s**2,size[i],e[i],pa[i],n[i])
    return f

def raytrace(f,xx,yy,deflectx,deflecty,dlsds=1,dx=0,dy=0,thresh_dis=0.1,thresh_val=0.001):
    ## ray traces a source plane image (with image coordinates given by mesh arrays
    ## xx and yy) back to the image plane
    ## duplicate code from K. Sharon's MATLAB scripts
    ## NOTE: this is the old ray tracing code that properly finds positions, but does not
    ## conserve the flux!

    xx = xx-dx
    yy = yy-dy

    dplx = deflectx*dlsds
    dply = deflecty*dlsds

    ## find the source plane position of every pixel in the image plane 
    dims = dplx.shape
    source_x = dplx*0
    source_y = dply*0
    if dims[0] == dims[1]:
        for i in range(dims[0]):
            source_x[:,i] = i + 1 - dplx[:,i]
            source_y[i,:] = i + 1 - dply[i,:]
    else:
        for j in range(dims[0]): source_x[:,j] = j + 1 - dplx[:,j]
        for k in range(dims[1]): source_y[k,:] = k + 1 - dply[k,:]

    ## array that has all the running indices of the deflection matrices
    all_indices = np.arange(dplx.size).reshape(dplx.shape)

    ## select pixels that are only lensed to a general region of the source plane
    conditions = np.array([source_x > np.amin(xx),
                           source_x < np.amax(xx),
                           source_y > np.amin(yy),
                           source_y < np.amax(yy)])
    mask = np.all(conditions,axis=0)

    index = all_indices[mask].flatten()
    

    ## now find the pixels that match the minimum distance and flux criteria
    fflat = f.flatten()
    sxflat = source_x.flatten()
    syflat = source_y.flatten()
    xxflat = xx.flatten()
    yyflat = yy.flatten()
    

    index_fine = np.array([])
    for i in range(f.size):
        if fflat[i] > thresh_val:
            dist = (sxflat[index]-xxflat[i])**2+(syflat[index]-yyflat[i])**2
            index_fine = np.append(index_fine,index[dist < thresh_dis])


    ## find the image plane positions that map to each source plane pixel
    index_fine_val = index_fine*0
    for i in range(index_fine.size):
        dist = (sxflat[index_fine[i]]-xxflat)**2+(syflat[index_fine[i]]-yyflat)**2
        closest = np.where(dist == np.amin(dist))[0][0]
        #index_fine_val[i] = fflat[dist == np.amin(dist)][0]
        index_fine_val[i] = fflat[closest]
    image = np.zeros(dplx.size)
    # changing this next line
    #for i in range(index_fine.size): image[index_fine[i]] = index_fine_val[i]
    for i in range(index_fine.size): image[index_fine[i]] = index_fine_val[i]
    image = image.reshape(dplx.shape)


    return image

def relens(x,y,deflectx,deflecty,xa,ya,dlsds=1,dx=0,dy=0):
    ## ray trace specific locations in the source plane back to the image plane
    
    x = x-dx
    y = y-dy

    dplx = deflectx*dlsds
    dply = deflecty*dlsds

    # create interpolation functions for the deflection matrices
    xpixvals = np.arange(dplx.shape[0])
    ypixvals = np.arange(dply.shape[1])
    dplx_interp = interpolate.interp2d(xpixvals,ypixvals,dplx)
    dply_interp = interpolate.interp2d(xpixvals,ypixvals,dply)
 
    ## find the source plane position of every pixel in the image plane 
    dims = dplx.shape
    source_x = dplx*0
    source_y = dply*0
    if dims[0] == dims[1]:
        for i in range(dims[0]):
            source_x[:,i] = i + 1 - dplx[:,i]
            source_y[i,:] = i + 1 - dply[i,:]
            #source_x[:,i] = i - dplx[:,i]
            #source_y[i,:] = i - dply[i,:]
    else:
        for j in range(dims[0]): source_x[:,j] = j + 1 - dplx[:,j]
        for k in range(dims[1]): source_y[k,:] = k + 1 - dply[k,:]
        #for j in range(dims[0]): source_x[:,j] = j - dplx[:,j]
        #for k in range(dims[1]): source_y[k,:] = k - dply[k,:]

    X,Y = np.meshgrid(np.arange(dplx.shape[1]),np.arange(dplx.shape[0]))
    conditions = np.array([X >= xa[0]-dx,
                           X <  xa[1]-dx,
                           Y >= ya[0]-dy,
                           Y <  ya[1]-dy])
    pixels = np.all(conditions,axis=0)

    #for i in range(n):
    #    deflectx = dplx_interp(x[i]-1,y[i]-1)[0]
    #    deflecty = dply_interp(x[i]-1,y[i]-1)[0]
    #    srcx[i] = x[i] - deflectx
    #    srcy[i] = y[i] - deflecty

    ix = np.zeros(x.size)
    iy = np.zeros(y.size)
    for i in range(len(x)):
        dist = (source_x-x[i])**2+(source_y-y[i])**2
        closest = np.where(dist[pixels].flat == np.amin(dist[pixels]))

        # find the approximate position of the source in the image plane
        ix_close = x[i] + (dplx[pixels]).flat[closest]
        iy_close = y[i] + (dply[pixels]).flat[closest]

        # trace around the approximate position until you find a spot really close to source plane
        gridsize = 1001
        ixval = np.linspace(ix_close-0.5,ix_close+0.5,gridsize)
        iyval = np.linspace(iy_close-0.5,iy_close+0.5,gridsize)
        ixgrid,iygrid = np.meshgrid(ixval,iyval)

        #deflectx = dplx_interp(ixval-1,iyval-1)
        #deflecty = dply_interp(ixval-1,iyval-1)
        deflectx = dplx_interp(ixval,iyval)
        deflecty = dply_interp(ixval,iyval)

        sxgrid = ixgrid - deflectx
        sygrid = iygrid - deflecty

        dist_fine = (sxgrid - x[i])**2+(sygrid - y[i])**2
        new_closest = np.where(dist_fine.flat == np.amin(dist_fine))[0][0]

        ix[i] = ixgrid.flat[new_closest]
        iy[i] = iygrid.flat[new_closest]

    ix = ix+dx
    iy = iy+dy

    return ix,iy


def addFreePars(x,params,fixfree):
    ## fills parameter matrix with the values from the free parameter vector x
    ## called by logL_image
    fp = 0
    for i in range(params.size):
        if fixfree.flat[i] == 1:
            params.flat[i] = x[fp]
            fp += 1
    return params

def makeParVector(params,fixfree):
    ## makes the vector for the params to shove into the logProb function
    x = []
    for i in range(params.size):
        if fixfree.flat[i] == 1:
            x.append(params.flat[i])
    return x

def drizzle_matrix(alpha_x,alpha_y,xa,ya,xas,yas,dx=0,dy=0,ps_s=0.1):
    ## -returns the matrix with the pixel fractions between the ray traced
    ##  image plane pixels in the source plane and the pixels on the new
    ##  grid in the source plane with a smaller pixel scale
    N = alpha_x.shape[0]

    ## correct for the bulk offset in pixels
    xa = xa-dx
    ya = ya-dy
    xas = xas-dx
    yas = yas-dy

    ## creates matrices of the indices assuming that 
    ## these are x,y coordinates of the centers
    ## of the pixels
    xmat,ymat = np.meshgrid(np.arange(xa[0],xa[1]),np.arange(ya[0],ya[1]))
    xmat0,ymat0 = np.meshgrid(np.arange(N),np.arange(N))

    ## create matrices of the coordinates of the corners
    ## assigned in clockwise order
    xmat_c = np.array([xmat+0.5,xmat-0.5,xmat-0.5,xmat+0.5])
    ymat_c = np.array([ymat-0.5,ymat-0.5,ymat+0.5,ymat+0.5])

    ## interpolate the alpha_x and alpha_y to give the deflection of the corners
    ## NOTE: the interpolation has edge effects, make sure the input deflection
    ## matrices are larger (by at least a pixel) than the cropped region
    pixvals = np.arange(N)
    xpixcorners = np.linspace(xa[0],xa[1],xa[1]-xa[0]+1)-0.5
    ypixcorners = np.linspace(ya[0],ya[1],ya[1]-ya[0]+1)-0.5
    alpha_x_interp = interpolate.interp2d(pixvals,pixvals,alpha_x)
    alpha_y_interp = interpolate.interp2d(pixvals,pixvals,alpha_y)
    ax = alpha_x_interp(xpixcorners,ypixcorners)
    ay = alpha_y_interp(xpixcorners,ypixcorners)

    ## output of the 2d interpolation is a 2d matrix that has all combinations the xy
    ## coordinates can mesh together, convert it to match the
    ## pixel coordinates matrices for the corners, match with xmat_c and ymat_c
    alpha_x_c = np.array([ax[:-1,1:],ax[:-1,:-1],ax[1:,:-1],ax[1:,1:]])
    alpha_y_c = np.array([ay[:-1,1:],ay[:-1,:-1],ay[1:,:-1],ay[1:,1:]])

    ## delens the corners, has source plane coordinates of each image plane pixel
    xs_c = xmat_c + 1 - alpha_x_c
    ys_c = ymat_c + 1 - alpha_y_c

    ## simple delensing of image plane pixel centers to source plane
    xs = xmat + 1 - alpha_x[int(ya[0]):int(ya[1]),int(xa[0]):int(xa[1])]
    ys = ymat + 1 - alpha_y[int(ya[0]):int(ya[1]),int(xa[0]):int(xa[1])]
 
    buff = 0
    xvec_s = np.arange(xas[0]-buff,xas[1]+buff,ps_s)
    yvec_s = np.arange(yas[0]-buff,yas[1]+buff,ps_s)
    xmat_s,ymat_s = np.meshgrid(xvec_s,yvec_s)

    ## make matrices of the corners of each of the source plane pixels
    xmat_s_c = np.array([xmat_s+0.5*ps_s,xmat_s-0.5*ps_s,xmat_s-0.5*ps_s,xmat_s+0.5*ps_s])
    ymat_s_c = np.array([ymat_s-0.5*ps_s,ymat_s-0.5*ps_s,ymat_s+0.5*ps_s,ymat_s+0.5*ps_s])
    
    ## sparse matrix: matrix of each image plane-source plane pixel pair
    ## most pixels don't overlap, so its mostly zeros
    ## only stores non-zero values loaded to matrix
    w=sparse.lil_matrix((xmat_s.size,xmat.size))
    a=sparse.lil_matrix((xmat_s.size,xmat.size))

    ## loop through all the source plane pixels
    print "STATUS: drizzling pixels, creating pixel fraction matrix"
    with ProgressBar(xmat_s.size) as bar:
        for i in range(xmat_s.size):
            ## update the progress bar
            bar.update()

            ## make a polygon for the source plane pixel
            poly_source=Polygon([(xmat_s_c[0,:,:].flat[i],ymat_s_c[0,:,:].flat[i]),
                                 (xmat_s_c[1,:,:].flat[i],ymat_s_c[1,:,:].flat[i]),
                                 (xmat_s_c[2,:,:].flat[i],ymat_s_c[2,:,:].flat[i]),
                                 (xmat_s_c[3,:,:].flat[i],ymat_s_c[3,:,:].flat[i])])
            ## now find the ray traced image plane pixels that are close enough to
            ## potentially overlap with this source plane pixel
            R = (xmat_s.flat[i]-xs)**2+(ymat_s.flat[i]-ys)**2
            I = np.where(R.flatten()<0.25)[0]
            for j in range(I.size):
                ## make a polygon for each image plane pixel traced back
                ## to the source plane
                jj = I[j]
                poly_image=Polygon([(xs_c[0,:,:].flat[jj],ys_c[0,:,:].flat[jj]),
                                    (xs_c[1,:,:].flat[jj],ys_c[1,:,:].flat[jj]),
                                    (xs_c[2,:,:].flat[jj],ys_c[2,:,:].flat[jj]),
                                    (xs_c[3,:,:].flat[jj],ys_c[3,:,:].flat[jj])])
                ## check if polygon has a self interaction, if it does, convert
                ## the polygon into two connected polygons
                if not poly_image.is_valid:
                    ls = LineString([(xs_c[0,:,:].flat[jj],ys_c[0,:,:].flat[jj]),
                                     (xs_c[1,:,:].flat[jj],ys_c[1,:,:].flat[jj]),
                                     (xs_c[2,:,:].flat[jj],ys_c[2,:,:].flat[jj]),
                                     (xs_c[3,:,:].flat[jj],ys_c[3,:,:].flat[jj])])
                    lr = LineString(ls.coords[:] + ls.coords[0:1])
                    mls = unary_union(lr)
                    poly_image = MultiPolygon(list(polygonize(mls)))
                ## if the image plane and source plane pixels intersect,
                ## then add the weight and area to the sparse matrices    
                poly_ovlp = poly_source.intersection(poly_image)
                if poly_ovlp.area > 0:
                    a[i,jj] = poly_ovlp.area/poly_source.area

    return a

def fo_drizzle(source,a,xa,ya,ps_s=0.1):
    ## forward drizzle a source plane sky model to the image plane
    ## takes as input the output matrix from drizzle_matrix

    ## create the blank matrix for the source plane grid
    #image = np.zeros((ya[1]-ya[0],xa[1]-xa[0]))
    source_flat = source.flatten()

    ## determine the outputs to generate
    #for i in range(image.size):
    #   if np.sum(a[:,i].toarray()) > 0:
    #       image.flat[i] = np.sum(np.squeeze(a[:,i].toarray())*source_flat)/ps_s**2

    ## faster version of the above method using pure array arithmetic
    #s = np.tile(source_flat,image.size).reshape(image.size,source.size).transpose()
    #image = np.sum(a.toarray()*s,axis=0)/ps_s**2
    #image = image.reshape(ya[1]-ya[0],xa[1]-xa[0])    

    ## memory-saving version of the method above, dot sparse matrix with the source image
    ## requires two transpositions of the sparse matrix to get the right dimensions
    image = a.transpose().dot(source_flat)/ps_s**2
    image = image.reshape(ya[1]-ya[0],xa[1]-xa[0])

    return image



def logProb_image_drz(x,data=None,aperture=None,params=None,fixfree=None,
                        xa=None,ya=None,xas=None,yas=None,dx=None,dy=None,
                        rms=None,psf=None,deflectx=None,deflecty=None,
                        priors=None,ps_s=None,a=None,profile=None):

    ## computes the posterior probability at the location in parameter space specified by the vector x
    ## function called by emcee
    ## can be modified to return 'arbitrary metadata blobs' by stating 'blobs=True'
    ## in that case, the function will return a list of the arrays with image, image+convolution, and source plane model
 
    ## place the free parameters in the correct positions using
    ## the params and fitfree arrays
    params = addFreePars(x,params,fixfree)    

    ## assemble the model for the parameters given
    x0 = params[:,0]
    y0 = params[:,1]
    f = params[:,2]
    size = params[:,3]
    e = params[:,4]
    pa = params[:,5]
    n = params[:,6]

    ## insert priors here
    logPrior = 0
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            if fixfree[i,j] == 1:
                if params[i,j] < priors[i,j,0] or params[i,j] > priors[i,j,1]: return -np.inf
            if fixfree[i,j] == 2:
                logPrior += (params[i,j]-priors[i,j,0])**2/(2*priors[i,j,1]**2) 

    ## create source plane model based on the free parameter selection for this walker
    source_model = model_clumps(xas,yas,x0,y0,f,size,e,pa,n,ps_s=ps_s,profile=profile)
    
    ## ray trace the source plane model to image plane
    #image_model = raytrace(source_model,xx,yy,deflectx,deflecty,dx=dx,dy=dy)
    image_model = fo_drizzle(source_model,a,xa,ya,ps_s=ps_s)

    ## crop data
    data *= aperture
    data = data[ya[0]:ya[1],xa[0]:xa[1]]

    ## convolve the model
    image_model_conv = convolve_fft(image_model,psf)
 
    ## compute log likelihood and log probability
    res = image_model_conv-data
    logL = -np.sum(res**2/(2*rms**2))#-0.5*n*np.log(2*np.pi)-np.sum(np.log(rms))
    logProb = logPrior + logL

    num_lines = sum(1 for line in open('chain.dat'))
    file = open('chain.dat','a')
    file.write(str(num_lines+1)+" ")
    for par in x:
        file.write("{0:0.4f} ".format(par))  
    file.write("{0:0.8f}\n".format(logProb))
    file.close()

    return logProb

def logProb_image(x,data=None,mask=None,params=None,fixfree=None,
    xx=None,yy=None,dx=None,dy=None,rms=None,psf=None,deflectx=None,deflecty=None,
    results=None,progress=None,blobs=False,priors=None,aperture=None):
    ## computes the posterior probability at the location in paramter space specified by the vector x
    ## function called by emcee
    ## can be modified to return 'arbitrary metadata blobs' by stating 'blobs=True'
    ## in that case, the function will return a list of the arrays with image, image+convolution, and source plane model
    ## NOTE: this is the old script that runs with the ray tracing code


    ## place the free parameters in the correct positions using
    ## the params and fitfree arrays
    params = addFreePars(x,params,fixfree)    

    ## assemble the model for the parameters given
    x0 = params[:,0]
    y0 = params[:,1]
    f = params[:,2]
    size = params[:,3]
    e = params[:,4]
    pa = params[:,5]

    ## insert priors here
    logPrior = 0
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            if fixfree[i,j] == 1:
                if params[i,j] < priors[i,j,0] or params[i,j] > priors[i,j,1]: return -np.inf
            if fixfree[i,j] == 2:
                logPrior += (params[i,j]-priors[i,j,0])**2/(2*priors[i,j,1]**2) 
    ## set wall on parameter selections that have stupid values
    #if np.amax(x0) > np.amax(xx) or np.amin(x0) < np.amin(xx): return -np.inf
    #if np.amax(y0) > np.amax(yy) or np.amin(y0) < np.amin(yy): return -np.inf
    #if np.amin(f) < 0 or np.amax(f) > 5: return  -np.inf
    #if np.amin(size) < 0.05 or np.amax(size) > 0.4: return -np.inf
    #if np.amax(e) >= 0.95 or np.amin(e) < 0: return -np.inf
    #if np.amax(pa) > 180 or np.amin(pa) < 0: return -np.inf

    ## create source plane model based on the free parameter selection for this walker
    source_model = model_clumps(xx,yy,x0,y0,f,size,e,pa)
    
    ## ray trace the source plane model to image plane
    image_model = raytrace(source_model,xx,yy,deflectx,deflecty,dx=dx,dy=dy)

    ## mask out image and data
    image_model = image_model[mask[2]-dy:mask[3]-dy,mask[0]-dx:mask[1]-dx]
    data = data[mask[2]:mask[3],mask[0]:mask[1]]
    aperture = aperture[mask[2]:mask[3],mask[0]:mask[1]]

    ## convolve the model
    image_model_conv = convolve_fft(image_model,psf)

    res = image_model_conv-data
    res = res*aperture

    # weight the data by S/N
    logL = -np.sum(res**2/(2*rms**2))#-0.5*n*np.log(2*np.pi)-np.sum(np.log(rms))

    logProb = logPrior + logL

    if results != None:
        num_lines = sum(1 for line in open(results))
        file = open(results,'a')
        file.write(str(num_lines+1)+" ")
        for par in x:
            file.write("{0:0.4f} ".format(par))  
        file.write("{0:0.8f}\n".format(logProb))
        file.close()
   
    if blobs:
        blobs = [image_model_conv,source_model]
        return logProb,blobs
    else:
        return logProb

def createOutputs_drz(params,str='',data=None,xa=None,ya=None,xas=None,yas=None,
        dx=None,dy=None,psf=None,a=None,ps_s=None,profile=None):
    ## creates the fits files outputs given the parameter array
    ## outputs include:
    ## -- source plane model
    ## -- image plane model (masked)
    ## -- image plane model, convolved w/ psf (masked)
    ## -- data (masked)

    ## assemble the model for the parameters given
    x0 = params[:,0]
    y0 = params[:,1]
    f = params[:,2]
    size = params[:,3]
    e = params[:,4]
    pa = params[:,5]
    n = params[:,6]

    source_model = model_clumps(xas,yas,x0,y0,f,size,e,pa,n,ps_s=ps_s,profile=profile)
    fits.writeto('source'+str+'.fits',source_model,overwrite=True)    

    #print a.shape
    #print source_model.shape

    image_model = fo_drizzle(source_model,a,xa,ya,ps_s=ps_s)

    data = data[ya[0]:ya[1],xa[0]:xa[1]]

    fits.writeto('image'+str+'.fits',image_model,overwrite=True)
    fits.writeto('data'+str+'.fits',data,overwrite=True)

    ## convolve the model
    image_model_conv = convolve_fft(image_model,psf)
    fits.writeto('image_convolve'+str+'.fits',image_model_conv,overwrite=True)

    res = image_model_conv-data
    fits.writeto('residual'+str+'.fits',res,overwrite=True)

def createOutputs(params,str='',data=None,mask=None,xx=None,yy=None,dx=None,dy=None,
        psf=None,deflectx=None,deflecty=None,profile=None):
    ## creates the fits files outputs given the parameter array
    ## outputs include:
    ## -- source plane model
    ## -- image plane model (masked)
    ## -- image plane model, convolved w/ psf (masked)
    ## -- data (masked)

    ## assemble the model for the parameters given
    x0 = params[:,0]
    y0 = params[:,1]
    f = params[:,2]
    size = params[:,3]
    e = params[:,4]
    pa = params[:,5]

    source_model = model_clumps(xx,yy,x0,y0,f,size,e,pa,profile=profile)
    fits.writeto('source'+str+'.fits',source_model,overwrite=True)    

    image_model = raytrace(source_model,xx,yy,deflectx,deflecty,dx=dx,dy=dy)

    image_model = image_model[mask[2]-dy:mask[3]-dy,mask[0]-dx:mask[1]-dx]
    data = data[mask[2]:mask[3],mask[0]:mask[1]]

    fits.writeto('image'+str+'.fits',image_model,overwrite=True)
    fits.writeto('data'+str+'.fits',data,overwrite=True)

    ## convolve the model
    image_model_conv = convolve_fft(image_model,psf)
    fits.writeto('image_convolve'+str+'.fits',image_model_conv,overwrite=True)

    res = image_model_conv-data
    fits.writeto('residual'+str+'.fits',res,overwrite=True)
 
#####################################
# scripts for loading inputs to model   
#####################################
class Inputs:
    def __init__(self,file='clump_input.par'):
        f = open(file,'r')
        lines = f.readlines()
        f.close()

        index = np.zeros(100)
        params = np.zeros((100,7))
        fixfree = np.zeros((100,7))
        priors = np.zeros((100,7,2))
        profile = []

        par_list = {'x_im':0,'y_im':1,'flux':2,'size':3,'e':4,'pa':5,
                    'x':0, 'y':1, 'n':6}

        i = 0
        j = 0
        while i < len(lines):
            if lines[i][0] == '#':
                lines[i][0]
                i+=1
                continue
            ls = lines[i].rsplit()
            if len(ls) == 0:
                i+=1
                continue
            ## load setup parameters
            if ls[0] == 'nthreads':
                self.nthreads = int(ls[1])
            if ls[0] == 'burnin':
                self.burnin = int(ls[1])
            if ls[0] == 'sample':
                self.sample = int(ls[1])
            if ls[0] == 'img_crop':
                self.img_crop = np.array(ls[1:5]).astype(int)
            if ls[0] == 'deflect':
                self.deflectx = ls[1]
                self.deflecty = ls[2]
            if ls[0] == 'offset': 
                self.dx = int(ls[1])
                self.dy = int(ls[2])
            if ls[0] == 'data': self.data = ls[1]
            if ls[0] == 'rms': self.rms = float(ls[1])
            if ls[0] == 'ps': self.ps = float(ls[1])
            if ls[0] == 'kernel': self.kernel = float(ls[1])
            if ls[0] == 'resolution': self.resolution = float(ls[1])
            if ls[0] == 'aperture': self.aperture = ls[1]
            ## load in the clump parameters
            if ls[0] == 'clump':
                index[j] = int(ls[1])
                i += 1
                ls = lines[i].rsplit()
                while ls[0] != 'end':
                    if ls[0] == 'profile':
                        profile.append(ls[1])
                    else:
                        params[j,par_list[ls[0]]] = ls[1]
                    i += 1
                    ls = lines[i].rsplit()
                i+=2
                ls = lines[i].rsplit()
                while ls[0] != 'end':
                    fixfree[j,par_list[ls[0]]] = ls[1]
                    priors[j,par_list[ls[0]],:] = ls[2:]
                    i+=1
                    ls = lines[i].rsplit()
                j+=1
            i+=1
        
        self.index = index[:j].astype('int64')
        self.params = params[:j,:]
        self.fixfree = fixfree[:j,:]
        self.priors = priors[:j,:,:]

        self.nclumps = j

        if len(profile) == 0:
            self.profile = ['gaussian']*j
        else:
            self.profile = profile
#################################
# plotting scripts
#################################
def plotWalker(chain,param):
    ## plot each step of walkers
    for i in range(chain.shape[0]):
        plt.plot(chain[i,:,param])
    plt.show()

def getLabels(params,fixfree,index):
    ## returns array of strings for labeling the plots in cornerplot()
    index_arr = np.array([index,]*params.shape[1]).transpose().astype('str')
    names_arr = np.array([['x','y','flux','size','e','pa','n'],]*params.shape[0])
    labels = []
    for i in range(params.size):
        if fixfree.flat[i] > 0:
            labels.append('#'+index_arr.flat[i]+' '+names_arr.flat[i])
    return labels

def cornerplot(flatchain,flatlnprobability,params=np.array([]),labels=None,chi2=False):
    ## plots each of the MCMC parameters against one another in a 'corner plot'
    if len(params) > 1:
        flatchain = flatchain[:,params]
        if len(labels) > 1: labels = labels[params]
    nparams = flatchain.shape[1]
    fig, axarr = plt.subplots(nparams,nparams)#,sharex='row',sharey='col')
    fig.subplots_adjust(hspace=0,wspace=0)

    best = np.where(flatlnprobability == np.amax(flatlnprobability))[0][0]
    for j in range(nparams):
        for i in range(nparams):
            xbins = np.linspace(np.amin(flatchain[:,i]),np.amax(flatchain[:,i]),25)
            if i==j:
                if chi2:
                    axarr[j,i].plot(flatchain[:,i],-flatlnprobability,'o',color='blue')
                    axarr[j,i].plot([flatchain[best,i]],[-flatlnprobability[best]],'s',color='red')
                    axarr[j,i].set_ylim(np.amin(-flatlnprobability)*0.9,np.amax(-flatlnprobability)*1.1)
                else:
                    h,b,p=axarr[j,i].hist(flatchain[:,i],bins=xbins)
                    axarr[j,i].plot(flatchain[best,i]*np.ones(2),np.array([0,np.amax(h)*1.1]),color='grey',linewidth=1)
                    axarr[j,i].set_ylim(0,np.max(h)*1.1)
                axarr[j,i].set_xlim(np.amin(flatchain[:,i]),np.amax(flatchain[:,i]))
            elif i<j:
                ybins = np.linspace(np.amin(flatchain[:,j]),np.amax(flatchain[:,j]),25)
                axarr[j,i].hist2d(flatchain[:,i],flatchain[:,j],bins=(xbins,ybins))
                axarr[j,i].plot(flatchain[best,i]*np.ones(2),ybins[[0,-1]],color='grey',linewidth=1)
                axarr[j,i].plot(xbins[[0,-1]],flatchain[best,j]*np.ones(2),color='grey',linewidth=1)
                axarr[j,i].set_xlim(np.amin(flatchain[:,i]),np.amax(flatchain[:,i]))
                axarr[j,i].set_ylim(np.amin(flatchain[:,j]),np.amax(flatchain[:,j]))
            else:
                axarr[j,i].axis('off')
            if i>0:
                axarr[j,i].yaxis.set_visible(False)
            else:
                if len(labels) > 1: axarr[j,i].set_ylabel(labels[j])
            if j<nparams-1:
                axarr[j,i].xaxis.set_visible(False)
            else:
                ticklabels = axarr[j,i].get_xticklabels() 
                for tl in ticklabels: tl.set_rotation(90) 
                if len(labels) > 1: axarr[j,i].set_xlabel(labels[i])
    axarr[0,0].yaxis.set_visible(False)

    plt.show()
