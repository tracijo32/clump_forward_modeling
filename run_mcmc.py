from clump_forward_modeling import *

#####################################
# RUN MCMC
####################################
if __name__ == '__main__':
    ## load the input file
    inputs = Inputs(file='clump_input.par')
    nclumps = inputs.nclumps
    dx = inputs.dx
    dy = inputs.dy
    kernel = inputs.kernel
    aperture = inputs.aperture
    ps = inputs.ps
    rms = inputs.rms
    profile = inputs.profile

    ## load deflection matrices
    deflectx = fits.getdata(inputs.deflectx)/ps
    deflecty = fits.getdata(inputs.deflecty)/ps

    ## find sou:qrce plane locations of clumps
    params = inputs.params

    ## create a mask for the cropped region of the image, covering only where we want the clumps
    ixmin = np.floor(np.amin(params[:,0]))-10
    ixmax = np.floor(np.amax(params[:,0]))+11
    iymin = np.floor(np.amin(params[:,1]))-10
    iymax = np.floor(np.amax(params[:,1]))+11

    ## find clump positions in the source plane
    params[:,0],params[:,1] = delens(params[:,0],params[:,1],deflectx,deflecty,dx=dx,dy=dy)

    ## select region of the source plane to place the grid 
    xmin = np.floor(np.amin(params[:,0]))-2
    xmax = np.floor(np.amax(params[:,0]))+3
    ymin = np.floor(np.amin(params[:,1]))-2
    ymax = np.floor(np.amax(params[:,1]))+3

    ## create fixed/free array: if the parameter is free, then set value equal to 1
    fixfree = inputs.fixfree
    ## or select your own parameters to set free
    #fixfree = params*0
    #fixfree[5,2:] += 1

    ## set the bounds for the priors: all assumed to be uniform random distributions
    priors = inputs.priors

    xa = np.array([ixmin,ixmax]).astype(int)
    ya = np.array([iymin,iymax]).astype(int)
    xas = np.array([xmin,xmax]).astype(int)
    yas = np.array([ymin,ymax]).astype(int)

    ps_s = 1.0/inputs.resolution
    psf = Gaussian2DKernel(inputs.kernel)

    data = fits.getdata(inputs.data)
    #aperture = fits.getdata(inputs.aperture)
    aperture = np.ones(data.shape)

    import os
    if os.path.exists('amat.p'):
        a = pickle.load(open('amat.p','rb'))
    else:
        a = np.zeros((2,2))
    if a.shape[0] != (xmax-xmin)*(ymax-ymin)/ps_s and a.shape[1] != (ixmax-ixmin)*(iymax-iymin):
        a = drizzle_matrix(deflectx,deflecty,xa,ya,xas,yas,ps_s=ps_s,dx=dx,dy=dy)
        pickle.dump(a,open('amat.p','wb'))
    
    if 1:
        ## set up # of walkers and # of free parameters
        ndim = int(np.sum(fixfree > 0))
        nwalkers = ndim*10

        ## set up initial positions of the walkers
        inits = []
        for n in range(nwalkers):
            x = []
            walker = np.zeros((nclumps,7))
            for i in range(nclumps):
                for j in range(7):
                    if fixfree[i,j] == 1:
                        walker[i,j] = np.random.uniform(low=priors[i,j,0],high=priors[i,j,1],size=1)
                    if fixfree[i,j] == 2:
                        walker[i,j] = np.random.normal(loc=priors[i,j,0],scale=priors[i,j,1],size=1)

            for i in range(walker.size):
                if fixfree.flat[i] == 1:
                    x.append(walker.flat[i])
            inits.append(x)

        #print logProb_image_drz(inits[0],data=data,params=params,fixfree=fixfree,xa=xa,ya=ya,xas=xas,yas=yas,dx=dx,dy=dy,rms=rms,psf=psf,deflectx=deflectx,deflecty=deflecty,priors=priors,ps_s=ps_s,a=a)

        ## get labels of the free parameters   
        labels = getLabels(params,fixfree,inputs.index)
        print labels
        ## create a blank file to write results, keep track of progress
        results = 'chain.dat'
        f = open(results,'w')
        f.close()
     
        ## set up the arguments to pass into the function that computes the log probability
        fit_args = {"params":params,
                    "fixfree":fixfree,
                    "priors":priors,
                    "data":data,
                    "rms":rms,
                    "ps_s":ps_s,
                    "a":a,
                    "dx":dx,
                    "dy":dy,
                    "xa":xa,
                    "ya":ya,
                    "xas":xas,
                    "yas":yas,
                    "psf":psf,
                    "deflectx":deflectx,
                    "deflecty":deflecty,
                    "aperture":aperture,
                    "profile":profile
                    }
    
        ## initiate the sampler
        nthreads=8
        print str(datetime.now())
        print 'initiating MCMC ensemble sampler ({2:d} threads) with {0:d} walkers {1:d} free parameters'.format(nwalkers, ndim, nthreads)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logProb_image_drz, kwargs=fit_args, threads=nthreads)
    
        ## run the burn-in
        nsteps = 50
        print 'running the burn in for {0:d} steps'.format(nsteps)
        start = timeit.default_timer()
        pos,prob,state = sampler.run_mcmc(inits,nsteps)   
        stop = timeit.default_timer()
        print "burn in complete. run time: {0:.1f} min".format((stop-start)/60)
    
        ## restart and run the sampling
        sampler.reset()
        nsteps = 500
        print 'sampling for {0:d} steps'.format(nsteps)
        start = timeit.default_timer()
        sampler.run_mcmc(pos,nsteps)
        stop = timeit.default_timer()
        print "MCMC complete. run time: {0:.1f} min".format((stop-start)/60)
    
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    
        ## create outputs for the best model in this chain
        bestfit = np.where(sampler.flatlnprobability == np.amax(sampler.flatlnprobability))[0][0]
        xbest = sampler.flatchain[bestfit,:]
        print xbest
        print 'creating outputs'
        bestparams = addFreePars(xbest,params,fixfree)
        createOutputs_drz(bestparams,str='',data=data,xa=xa,ya=ya,xas=xas,yas=yas,dx=dx,dy=dy,psf=psf,a=a,ps_s=ps_s)
        
        ## save outputs by pickling them
        print 'pickling sample chain'
        sys.setrecursionlimit(10000)
        pickle.dump(sampler.chain,open('chain.p','wb'))
        pickle.dump(sampler.flatchain,open('flatchain.p','wb'))
        pickle.dump(sampler.lnprobability,open('lnprobability.p','wb'))
        pickle.dump(sampler.flatlnprobability,open('flatlnprobability.p','wb')) 
        #pickle.dump(sampler.blobs,open('blobs.p','wb'))
        pickle.dump(labels,open('labels.p','wb'))
        pickle.dump(bestparams,open('bestparams.p','wb'))
        pickle.dump(fixfree,open('fixfree.p','wb'))
        #pickle.dump(sampler.acor,open('acor.p','wb'))
        #pickle.dump(sampler.acceptance_fraction,open('acceptance_fraction.p','rb'))
        print 'done.'
    else:
        createOutputs_drz(params,str='',data=data,xa=xa,ya=ya,xas=xas,yas=yas,dx=dx,dy=dy,psf=psf,a=a,ps_s=ps_s)
        #x = makeParVector(params,fixfree)
        #print x
        #print logProb_image_drz(x,data=data,aperture=aperture,params=params,fixfree=fixfree,xa=xa,ya=ya,xas=xas,yas=yas,dx=dx,dy=dy,rms=rms,psf=psf,deflectx=deflectx,deflecty=deflecty,priors=priors,ps_s=ps_s,a=a)
        
