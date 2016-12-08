import numpy as np
import pylab as pb
pb.ion()

def MA(z,w):
	nw = len(w)
	nz = len(z)
	nx = nz - nw
	x = np.zeros(nz)
	for i in range(nx):
		x[i+nw] = np.mean(z[i:i+nw]*w)
	return(x/np.mean(w))


####################################################
# white noise
x = np.arange(0,100)
y = np.random.normal(0,1,100)

pb.figure(figsize=(10,4))
pb.plot(x,y,"bx",mew=1.5)
pb.ylim((-3,3))
pb.savefig("1_whitenoise.pdf",bbox_inches="tight")

####################################################
# AR(1) process
x = np.arange(0,105)
y = np.random.normal(0,1,105)
z = y
for i in range(1,105):
	z[i] = 0.8*z[i-1] + y[i]

pb.figure(figsize=(8,4))
pb.plot(x,z,"x",mew=1.5)
pb.savefig("1_ar1a.pdf",bbox_inches="tight")


####################################################
# AR(1) process
x = np.arange(0,105)
y = np.random.normal(0,1,105)
z = np.cumsum(y)

pb.figure(figsize=(8,4))
pb.plot(x,z,"x",mew=1.5)
pb.savefig("1_ar1.pdf",bbox_inches="tight")

##
xpred = np.arange(105,120)
zpred = np.zeros(len(xpred))+z[-1]

pb.figure(figsize=(5,4))
pb.plot(x,z,"x",mew=1.5)
pb.plot(xpred,zpred,"xr",mew=1.5)
pb.savefig("1_ar1pred.pdf",bbox_inches="tight")

####################################################
# AR(2) process
x = np.arange(0,105)
y = np.random.normal(0,1,105)
z = y
for i in range(2,105):
	z[i] = 2*z[i-1] - z[i-2] + y[i]

pb.figure(figsize=(8,4))
pb.plot(x,z,"x",mew=1.5)
pb.savefig("1_ar2.pdf",bbox_inches="tight")

##
xpred = np.arange(105,120)
delta = z[-1] -z[-2]
zpred = z[-1] + np.arange(1,16) * delta

pb.figure(figsize=(5,4))
pb.plot(x,z,"x",mew=1.5)
pb.plot(xpred,zpred,"xr",mew=1.5)
pb.savefig("1_ar2pred.pdf",bbox_inches="tight")


####################################################
# MA process
x = np.arange(0,105)
y = np.random.normal(0,1,110)
nw=10
w = np.ones(nw)
z = MA(y,w)

pb.figure(figsize=(8,4))
pb.plot(x[:100],z[nw:],"x",mew=1.5)
pb.ylim((-1,1))
pb.xlim((0,120))
pb.savefig("1_ma.pdf",bbox_inches="tight")

def gamma(h,w):
	q = len(w)
	g = 0
	if h<q:
		g = np.sum(w[:q-h]*w[h:])
	return(g)

xpred = range(100,120)
gx = np.zeros((20,nw))
for i in range(20):
	for j in range(nw):
		gx[i,j] = gamma(xpred[i]-x[100-nw+j],w)


G = np.zeros((nw,nw))
for i in range(nw):
	for j in range(nw):
		G[i,j] = gamma(np.abs(x[100-nw+i]-x[100-nw+j]),w)

zpred = np.dot(gx,np.dot(np.linalg.inv(G),z[-nw:,None]))
pb.figure(figsize=(8,4))
pb.plot(x[:100],z[nw:],"x",mew=1.5)
pb.plot(xpred,zpred,"xr",mew=1.5)

pb.ylim((-1.5,1.5))
pb.xlim((0,120))
pb.savefig("1_mapred.pdf",bbox_inches="tight")

#{##} confidence intervals
zpred = np.dot(gx,np.dot(np.linalg.inv(G),z[-nw:,None]))
vpred = (nw - np.diag(np.dot(gx,np.dot(np.linalg.inv(G),gx.T))))/100
pb.figure(figsize=(8,4))
pb.plot(x[:100],z[nw:],"x",mew=1.5)
pb.plot(xpred,zpred,"xr",mew=1.5)
pb.plot(xpred,zpred-1.96*np.sqrt(vpred)[:,None],"-r",mew=1.5)
pb.plot(xpred,zpred+1.96*np.sqrt(vpred)[:,None],"-r",mew=1.5)

pb.ylim((-1.5,1.5))
pb.xlim((0,120))
pb.savefig("1_mapredconf.pdf",bbox_inches="tight")


####################################################
# MA process estimate q
x = np.arange(0,105)
y = np.random.normal(0,1,110)
nw=10
w = np.ones(nw)
z = MA(y,w)

pb.figure(figsize=(8,4))
pb.plot(x[:100],z[nw:],"x",mew=1.5)
pb.ylim((-1,1))
pb.xlim((0,100))
pb.savefig("1_mab.pdf",bbox_inches="tight")

pb.figure(figsize=(8,4))
pb.plot(acf(z))
pb.xlim((0,30))
pb.plot((0,30),[1.96/np.sqrt(100)]*2,'b--')
pb.plot((0,30),[-1.96/np.sqrt(100)]*2,'b--')
pb.savefig("1_mabEstQ.pdf",bbox_inches="tight")


def acovf(x, unbiased=False, demean=False, fft=False):
    '''
    Autocovariance for 1D

    Parameters
    ----------
    x : array
        Time series data. Must be 1d.
    unbiased : bool
        If True, then denominators is n-k, otherwise n
    demean : bool
        If True, then subtract the mean x from each element of x
    fft : bool
        If True, use FFT convolution.  This method should be preferred
        for long time series.

    Returns
    -------
    acovf : array
        autocovariance function
    '''
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)
    n = len(x)

    if demean:
        xo = x - x.mean()
    else:
        xo = x
    if unbiased:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    else:
        d = n * np.ones(2 * n - 1)
    if fft:
        nobs = len(xo)
        Frf = np.fft.fft(xo, n=nobs * 2)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[n - 1:]
        return acov.real
    else:
        return (np.correlate(xo, xo, 'full') / d)[n - 1:]




def acf(x, unbiased=False, nlags=40, qstat=False, fft=False, alpha=None):
    '''
    Autocorrelation function for 1d arrays.

    Parameters
    ----------
    x : array
       Time series data
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n
    nlags: int, optional
        Number of lags to return autocorrelation for.
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett\'s formula.

    Returns
    -------
    acf : array
        autocorrelation function
    confint : array, optional
        Confidence intervals for the ACF. Returned if confint is not None.
    qstat : array, optional
        The Ljung-Box Q-Statistic.  Returned if q_stat is True.
    pvalues : array, optional
        The p-values associated with the Q-statistics.  Returned if q_stat is
        True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    This is based np.correlate which does full convolution. For very long time
    series it is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted
    but the autocorrelation is not an unbiased estimtor.
    '''
    nobs = len(x)
    d = nobs  # changes if unbiased
    if not fft:
        avf = acovf(x, unbiased=unbiased, demean=True)
        #acf = np.take(avf/avf[0], range(1,nlags+1))
        acf = avf[:nlags + 1] / avf[0]
    else:
        x = np.squeeze(np.asarray(x))
        #JP: move to acovf
        x0 = x - x.mean()
        # ensure that we always use a power of 2 or 3 for zero-padding,
        # this way we'll ensure O(n log n) runtime of the fft.
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(x0, n=n)  # zero-pad for separability
        if unbiased:
            d = nobs - np.arange(nobs)
        acf = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d
        acf /= acf[0]
        #acf = np.take(np.real(acf), range(1,nlags+1))
        acf = np.real(acf[:nlags + 1])   # keep lag 0
    if not (qstat or alpha):
        return acf
    if alpha is not None:
        varacf = np.ones(nlags + 1) / nobs
        varacf[0] = 0
        varacf[1] = 1. / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1]**2)
        interval = stats.norm.ppf(1 - alpha / 2.) * np.sqrt(varacf)
        confint = np.array(lzip(acf - interval, acf + interval))
        if not qstat:
            return acf, confint
    if qstat:
        qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0
        if alpha is not None:
            return acf, confint, qstat, pvalue
        else:
            return acf, qstat, pvalue

