import numpy as np
from scipy import stats
from astropy import units as u

def power_spect_nd(input_array, box_dims, verbose=True):
	''' 
	Calculate the power spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array
	box_side is the size of the box in comoving Mpc. If this is set to None (default),
	the internal box size is used
	
	Parameters:
		* input_array (numpy array): the array to calculate the 
			power spectrum of. Can be of any dimensions.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
	
	Returns:
		The power spectrum in the same dimensions as the input array.		
	'''

	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
	if verbose: print( 'Calculating power spectrum...')
	ft = np.fft.fftshift(np.fft.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2
	if verbose: print( '...done')

	# scale
	#print(box_dims)
	boxvol = np.product(box_dims)
	#print(boxvol)
	pixelsize = boxvol/(np.product(input_array.shape))
	power_spectrum *= pixelsize**2/boxvol
	
	return power_spectrum

# def _get_k(input_array, box_dims):
# 	'''
# 	Get the k values for input array with given dimensions.
# 	Return k components and magnitudes.
# 	For internal use.
# 	'''
# 	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
# 	dim = len(input_array.shape)
# 	if dim == 1:
# 		x = np.arange(len(input_array))
# 		center = x.max()/2.
# 		kx = 2.*np.pi*(x-center)/box_dims[0]
# 		return [kx], kx
# 	elif dim == 2:
# 		x,y = np.indices(input_array.shape, dtype='int32')
# 		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2])
# 		kx = 2.*np.pi * (x-center[0])/box_dims[0]
# 		ky = 2.*np.pi * (y-center[1])/box_dims[1]
# 		k = np.sqrt(kx**2 + ky**2)
# 		return [kx, ky], k
# 	elif dim == 3:
# 		x,y,z = np.indices(input_array.shape, dtype='int32')
# 		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2, \
# 						(z.max()-z.min())/2])
# 		kx = 2.*np.pi * (x-center[0])/box_dims[0]
# 		ky = 2.*np.pi * (y-center[1])/box_dims[1]
# 		kz = 2.*np.pi * (z-center[2])/box_dims[2]

# 		k = np.sqrt(kx**2 + ky**2 + kz**2 )
# 		return [kx,ky,kz], k

def _get_k(input_array, box_dims):
    '''
    Get the k values for input array with given dimensions.
    Return k components and magnitudes.
    For internal use.
    '''
    dim = len(input_array.shape)
    if np.array(box_dims).size!=dim: box_dims = np.array([box_dims for i in range(dim)])
    if dim == 1:
        nx = input_array.shape[0]
        x = np.arange(len(input_array))
        center = nx/2 if nx%2==0 else (nx-1)/2
        kx = 2.*np.pi*(x-center)/box_dims[0]
        return [kx], kx
    elif dim == 2:
        nx,ny = input_array.shape
        x,y = np.indices(input_array.shape, dtype='int32')
        center = np.array([nx/2 if nx%2==0 else (nx-1)/2, ny/2 if ny%2==0 else (ny-1)/2])
        kx = 2.*np.pi * (x-center[0])/box_dims[0]
        ky = 2.*np.pi * (y-center[1])/box_dims[1]
        k = np.sqrt(kx**2 + ky**2)
        return [kx, ky], k
    elif dim == 3:
        nx,ny,nz = input_array.shape
        x,y,z  = np.indices(input_array.shape, dtype='int32')
        center = np.array([nx/2 if nx%2==0 else (nx-1)/2, ny/2 if ny%2==0 else (ny-1)/2, \
                            nz/2 if nz%2==0 else (nz-1)/2])
        kx = 2.*np.pi * (x-center[0])/box_dims[0]
        ky = 2.*np.pi * (y-center[1])/box_dims[1]
        kz = 2.*np.pi * (z-center[2])/box_dims[2]

        k = np.sqrt(kx**2 + ky**2 + kz**2 )     
        return [kx,ky,kz], k

def _get_kF(box_dims):
    '''
    Get the k values for input array with given dimensions.
    Return k components and magnitudes.
    For internal use.
    '''
    dim = 1 if isinstance(box_dims, (int,float)) else len(box_dims)
    if dim == 1:
        kx = 2.*np.pi/box_dims
        return kx
    return 2.*np.pi/np.array(box_dims)

def _get_nk(input_array, box_dims):
    '''
    Get the k values for input array with given dimensions.
    Return k components and magnitudes.
    For internal use.
    '''
    dim = len(input_array.shape)
    if np.array(box_dims).size!=dim: box_dims = np.array([box_dims for i in range(dim)])
    if dim == 1:
        nx = input_array.shape[0]
        x = np.arange(len(input_array))
        center = nx/2 if nx%2==0 else (nx-1)/2
        kx = x-center
        return [kx], kx
    elif dim == 2:
        nx,ny = input_array.shape
        x,y = np.indices(input_array.shape, dtype='int32')
        center = np.array([nx/2 if nx%2==0 else (nx-1)/2, ny/2 if ny%2==0 else (ny-1)/2])
        kx = (x-center[0])
        ky = (y-center[1])
        k = np.sqrt(kx**2 + ky**2)
        return [kx, ky], k
    elif dim == 3:
        nx,ny,nz = input_array.shape
        x,y,z  = np.indices(input_array.shape, dtype='int32')
        center = np.array([nx/2 if nx%2==0 else (nx-1)/2, ny/2 if ny%2==0 else (ny-1)/2, \
                            nz/2 if nz%2==0 else (nz-1)/2])
        kx = (x-center[0])
        ky = (y-center[1])
        kz = (z-center[2])

        k = np.sqrt(kx**2 + ky**2 + kz**2 )     
        return [kx,ky,kz], k

def power_spect_1d(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False, kmin=None, kmax=None):
	power = power_spect_nd(input_array, box_dims, verbose=0)
	[kx,ky,kz], k = _get_k(input_array, box_dims)
	if kmin is None: kmin = 2*np.pi/np.array(box_dims).max()
	if kmax is None: kmax = k.max()
	if binning=='log':
		k  = np.log10(k)
		ks = get_auto_bins(k) if kbins=='auto' else np.linspace(np.log10(kmin), np.log10(kmax), kbins+1)
	elif binning=='linear': 
		ks = get_auto_bins(k) if kbins=='auto' else np.linspace(kmin, kmax, kbins+1)

	if kbins!='auto': ks = (ks[:-1]+ks[1:])/2.
	kkk = np.hstack((k.min(),(ks[1:]+ks[:-1])/2, k.max()))
	k_width = kkk[1:]-kkk[:-1]
	kbins = ks.size
	ps = np.zeros(kbins)
	n_modes = np.zeros(kbins)
	k, power = k.flatten(), power.flatten()
	for i,a in enumerate(ks):
		arg = np.argwhere(np.abs(k-a)<=k_width[i]/2.)
		ps[i] = power[arg].sum()
		n_modes[i] = arg.size

	ps = ps/n_modes
	if binning=='log': ks = 10**ks
	if return_modes: return ps, ks, n_modes
	return ps, ks

def get_auto_bins(k):
	#kernel = stats.gaussian_kde(values)
	kbins  = 100.
	n_bin  = 1.*k.size/kbins
	k_sort = np.sort(k.flatten())
	ks = np.array([k_sort[int(i)] for i in np.arange(n_bin/2., k.size, n_bin)])
	return ks

def power_spect_2d(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False, nu_axis=2):
	if np.array(kbins).size==1: kbins = [kbins, kbins]
	power = power_spect_nd(input_array, box_dims, verbose=0)
	[kx,ky,kz], k = _get_k(input_array, box_dims)
	kdict = {}
	kdict['0'], kdict['1'], kdict['2'] = kx, ky, kz
	del kx, ky, kz
	kz = kdict[str(nu_axis)]
	kp = np.sqrt(kdict[str(np.setdiff1d([0,1,2],nu_axis)[0])]**2+kdict[str(np.setdiff1d([0,1,2],nu_axis)[1])]**2)
	if binning=='log': 
		kper = np.linspace(np.log10(np.abs(kp[kp!=0]).min()), np.log10(kp.max()), kbins[0]+1)
		kpar = np.linspace(np.log10(np.abs(kz[kz!=0]).min()), np.log10(kz.max()), kbins[1]+1)
		kp, kz  = np.log10(kp), np.log10(kz)
	elif binning=='linear':
		kper = np.linspace(np.abs(kp[kp!=0]).min(), kp.max(), kbins[0]+1)
		kpar = np.linspace(np.abs(kz[kz!=0]).min(), kz.max(), kbins[1]+1)
	k_width = kper[1]-kper[0], kpar[1]-kpar[0]
	kper = (kper[:-1]+kper[1:])/2.
	kpar = (kpar[:-1]+kpar[1:])/2.
	ps = np.zeros((kbins[0],kbins[1]))
	n_modes = np.zeros((kbins[0],kbins[1]))
	kp, kz, power = kp.flatten(), kz.flatten(), power.flatten()
	for i,a in enumerate(kper):
		for j,b in enumerate(kpar):
			arg = np.intersect1d(np.argwhere(np.abs(kp-a)<=k_width[0]/2.), np.argwhere(np.abs(kz-b)<=k_width[1]/2.))
			ps[i,j] = power[arg].sum()
			n_modes[i,j] = arg.size

	ps = ps/n_modes
	if binning=='log': kper, kpar = 10**kper, 10**kpar
	if return_modes: return ps, kper, kpar, n_modes
	return ps, kper, kpar


def dimensional_power(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False):
	ps, ks, n_modes = power_spect_1d(input_array, kbins=kbins, binning=binning, box_dims=box_dims, return_modes=1)
	if return_modes: return ps*ks**3/2/np.pi**2, ks, n_modes
	return ps*ks**3/2/np.pi**2, ks


def power_spect_mu(input_array, kbins=10, box_dims=244/.7, return_modes=False, mubins=10, binning='log', nu_axis=2):
	if type(binning)==str: binning = [binning, binning]
	power = power_spect_nd(input_array, box_dims, verbose=0)
	[kx,ky,kz], k = _get_k(input_array, box_dims)
	kdict = {}
	kdict['0'], kdict['1'], kdict['2'] = kx, ky, kz
	del kx, ky, kz
	kpar = kdict[str(nu_axis)]
	kper = np.sqrt(kdict[str(np.setdiff1d([0,1,2],nu_axis)[0])]**2+kdict[str(np.setdiff1d([0,1,2],nu_axis)[1])]**2)
	m = np.abs(kpar/k)

	if binning[0]=='log': 
		ks = np.linspace(np.log10(np.abs(k[k!=0]).min()), np.log10(k.max()), kbins+1)
		k  = np.log10(k)
	else: ks = np.linspace(np.abs(k[k!=0]).min(), k.max(), kbins+1)
	ks = (ks[:-1]+ks[1:])/2.
	k_width = ks[1]-ks[0]

	if binning[1]=='log': 
		m1 = m[np.isfinite(m)]
		mu = np.linspace(np.log10(np.abs(m1[m1!=0]).min()), np.log10(m1.max()), mubins+1)
		m  = np.log10(m)
	else: mu = np.linspace(0,1,mubins+1); 
	mu = (mu[:-1]+mu[1:])/2.
	m_width = mu[1]-mu[0]

	ps = np.zeros((kbins, mubins))
	n_modes = np.zeros((kbins, mubins))

	k, m, power = k.flatten(), m.flatten(), power.flatten()
	for i,a in enumerate(ks):
		for j,b in enumerate(mu):
			arg = np.intersect1d(np.argwhere(np.abs(k-a)<=k_width/2.), np.argwhere(np.abs(m[np.isfinite(m)]-b)<=m_width/2.))
			ps[i,j] = power[arg].sum()
			n_modes[i,j] = arg.size


	ps = ps/n_modes
	if binning[0]=='log': ks = 10**ks
	if binning[1]=='log': mu = 10**mu
	if return_modes: return ps, ks, mu, n_modes
	return ps, ks, mu


def plot_2d_power(ps, xticks, yticks, xlabel, ylabel, color_label=None):
	import matplotlib.pyplot as plt
	xticks, yticks = np.round(xticks, decimals=2), np.round(yticks, decimals=2)
	plt.imshow(ps, origin='lower')
	locs, labels = plt.yticks()
	new_labels = yticks[locs.astype(int)[1:-1]]
	plt.yticks(locs[1:-1], new_labels)
	plt.ylabel(ylabel)
	locs, labels = plt.xticks()
	new_labels = xticks[locs.astype(int)[1:-1]]
	plt.xticks(locs[1:-1], new_labels)
	plt.xlabel(xlabel)
	plt.colorbar(label=color_label)
	plt.show()



