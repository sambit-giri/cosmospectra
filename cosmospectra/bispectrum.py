import numpy as np
from scipy import stats
from astropy import units as u

class Bispectrum:
    def __init__(self, box_dims, nGrid):
        self.box_dims = box_dims
        self.nGrid    = nGrid
        self.get_k(np.zeros((self.nGrid,self.nGrid,self.nGrid)), self.box_dims)

    def get_k(self, data, box_dims):
        [kx,ky,kz],k = _get_k(data, box_dims)
        self.ks = {'kx': kx, 'ky': ky, 'kz': kz, 'k':k}

    def read_data(self, data=None, filename=None, file_reader=np.load):
        if data is None: data = file_reader(filename)
        self.data   = data
        self.dataft = np.fft.fftshift(np.fft.fftn(data.astype('float64'))))

    def bispect(self, k1, k2, k3):
        


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

def _get_k(input_array, box_dims):
	'''
	Get the k values for input array with given dimensions.
	Return k components and magnitudes.
	For internal use.
	'''
	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
	dim = len(input_array.shape)
	if dim == 1:
		x = np.arange(len(input_array))
		center = x.max()/2.
		kx = 2.*np.pi*(x-center)/box_dims[0]
		return [kx], kx
	elif dim == 2:
		x,y = np.indices(input_array.shape, dtype='int32')
		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		k = np.sqrt(kx**2 + ky**2)
		return [kx, ky], k
	elif dim == 3:
		x,y,z = np.indices(input_array.shape, dtype='int32')
		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2, \
						(z.max()-z.min())/2])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		kz = 2.*np.pi * (z-center[2])/box_dims[2]

		k = np.sqrt(kx**2 + ky**2 + kz**2 )
		return [kx,ky,kz], k
