import numpy as np
from scipy import stats
from astropy import units as u
from . import useful_speedup, useful
from .power_spect_fast import _get_k
import tqdm
from time import time

class SymmetricPolyspectrum:
    def __init__(self, box_dims, nGrid, dk=0.05):
        self.box_dims = box_dims
        self.nGrid    = nGrid
        self.Get_k(np.zeros((self.nGrid,self.nGrid,self.nGrid)), self.box_dims)
        self.Binned_k(dk=dk)
        self.data   = None

    def Get_k(self, data, box_dims):
        [kx,ky,kz],k = _get_k(data, box_dims)
        self.ks = {'kx': kx, 'ky': ky, 'kz': kz, 'k':k}

    def Binned_k(self, binned_k=None, dk=0.05):
        self.dk = dk
        if binned_k is None:
            bink = np.arange(self.ks['k'].min(), self.ks['k'].max(), self.dk)
            self.binned_k = bink[1:]/2.+bink[:-1]/2.
        else: self.binned_k = binned_k
        self.cube_k   = useful_speedup.put_nearest(self.ks['k'], self.binned_k)

    def Data(self, data=None, filename=None, file_reader=np.load):
        if data is None: data = file_reader(filename)
        if data.shape[0]!=self.nGrid:
            self.nGrid = data.shape[0]
            self.Get_k(np.zeros((self.nGrid,self.nGrid,self.nGrid)), self.box_dims)
            self.Binned_k(dk=dk)
        self.data   = data  
        self.boxvol = self.box_dims**3
        self.nPixel = self.nGrid**3
        self.pixelsize = self.boxvol/self.nPixel
        self.dataft  = _unnormalised_fftn(self.data, boxvol=None) #np.fft.fftshift(np.fft.fftn(self.data.astype('float64')))
        #self.dataft *= self.pixelsize

    def Calc_spec(self, binned_k=None, dk=0.05, order=3):
        assert self.data is not None
        if binned_k is not None: self.Binned_k(binned_k=binned_k, dk=dk)
        binned_N = self.binned_k.size
        Pks = np.zeros((binned_N))
        for p,k1 in enumerate(self.binned_k):
            Ifft1 = np.zeros_like(self.cube_k)
            Ifft1[np.abs(self.cube_k-k1)<self.dk/2] = 1
            dfft1 = self.dataft*Ifft1
            I1 = _unnormalised_ifftn(Ifft1, boxvol=None)
            d1 = _unnormalised_ifftn(dfft1, boxvol=None)
            d123 = np.sum(np.real(d1**order))
            I123 = np.sum(np.real(I1*order))
            pk   = d123/I123*(self.boxvol)**(order-1)/(self.nPixel)**order
            Pks[p] = pk
            #print(k1, (k1**3/(2*np.pi**2))*pk)
            print('%d / %d'%(p+1,binned_N))
        return {'k': self.binned_k, 'Pk': Pks}

    def Polyspec(self, data=None, order=3):
        if data is not None: self.Data(data=data)
        self.Calc_spec(order=3)
        return {'k': self.binned_k, 'spec': self.Bks}


class Powerspectrum:
    def __init__(self, box_dims, nGrid, dk=0.05, n_jobs=1, use_io=True):
        self.box_dims = box_dims
        self.nGrid    = nGrid
        self.Get_k(np.zeros((self.nGrid,self.nGrid,self.nGrid)), self.box_dims)
        self.Binned_k(dk=dk)
        self.data   = None
        self.n_jobs = n_jobs
        self.temp_file = None
        if use_io:
            self.temp_file = use_io if type(use_io)==str else 'temp_{}.pkl'.format(int(time()))
        
    def Get_k(self, data, box_dims):
        [kx,ky,kz],k = _get_k(data, box_dims)
        self.ks = {'kx': kx, 'ky': ky, 'kz': kz, 'k':k}

    def Binned_k(self, binned_k=None, dk=0.05):
        self.dk = dk
        if binned_k is None:
            bink = np.arange(self.ks['k'].min(), self.ks['k'].max(), self.dk)
            self.binned_k = bink[1:]/2.+bink[:-1]/2.
        else: self.binned_k = binned_k
        self.cube_k   = useful_speedup.put_nearest(self.ks['k'], self.binned_k)

    def Data(self, data=None, filename=None, file_reader=np.load):
        if data is None: data = file_reader(filename)
        if data.shape[0]!=self.nGrid:
            self.nGrid = data.shape[0]
            self.Get_k(np.zeros((self.nGrid,self.nGrid,self.nGrid)), self.box_dims)
            self.Binned_k(dk=dk)
        self.data   = data
        self.boxvol = self.box_dims**3
        self.nPixel = self.nGrid**3
        self.pixelsize = self.boxvol/self.nPixel
        self.dataft  = _unnormalised_fftn(self.data, boxvol=None) #np.fft.fftshift(np.fft.fftn(self.data.astype('float64')))
        #self.dataft *= self.pixelsize

    def Calc_Pk(self, binned_k=None, dk=0.05):
        assert self.data is not None
        if binned_k is not None: self.Binned_k(binned_k=binned_k, dk=dk)
        binned_N = self.binned_k.size
        # Pks = np.zeros((binned_N))
        # for p,k1 in enumerate(self.binned_k):
        #     Ifft1 = np.zeros_like(self.cube_k)
        #     Ifft1[np.abs(self.cube_k-k1)<self.dk/2] = 1
        #     dfft1 = self.dataft*Ifft1
        #     I1 = _unnormalised_ifftn(Ifft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(Ifft1))#/self.nPixel
        #     d1 = _unnormalised_ifftn(dfft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(dfft1))#/self.boxvol
        #     d123 = np.sum(np.real(d1*d1))
        #     I123 = np.sum(np.real(I1*I1))
        #     pk   = d123/I123*(self.boxvol)/(self.nPixel)**2
        #     Pks[p] = pk
        #     print(k1, (k1**3/(2*np.pi**2))*pk)
        #     print('%d / %d'%(p+1,binned_N))

        if self.temp_file is not None:
            temp_saved = {}
            temp_saved['binned_k'] = self.binned_k
            temp_saved['cube_k']   = self.cube_k
            temp_saved['dataft']   = self.dataft
            temp_saved['dk']       = dk
            pickle.dump(temp_saved, open(self.temp_file, 'wb'))
            
        def create_Ifft_io(p):
            temp_saved = pickle.load(open(self.temp_file, 'rb'))
            k1 = temp_saved['binned_k'][p]
            Ifft1 = np.zeros_like(temp_saved['cube_k'])
            Ifft1[np.abs(temp_saved['cube_k']-k1)<temp_saved['dk']/2] = 1
            temp_saved['{}'.format(int(p))] = Ifft1
            pickle.dump(temp_saved, open(self.temp_file, 'wb'))
            
        def temp_io(p):
            temp_saved = pickle.load(open(self.temp_file, 'rb'))
            I1 = _unnormalised_ifftn(Ifft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(Ifft1))#/self.nPixel
            d1 = _unnormalised_ifftn(dfft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(dfft1))#/self.boxvol
            d123 = np.sum(np.real(d1*d1))
            I123 = np.sum(np.real(I1*I1))
            pk   = d123/I123*(boxvol)/(nPixel)**2
            return pk
            
            
        def temp(p, binned_k, dk, cube_k, dataft, boxvol, nPixel):
            k1 = binned_k[p]
            Ifft1 = np.zeros_like(cube_k)
            Ifft1[np.abs(cube_k-k1)<dk/2] = 1
            dfft1 = dataft*Ifft1
            I1 = _unnormalised_ifftn(Ifft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(Ifft1))#/self.nPixel
            d1 = _unnormalised_ifftn(dfft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(dfft1))#/self.boxvol
            d123 = np.sum(np.real(d1*d1))
            I123 = np.sum(np.real(I1*I1))
            pk   = d123/I123*(boxvol)/(nPixel)**2
            return pk

        if self.temp_file is not None:
            for p in tqdm(range(binned_N)): create_Ifft_io(p)
            Pks = [temp_io(p) for p in tqdm(range(binned_N))]
        else:
            Pks = [temp(p, self.binned_k, self.dk, self.cube_k, self.dataft, self.boxvol, self.nPixel) for p in tqdm(range(binned_N))]

        Pks = np.array(Pks)
        return {'k': self.binned_k, 'Pk': Pks}

    def Powerspec(self, data=None):
        if data is not None: self.Data(data=data)
        self.Calc_Pk()
        return {'k': self.binned_k, 'Bk': self.Bks}

class Bispectrum:
    def __init__(self, box_dims, nGrid, dk=0.05):
        self.box_dims = box_dims
        self.nGrid    = nGrid
        self.Get_k(np.zeros((self.nGrid,self.nGrid,self.nGrid)), self.box_dims)
        self.Binned_k(dk=dk)
        self.data   = None

    def Get_k(self, data, box_dims):
        [kx,ky,kz],k = _get_k(data, box_dims)
        self.ks = {'kx': kx, 'ky': ky, 'kz': kz, 'k':k}

    def Binned_k(self, binned_k=None, dk=0.05):
        self.dk = dk
        if binned_k is None:
            bink = np.arange(self.ks['k'].min(), self.ks['k'].max(), self.dk)
            self.binned_k = bink[1:]/2.+bink[:-1]/2.
        else: self.binned_k = binned_k
        self.cube_k   = useful_speedup.put_nearest(self.ks['k'], self.binned_k)

    def Data(self, data=None, filename=None, file_reader=np.load):
        if data is None: data = file_reader(filename)
        if data.shape[0]!=self.nGrid:
            self.nGrid = data.shape[0]
            self.Get_k(np.zeros((self.nGrid,self.nGrid,self.nGrid)), self.box_dims)
            self.Binned_k(dk=dk)
        self.data   = data	
        self.boxvol = self.box_dims**3
        self.nPixel = self.nGrid**3
        self.pixelsize = self.boxvol/self.nPixel
        self.dataft  = _unnormalised_fftn(self.data, boxvol=None) #np.fft.fftshift(np.fft.fftn(self.data.astype('float64')))
        #self.dataft *= self.pixelsize

    def Calc_Bk_full(self):
        assert self.data is not None
        binned_N = self.binned_k.size
        #kF = 2*np.pi/box_dims
        self.Bks = np.zeros((binned_N,binned_N,binned_N))
        for p,k1 in enumerate(self.binned_k):
            Ifft1 = np.zeros_like(self.cube_k)
            Ifft1[np.abs(self.cube_k-k1)<self.dk/2] = 1
            dfft1 = self.dataft*Ifft1
            I1 = np.fft.ifftn(np.fft.fftshift(Ifft1))
            d1 = np.fft.ifftn(np.fft.fftshift(dfft1))
            for q,k2 in enumerate(self.binned_k):
                Ifft2 = np.zeros_like(self.cube_k)
                Ifft2[np.abs(self.cube_k-k2)<self.dk/2] = 1
                dfft2 = self.dataft*Ifft2
                I2 = np.fft.ifftn(np.fft.fftshift(Ifft2))
                d2 = np.fft.ifftn(np.fft.fftshift(dfft2))
                for r,k3 in enumerate(self.binned_k):
                    Ifft3 = np.zeros_like(self.cube_k)
                    Ifft3[np.abs(self.cube_k-k3)<self.dk/2] = 1
                    dfft3 = self.dataft*Ifft3
                    I3 = np.fft.ifftn(np.fft.fftshift(Ifft3))
                    d3 = np.fft.ifftn(np.fft.fftshift(dfft3))
                    
                    d123 = np.real(d1*d2*d3)
                    I123 = np.real(I1*I2*I3)
                    bk = np.sum(d123)/np.sum(I123)
                    self.Bks[p,q,r] = bk
                    count = p*binned_N*binned_N+q*binned_N+r+1
                    print(bk)
                    print('%d / %d'%(count,binned_N**3))

    def Calc_Bk_equilateral(self, binned_k=None, dk=0.05):
        assert self.data is not None
        if binned_k is not None: self.Binned_k(binned_k=binned_k, dk=dk)
        binned_N = self.binned_k.size
        Bks = np.zeros((binned_N))
        tstart = time()
        for p,k1 in enumerate(self.binned_k):
            Ifft1 = np.zeros_like(self.cube_k)
            Ifft1[np.abs(self.cube_k-k1)<self.dk/2] = 1
            dfft1 = self.dataft*Ifft1
            I1 = _unnormalised_ifftn(Ifft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(Ifft1))#/self.nPixel
            d1 = _unnormalised_ifftn(dfft1, boxvol=None) #np.fft.ifftn(np.fft.fftshift(dfft1))#/self.boxvol
            d123 = np.sum(np.real(d1*d1*d1))
            I123 = np.sum(np.real(I1*I1*I1)) #8*np.pi**2*k1*3*self.dk**3/kF**6 
            bk   = d123/I123*(self.boxvol)**2/(self.nPixel)**3
            Bks[p] = bk
            count = p+1
            #print(k1, (k1**6/(2*np.pi**2)**2)*bk)
            #print('%d / %d'%(count,binned_N))
            useful.loading_verbose('%d / %d | Elapsed time %d s'%(count,binned_N,int(time()-tstart)))
        print('...done')
        return {'k': self.binned_k, 'Bk': Bks}

    def Calc_Bk_equilateral_foreman(self, binned_k=None, dk=0.05):
        assert self.data is not None
        if binned_k is not None: self.Binned_k(binned_k=binned_k, dk=dk)
        binned_N = self.binned_k.size
        Bks = np.zeros((binned_N))
        for p,k1 in enumerate(self.binned_k):
            Ifft1 = np.zeros_like(self.cube_k)
            Ifft1[np.abs(self.cube_k-k1)<self.dk/2] = 1
            dfft1 = self.dataft*Ifft1
            #I1 = np.fft.ifftn(np.fft.fftshift(Ifft1))#/self.boxvol
            d1 = np.fft.ifftn(np.fft.fftshift(dfft1))*self.pixelsize#/self.boxvol
            d123 = np.real(d1*d1*d1)
            #I123 = np.real(I1*I1*I1)
            V_Del = np.sum(Ifft1)
            bk = np.sum(d123)/V_Del
            Bks[p] = bk
            count = p+1
            print((k1**6/(2*np.pi**2)**2)*bk)
            print('%d / %d'%(count,binned_N))
        return {'k': self.binned_k, 'Bk': Bks}

    def Bispec(self, data=None):
        if data is not None: self.Data(data=data)
        self.Calc_Bk()
        return {'k': self.binned_k, 'Bk': self.Bks}

"""
def bispectrum_fast_equilateral(data, box_dims, s=None, dk=0.05, dlnk=None):
    nGridx, nGridy, nGridz = data.shape
    Mx, My, Mz = nGridx/2, nGridy/2, nGridz/2
    kF = 2*np.pi/box_dims
    if s is None: 
        s  = round(dk/kF)
        dk = s*kF
        print('The k bin width is recalculated to be %.4f/Mpc.'%dk)
    dataft  = np.fft.fftn(data.astype('float64'))
    #dataft  = np.fft.fftshift(dataft)
    ns1, ns2, ns3 = np.arange(0,Mx,s)+s/2, np.arange(0,Mx,s)+s/2, np.arange(0,Mx,s)+s/2
    
"""

def round_nearest_float(n, num=0.5):
    return np.round(n/num)*num

def put_nearest(array, ref_list):
    fltn = np.array([array]) if type(array) in [int, float] else array.flatten()
    for i,ft in enumerate(fltn):
        fltn[i] = ref_list[np.abs(ref_list-ft).argmin()]
    return fltn.reshape(array.shape)

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

def _unnormalised_fftn(data, boxvol=None, box_dims=None):
    dataft  = np.fft.fftshift(np.fft.fftn(data.astype('float64')))
    Npix    = dataft.size
    if box_dims is not None: boxvol = box_dims**3
    if boxvol is not None: dataft *= boxvol/Npix
    return dataft

def _unnormalised_ifftn(dataft, boxvol=None, box_dims=None):
    data = np.fft.ifftn(np.fft.fftshift(dataft))
    Npix = data.size
    if box_dims is not None: boxvol = box_dims**3
    if boxvol is not None: data *= Npix/boxvol
    return data


def bisp_equilateral_fast(data, box_dims, dk=0.05, kbins=None):
    tstart = time()
    print('Preparing estimator...')
    bisp = Bispectrum(box_dims, data.shape[0], dk=dk)
    bisp.Data(data)
    print('...done')
    print('Estimating...')
    equi = bisp.Calc_Bk_equilateral(kbins)
    print('Runtime: {:.2f} mins'.format((time()-tstart)/60))
    return equi['Bk'], equi['k']
