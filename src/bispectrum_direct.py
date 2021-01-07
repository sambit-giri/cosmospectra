import numpy as np
from scipy import stats
from astropy import units as u
from .power_spect_fast import _get_k, _get_kF, _get_nk
from time import time

def bisp_equilateral(input_array, box_dims, verbose=True, kbins=20):
	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
	if verbose: print( 'Calculating bispectrum...')
	ft = np.fft.fftshift(np.fft.fftn(input_array.astype('float64')))
	ft_real = np.real(ft)

	boxvol = np.product(box_dims)
	#print(boxvol)
	pixelsize = boxvol/(np.product(input_array.shape)) #H^3 as per Jeong (2010)
	fctr = pixelsize**3/boxvol
	
	[kx,ky,kz],k  = _get_k(input_array, box_dims)
	kFx, kFy, kFz = _get_kF(box_dims)
	[mx,my,mz],m  = _get_nk(input_array, box_dims)

	n_edge = np.linspace(0,min([mx.max(), my.max(), mz.max()]), kbins+1)
	n_high, n_low = n_edge[1:], n_edge[:-1]
	ns = n_high/2. + n_low/2.
	ks = ns*kFx

	Bm, Tri = [], []
	msum_cond = 1

	for ni,nl,nh in zip(ns,n_low,n_high):
		print(nl,ni,nh)
		tstart = time()

		cond1 = np.argwhere(np.abs(m-ni)<0.5)
		b_unnorm, n_tri = _bisp_equilateral_direct(mx, my, mz, cond1, msum_cond)

		# b_unnorm = 0
		# n_tri    = 0

		# for i1 in cond1:
		# 	for i2 in cond1:
		# 		for i3 in cond1:
		# 			m1 = np.array([mx[i1[0],i1[1],i1[2]],my[i1[0],i1[1],i1[2]],mz[i1[0],i1[1],i1[2]]])
		# 			m2 = np.array([mx[i2[0],i2[1],i2[2]],my[i2[0],i2[1],i2[2]],mz[i2[0],i2[1],i2[2]]])
		# 			m3 = np.array([mx[i3[0],i3[1],i3[2]],my[i3[0],i3[1],i3[2]],mz[i3[0],i3[1],i3[2]]])
		# 			m_sum = m1+m2+m3
		# 			cond2 = np.all(np.abs(m_sum)<msum_cond)
		# 			if cond2: 
		# 				#print(m_sum)
		# 				b_unnorm += ft_real[i1[0],i1[1],i1[2]]*ft_real[i1[0],i1[1],i1[2]]*ft_real[i1[0],i1[1],i1[2]]
		# 				n_tri += 1

		tend = time()
		ki, Bi = ni*kFx, fctr*b_unnorm/n_tri
		if verbose: 
			print('Total number of triangle: {0:d}'.format(n_tri))
			print('k, B(k), (k^6/(2pi^2)^2)B(k) = {0:.5f}, {1:.5f}, {2:.5f}'.format(ki,Bi,Bi*(ki**3/2/np.pi**2)**2))
			print('Time taken: {0:.3f} minutes.'.format((tend-tstart)/60))
		Bm.append(Bi)
		Tri.append(n_tri)

	return np.array(Bm), ks, np.array(Tri)

def bisp_equilateral_mc(input_array, box_dims, verbose=True, kbins=20, n_samples=100):
	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
	if verbose: print( 'Calculating bispectrum...')
	ft = np.fft.fftshift(np.fft.fftn(input_array.astype('float64')))
	ft_real = np.real(ft)

	boxvol = np.product(box_dims)
	#print(boxvol)
	pixelsize = boxvol/(np.product(input_array.shape)) #H^3 as per Jeong (2010)
	fctr = pixelsize**3/boxvol
	
	[kx,ky,kz],k  = _get_k(input_array, box_dims)
	kFx, kFy, kFz = _get_kF(box_dims)
	[mx,my,mz],m  = _get_nk(input_array, box_dims)

	n_edge = np.linspace(0,min([mx.max(), my.max(), mz.max()]), kbins+1)
	n_high, n_low = n_edge[1:], n_edge[:-1]
	ns = n_high/2. + n_low/2.
	ks = ns*kFx

	Bm, Bm_err, Tri = [], [], []
	msum_cond = 1

	for ni,nl,nh in zip(ns,n_low,n_high):
		print(nl,ni,nh)
		tstart = time()

		cond1 = np.argwhere(np.abs(m-ni)<0.5)
		#b_unnorm, n_tri = _bisp_equilateral_direct(mx, my, mz, cond1, msum_cond)

		b_unnorms = []
		n_tri     = 0

		while n_tri<n_samples:
			i1 = cond1[np.random.randint(cond1.shape[0]),:]
			i2 = cond1[np.random.randint(cond1.shape[0]),:]
			i3 = cond1[np.random.randint(cond1.shape[0]),:]

			m1 = np.array([mx[i1[0],i1[1],i1[2]],my[i1[0],i1[1],i1[2]],mz[i1[0],i1[1],i1[2]]])
			m2 = np.array([mx[i2[0],i2[1],i2[2]],my[i2[0],i2[1],i2[2]],mz[i2[0],i2[1],i2[2]]])
			m3 = np.array([mx[i3[0],i3[1],i3[2]],my[i3[0],i3[1],i3[2]],mz[i3[0],i3[1],i3[2]]])
			m_sum = m1+m2+m3
			
			cond2 = np.all(np.abs(m_sum)<msum_cond)
			if cond2: 
				#print(m_sum)
				b_unnorms.append(ft_real[i1[0],i1[1],i1[2]]*ft_real[i1[0],i1[1],i1[2]]*ft_real[i1[0],i1[1],i1[2]])
				n_tri += 1
				print(n_tri)

		tend = time()

		b_unnorms = np.array(b_unnorms)
		b_unnorm, b_unnorm_std = b_unnorms.sum(), b_unnorms.std()

		ki, Bi, Bi_err = ni*kFx, fctr*b_unnorm/n_tri, fctr*b_unnorm_std/n_tri
		if verbose: 
			print('Total number of triangle: {0:d}'.format(n_tri))
			print('k, B(k), (k^6/(2pi^2)^2)B(k) = {0:.5f}, {1:.5f}, {2:.5f}'.format(ki,Bi,Bi*(ki**3/2/np.pi**2)**2))
			print('Time taken: {0:.3f} minutes.'.format((tend-tstart)/60))
		Bm.append(Bi)
		Bm_err.append(Bi_err)
		Tri.append(n_tri)

	return np.array(Bm), ks, np.array(Tri), np.array(Bm_err)


def _bisp_equilateral_direct(ft_real, mx, my, mz, cond1, msum_cond=1):
    b_unnorm = 0
    n_tri    = 0

    for i1 in cond1:
        for i2 in cond1:
            for i3 in cond1:
                m1 = np.array([mx[i1[0],i1[1],i1[2]],my[i1[0],i1[1],i1[2]],mz[i1[0],i1[1],i1[2]]])
                m2 = np.array([mx[i2[0],i2[1],i2[2]],my[i2[0],i2[1],i2[2]],mz[i2[0],i2[1],i2[2]]])
                m3 = np.array([mx[i3[0],i3[1],i3[2]],my[i3[0],i3[1],i3[2]],mz[i3[0],i3[1],i3[2]]])
                m_sum = m1+m2+m3
                cond2 = np.all(np.abs(m_sum)<msum_cond)
                if cond2: 
                    #print(m_sum)
                    b_unnorm += ft_real[i1[0],i1[1],i1[2]]*ft_real[i1[0],i1[1],i1[2]]*ft_real[i1[0],i1[1],i1[2]]
                    n_tri += 1

    return b_unnorm, n_tri   



