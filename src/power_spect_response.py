import numpy as np
from power_spect_fast import *

def integrated_bispectrum_cross(cube1, cube2, Ncuts=4, kbins=200, box_dims=None, binning='log', normalize=False):
	assert cube1.shape == cube2.shape
	assert cube1.shape[0]%Ncuts==0 and cube1.shape[1]%Ncuts==0 and cube1.shape[2]%Ncuts==0
	Lx,Ly,Lz = cube1.shape[0]/Ncuts,cube1.shape[1]/Ncuts,cube1.shape[2]/Ncuts
	rLs = [[Lx/2.+i*Lx,Ly/2.+j*Ly,Lz/2.+k*Lz] for i in range(Ncuts) for j in range(Ncuts) for k in range(Ncuts)]
	B_k   = np.zeros(kbins, dtype=np.float64)
	P_k   = np.zeros(kbins, dtype=np.float64)
	sig2  = 0
	n_box = Ncuts**3
	V_L   = (Lx*Ly*Lz)
	for i in range(n_box):
		w1 = _W_L(cube1, rLs[i], [Lx,Ly,Lz])
		w2 = _W_L(cube2, rLs[i], [Lx,Ly,Lz])
		c1 = cube1 * w1
		c2 = cube2 * w2
		pk, ks = power_spect_1d(c1, kbins=kbins, box_dims=box_dims, binning=binning)
		d_mean = c2.sum(dtype=np.float64)/V_L
		B_k   += pk*d_mean
		P_k   += pk
		sig2  += (d_mean)**2   #c2.var(dtype=np.float64)
		print(100*(i+1)/n_box, "%")
	B_k  = B_k/n_box
	P_k  = P_k/n_box
	sig2 = sig2/n_box
	if normalize: return B_k/P_k/sig2, ks
	return B_k, ks

def _W_L(array, rL, L):
	assert array.ndim == np.array(rL).size
	out = np.zeros(array.shape)
	if np.array(L).size==1:out[(rL[0]-L/2):(rL[0]+L/2),(rL[1]-L/2):(rL[1]+L/2),(rL[2]-L/2):(rL[2]+L/2)] = 1
	else:out[(rL[0]-L[0]/2):(rL[0]+L[0]/2),(rL[1]-L[1]/2):(rL[1]+L[1]/2),(rL[2]-L[2]/2):(rL[2]+L[2]/2)] = 1
	return out

