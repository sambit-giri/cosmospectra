from numba import njit, prange
import numpy as np

def put_nearest(array, ref_list):
    fltn = np.array([array]) if type(array) in [int, float] else array.flatten()
    fltn = _replace(fltn, ref_list)
    return fltn.reshape(array.shape)

def put_sphere(data, centre, radius, label=1, periodic=True, refresh=False):
    assert data.ndim == 3
    array = np.zeros_like(data) if refresh else data.copy()
    array = _sphere(array, radius, centre[0], centre[1], centre[2], label, periodic)
    return array

@njit(parallel=True)
def _replace(fltn, ref_list):
    for i in prange(fltn.shape[0]):
        fltn[i] = ref_list[np.abs(ref_list-fltn[i]).argmin()]
    return fltn

@njit(parallel=True)
def _sphere(array, radius, cx, cy, cz, label, periodic):
    nx, ny, nz = array.shape
    if periodic: ix, iy, iz = -nx, -ny, -nz
    else: ix, iy, iz = 0, 0, 0
    rad2 = radius**2
    for i in prange(ix,nx):
        for j in prange(iy,ny):
            for k in prange(iz,nz):
                dist2 = (cx-i)**2+(cy-j)**2+(cz-k)**2
                if dist2<rad2: array[i,j,k] = label
    return array

def bispectrum_fast_equilateral(data, box_dims, s=None, dk=0.05, dlnk=None):
    nGridx, nGridy, nGridz = data.shape
    Mx, My, Mz = int(nGridx/2), int(nGridy/2), int(nGridz/2)
    kF = 2*np.pi/box_dims
    if s is None: 
        s  = round(dk/kF)
        dk = s*kF
        print('The k bin width is recalculated to be %.4f/Mpc.'%dk)
    dataft  = np.fft.fftn(data.astype('float64'))
    #dataft  = np.fft.fftshift(dataft)
    ns1, ns2, ns3 = np.arange(0,Mx,s)+s/2, np.arange(0,Mx,s)+s/2, np.arange(0,Mx,s)+s/2
    num = _bispectrum_fast_equilateral_num(dataft, ns1)
    den = 8*np.pi**2*s**3*ns1*ns1*ns1
    V   = box_dims**3
    return V**2/(nGridx*nGridy*nGridz)**9*num/den


@njit(parallel=True)
def _bispectrum_fast_equilateral_num(dataft, ns1):
    nGridx, nGridy, nGridz = dataft.shape
    Mx, My, Mz = int(nGridx/2), int(nGridy/2), int(nGridz/2)
    fltn = dataft[:Mx,:My,:Mz].flatten()
    N    = fltn.shape[0]
    num  = np.zeros_like(ns1)
    for m1 in prange(N):
        for m2 in prange(N):
            for m3 in prange(N):
                for i,n1 in enumerate(ns1):
                    print(m1,m2,m3,i)
                    if np.abs(m1-n1)<s/2 and np.abs(m2-n1)<s/2 and np.abs(m3-n1)<s/2: 
                        num[i] += fltn[m1]*fltn[m2]*fltn[m3]
    return num


@njit(parallel=True)
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


