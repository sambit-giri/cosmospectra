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
