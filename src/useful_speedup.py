from numba import njit, prange
import numpy as np

def put_nearest(array, ref_list):
    fltn = np.array([array]) if type(array) in [int, float] else array.flatten()
    fltn = _replace(fltn, ref_list)
    return fltn.reshape(array.shape)

@njit(parallel=True)
def _replace(fltn, ref_list):
    for i in prange(fltn.shape[0]):
        fltn[i] = ref_list[np.abs(ref_list-fltn[i]).argmin()]
    return fltn

