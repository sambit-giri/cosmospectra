import numpy as np
import sys

def loading_verbose(string):
    msg = string if isinstance(string, (str)) else str(string)
    sys.stdout.write('\r'+msg)
    sys.stdout.flush()

def put_sphere(data, centre, radius, label=1, periodic=True, refresh=False):
    assert data.ndim == 3
    nx, ny, nz = data.shape
    array = np.zeros_like(data) if refresh else data.copy()
    aw  = np.argwhere(np.isfinite(array))
    RR  = ((aw[:,0]-centre[0])**2 + (aw[:,1]-centre[1])**2 + (aw[:,2]-centre[2])**2).reshape(array.shape)
    array[RR<=radius**2] = label
    if periodic: 
        RR2 = ((aw[:,0]+nx-centre[0])**2 + (aw[:,1]+ny-centre[1])**2 + (aw[:,2]+nz-centre[2])**2).reshape(array.shape)
        array[RR2<=radius**2] = label
    return array

