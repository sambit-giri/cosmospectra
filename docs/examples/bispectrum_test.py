import numpy as np
import cosmospectra as cs
from cosmospectra import toymodel, bispectrum

box_dims, nGrid = 500/.7, 300
rMpc = 10
rPix = round(rMpc*box_dims/nGrid)

randsphere = toymodel.RandomSpheres(
    nGrid=nGrid,
    allow_overlap=True,
    background=0,
    label=1,
    Rs=rPix,
)
out = randsphere.GetCube_FillingFraction(0.01)

bisp = bispectrum.Bispectrum(box_dims, nGrid)
bisp.Data(out['data'])

equi = bisp.Calc_Bk_equilateral()


