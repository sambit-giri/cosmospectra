import numpy as np
import cosmospectra as cs
from cosmospectra import toymodel, bispectrum

box_dims, nGrid = 600., 300
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

ks   = 10**np.linspace(-1.2,0.3,50)
equi = bisp.Calc_Bk_equilateral(ks)

plt.plot(equi['k'], equi['Bk']*(equi['k']**3/2/np.pi**2)**2)
plt.xscale('log')
