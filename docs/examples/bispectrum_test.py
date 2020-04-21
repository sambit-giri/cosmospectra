import numpy as np
import cosmospectra as cs
from cosmospectra import toymodel

randsphere = toymodel.RandomSpheres(
    nGrid=100,
    allow_overlap=True,
    background=0,
    label=1,
    Rs=10,
)
out = randsphere.GetCube_FillingFraction(0.01)

