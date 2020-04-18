import numpy as np
import cosmospectra as cs
from cosmospectra import toymodel

randsphere = toymodel.RandomSpheres(
    nGrid=300,
    allow_overlap=True,
    background=0,
    label=1,
)

