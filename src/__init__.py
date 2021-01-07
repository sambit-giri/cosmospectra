'''
CosmoSpectra package is meant for Fourier analysis of cosmological simulations.

One can also get documentation for all routines directory from
the interpreter using Python's built-in help() function.
For example:
>>> import cosmospectra as cs
>>> help(cs.power_spect_1d)

Python's built-in dir() function can be used to find the list of routines in the
package.
For example:
>>> dir(cs)

'''

from .power_spect_fast import *
from .power_spect_response import *
from . import toymodel
from .bispectrum import *

#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')
