'''
Created on 05 October 2019
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup


setup(name='cosmospectra',
      version='0.1',
      author='Sambit Giri',
      author_email='sambit.giri@astro.su.se',
      package_dir = {'cosmospectra' : 'src'},
      packages=['cosmospectra'],
      package_data={'share':['*'],},
      install_requires=['numpy','scipy','astropy', 'numba'],
      url="https://github.com/sambit-giri/cosmospectra.git",
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
)
