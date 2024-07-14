## python setup.py build_ext --inplace

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize(["cyPoisson_sim.pyx"]),
      include_dirs = [np.get_include()])
