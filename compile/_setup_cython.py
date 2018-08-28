from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os.path as osp
import numpy

import os
os.chdir(osp.join(osp.dirname(__file__), '..'))
resolve = lambda *parts: osp.join('src', *parts)  # noqa

extensions = [
    Extension("src.basics.ours._cython", [resolve('basics', 'ours', '_cython.pyx')],
              include_dirs=[numpy.get_include()],
              # libraries=[...],
              # library_dirs=[...]),
              ),
    # # Everything but primes.pyx is included here.
    # Extension("*", ["*.pyx"],
    #           include_dirs=[...],
    #           libraries=[...],
    #           library_dirs=[...]),
]

setup(name='xxx', ext_modules=cythonize(extensions))
