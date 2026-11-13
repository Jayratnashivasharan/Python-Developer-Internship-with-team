"""Build the optional Cython extension:  python setup.py build_ext --inplace
(the package works without it, using the numpy fallback)."""
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    ext = cythonize([Extension("gpualign.index._fmcore", ["gpualign/index/_fmcore.pyx"],
                               include_dirs=[np.get_include()],
                               extra_compile_args=["-O3"])],
                    compiler_directives={"language_level": "3"})
except ImportError:
    ext = []

setup(ext_modules=ext)
