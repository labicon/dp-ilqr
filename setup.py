from setuptools import setup
from Cython.Build import cythonize

setup(
    name="Barebones Dynamics Library",
    ext_modules=cythonize("dpilqr/bbdynamicswrap.pyx"),
    zip_safe=False,
)
