from setuptools import setup
from Cython.Build import cythonize

setup(
    name="Barebones Dynamics Library",
    ext_modules=cythonize("decentralized/bbdynamicswrap.pyx"),
    zip_safe=False,
)
