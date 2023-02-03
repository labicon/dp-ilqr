from setuptools import setup
from Cython.Build import cythonize

setup(
    name="bbdynamics",
    ext_modules=cythonize(
        "dpilqr/bbdynamicswrap.pyx",
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
