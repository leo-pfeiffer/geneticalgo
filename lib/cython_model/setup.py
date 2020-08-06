from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     name='GA app',
#     ext_modules=cythonize("supplychain.pyx"),
#     include_dirs=[numpy.get_include()],
#     zip_safe=False,
# )

setup(
    name='GA app',
    ext_modules=[Extension("supplychain", ["supplychain.c"],
                           include_dirs=[numpy.get_include()]),
                 ],
    zip_safe=False,
)
