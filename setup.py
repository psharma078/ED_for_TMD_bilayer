from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
extensions = [
    Extension("*", ["*.pyx"],include_dirs=[numpy.get_include()], extra_compile_args=['-O3', '-fopenmp'],
    extra_link_args=['-O3', '-fopenmp'])]
setup(name = "interaction", ext_modules = cythonize(extensions))
