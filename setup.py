from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('Portfolio/CythonizedCPO/*.pyx', language_level="3"))
