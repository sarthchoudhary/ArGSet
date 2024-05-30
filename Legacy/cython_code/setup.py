from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("/home/sarthak/my_projects/argset/scripts/cython_code/fingerplot_debug.pyx", 
                            annotate=True, compiler_directives={'language_level' : "3"}),
    include_dirs=[numpy.get_include()]
)
