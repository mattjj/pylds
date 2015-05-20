from distutils.core import setup
import numpy as np
from Cython.Build import cythonize

setup(
    name='pylds',
    version='0.0.1',
    author='Matthew James Johnson',
    author_email='mattjj@csail.mit.edu',
    url='https://github.com/mattjj/pylds',
    packages=['pylds'],
    install_requires=[
        'Cython >= 0.20.1', 'numpy', 'scipy', 'matplotlib',
        'pybasicbayes'],
    ext_modules=cythonize('pylds/**/*.pyx'),
    include_dirs=[np.get_include(),],
)

