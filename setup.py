from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from distutils.errors import CompileError
from warnings import warn
import os
import sys
from glob import glob

# generate list of .c extension modules based on the .pyx files, which are
# included in the pypi package (but aren't included in the git repo)
extension_pathspec = os.path.join('pylds','*.pyx')  # only searches pylds/*.pyx!
paths = [os.path.splitext(fp)[0] for fp in glob(extension_pathspec)]
names = ['.'.join(os.path.split(p)) for p in paths]
ext_modules = [
    Extension(name, sources=[path + '.c'], include_dirs=[os.path.join('deps')])
    for name, path in zip(names,paths)]

# alternatively, use cython to generate extension modules if we can import it
# (which is required if we're installing from the git repo)
try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

if use_cython:
    from Cython.Build import cythonize
    try:
        ext_modules = cythonize('**/*.pyx')  # recursive globbing!
    except:
        warn('Failed to generate extension module code from Cython files')
        sys.exit(1)

# if we run the dist command, regenerate the sources from cython
class sdist(_sdist):
    def run(self):
        from Cython.Build import cythonize
        cythonize(os.path.join('pylds','*.pyx'))
        _sdist.run(self)

# the final extension module build step should have numpy headers available
class build_ext(_build_ext):
    # see http://stackoverflow.com/q/19919905 for explanation
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy as np
        self.include_dirs.append(np.get_include())

setup(
    name='pylds',
    version='0.0.5',
    description="Learning and inference for linear dynamical systems"
    "with fast Cython and BLAS/LAPACK implementations",
    author='Matthew James Johnson and Scott W Linderman',
    author_email='mattjj@csail.mit.edu',
    license="MIT",
    url='https://github.com/mattjj/pylds',
    packages=['pylds'],
    install_requires=[
        'numpy>=1.9.3', 'scipy>=0.16', 'matplotlib',
        'pybasicbayes', 'autograd', 'pypolyagamma>=1.1'],
    setup_requires=['future'],
    ext_modules=ext_modules,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
    keywords=[
        'lds', 'linear dynamical system', 'kalman filter', 'kalman',
        'kalman smoother', 'rts smoother'],
    platforms="ALL",
    cmdclass={'build_ext': build_ext, 'sdist': sdist})
