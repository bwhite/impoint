from distutils.core import setup
import re
from distutils.extension import Extension
import numpy as np


def get_cython_version():
    """
    Returns:
        Version as a pair of ints (major, minor)

    Raises:
        ImportError: Can't load cython or find version
    """
    import Cython.Compiler.Main
    match = re.search('^([0-9]+)\.([0-9]+)',
                      Cython.Compiler.Main.Version.version)
    try:
        return map(int, match.groups())
    except AttributeError:
        raise ImportError

# Only use Cython if it is available, else just use the pre-generated files
try:
    cython_version = get_cython_version()
    # Requires Cython version 0.13 and up
    if cython_version[0] == 0 and cython_version[1] < 13:
        raise ImportError
    from Cython.Distutils import build_ext
    source_ext = '.pyx'
    cmdclass = {'build_ext': build_ext}
except ImportError:
    source_ext = '.c'
    cmdclass = {}

ext_modules = [Extension("_impoint",
                         ["impoint/impoint" + source_ext],
                         extra_compile_args=['-I', np.get_include()],
                         include_dirs=['impoint']),
               Extension("_impoint_surf",
                         ["impoint/_surf/surf" + source_ext,
                          'impoint/_surf/surf_feature.cpp', 'impoint/_surf/SurfDetect.cpp',
                          'impoint/_surf/SurfDescribe.cpp', 'impoint/_surf/SurfPoint.cpp',
                          'impoint/_surf/SurfMatch.cpp', 'impoint/_surf/MatchPair.cpp',
                          'impoint/_surf/integral_image/IntegralImage.cpp'],
                         extra_compile_args=['-I', np.get_include()],
                         include_dirs=['impoint'])]

setup(name='impoint',
      cmdclass=cmdclass,
      version='.01',
      packages=['impoint'],
      ext_modules=ext_modules)
