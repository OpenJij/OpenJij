import os
import platform

from skbuild import setup
from setuptools.command.build_ext import build_ext

cmake_args = []
if platform.system() == 'Darwin':
    # disable macos openmp since addtional dependency is needed.
    if not {'True': True, 'False': False}[os.getenv('FORCE_USE_OMP', 'False')]:
        print("FORCE_USE_OMP=No")
        cmake_args += ['-DFORCE_USE_OMP=No']
    else:
        print("FORCE_USE_OMP=Yes")
        cmake_args += ['-DFORCE_USE_OMP=Yes']

setup(
    cmdclass={"build_ext": build_ext},
    cmake_args = cmake_args,
    cmake_install_dir='openjij',
    cmake_with_sdist=True,
)
