import os
import platform
import re
import subprocess
import sys
import sysconfig
from distutils.version import LooseVersion
from importlib.util import find_spec
from multiprocessing import cpu_count

from setuptools import Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

# Package meta-data.
NAME = 'openjij'
DESCRIPTION = 'Framework for the Ising model and QUBO'
EMAIL = 'openjij@j-ij.com'
AUTHOR = 'Jij Inc.'

if platform.system() == "Windows":
    from setuptools import setup
elif find_spec('skbuild'):
    from skbuild import setup
elif os.getenv('NOT_USE_SKBUILD'):
    from setuptools import setup
elif os.getenv('READTHEDOCS'):
    from skbuild import setup
else:
    from setuptools import setup

CPU_COUNT = "-j" + str(cpu_count() + 1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.12.2':
                raise RuntimeError("CMake >= 3.12.2 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_kwargs = ['-DUSE_TEST=Yes',
                        '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                        #'-DCMAKE_VERBOSE_MAKEFILE=ON',
                        # '-DCMAKE_CUDA_FLAGS= -arch=sm_60',
                        '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_kwargs = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_kwargs += [
                    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_kwargs += ['-A', 'x64']
            build_kwargs += ['--', '/m']
        elif platform.system() == 'Darwin':
            cmake_kwargs += ['-DCMAKE_BUILD_TYPE=' + cfg]
            #cmake_kwargs += ['-DCMAKE_MAKE_PROGRAM=' + '\"make ' + str(CPU_COUNT) + '\"']
            build_kwargs += ['--',   str(CPU_COUNT)]
        else:
            cmake_kwargs += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_kwargs += ['--', str(CPU_COUNT)]

        # disable macos openmp since addtional dependency is needed.
        if platform.system() != "Windows" and platform.system() != "Linux":
            cmake_kwargs += ['-DUSE_OMP=No']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_kwargs, cwd=self.build_temp, env=env)
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', 'python'] + build_kwargs, cwd=self.build_temp)


class GoogleTestCommand(TestCommand):
    """
    A custom test runner to execute both Python unittest tests and C++ Google Tests.
    """

    def distutils_dir_name(self, dname):
        """Returns the name of a distutils build directory"""
        dir_name = "{dirname}.{platform}-{version[0]}.{version[1]}"
        return dir_name.format(dirname=dname,
                               platform=sysconfig.get_platform(),
                               version=sys.version_info)

    def run(self):
        # Run Python tests
        super(GoogleTestCommand, self).run()
        print("\nPython tests complete, now running C++ tests...\n")
        # Run catch tests
        print(os.path.join('build/', self.distutils_dir_name('lib')))
        subprocess.call(['make cxxjij_test'],
                        cwd=os.path.join('build',
                                         self.distutils_dir_name('temp')),
                        shell=True)
        subprocess.call(['./tests/cxxjij_test'],
                        cwd=os.path.join('build',
                                         self.distutils_dir_name('temp')),
                        shell=True)


class PyTestCommand(TestCommand):
    def run(self):
        super().run()

install_requires = [
        'numpy>=1.17.3, <=1.20.0',
        'dimod>=0.9.14, <=0.10.0',
        'jij-cimod >= 1.1.0',
        'scipy >= 1.6.2',
        'requests >= 2.25.1'
        ]

setup_requires = [
        'numpy>=1.17.3, <=1.20.0',
        'scikit-build> =0.11.1',
        'wheel >=0.36.2',
        'Cython >=0.29.21'
        'cmake>=3.18.4',
        'setuptools-git-versioning',
        'setuptools',
        ]
setup(
    name=NAME,
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
    author=AUTHOR,
    author_email='openjij@j-ij.com',
    url='https://openjij.github.io/OpenJij/',
    description='Framework for the Ising model and QUBO',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
  　install_requires=install_requires,
    setup_requires=setup_requires,
    ext_modules=[CMakeExtension('cxxjij')],
    cmdclass=dict(build_ext=CMakeBuild, test=GoogleTestCommand,
                  pytest=PyTestCommand),
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    license='Apache License 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
    zip_safe=False
)
