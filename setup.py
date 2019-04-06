import os
import re
import sys
import platform
import sysconfig
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from distutils.version import LooseVersion


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
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print(ext, ext.name)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      #'-DCMAKE_VERBOSE_MAKEFILE=ON',
                      #'-DCMAKE_CUDA_FLAGS= -arch=sm_30 ',
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'python'] + build_args, cwd=self.build_temp)

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

setup(
    name='openjij',
    version='0.0.1',
    author='Jij Inc.',
    author_email='openjij@j-ij.com',
    url = 'https://openjij.github.io/OpenJij/',
    description='Framework for ising model and QUBO',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy'],
    ext_modules=[CMakeExtension('cxxjij')],
    cmdclass=dict(build_ext=CMakeBuild, test=GoogleTestCommand),
    packages=find_packages(exclude=('tests', 'docs')),
    license='Apache License 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
    zip_safe=False
)
