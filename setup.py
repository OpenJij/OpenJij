import os
import re
import sys
import platform
import sysconfig
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from distutils.version import LooseVersion


# Package meta-data.
NAME = 'openjij'
DESCRIPTION = 'Framework for the Ising model and QUBO'
EMAIL = 'openjij@j-ij.com'
AUTHOR = 'Jij Inc.'

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

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
            if cmake_version < '3.17.0':
                raise RuntimeError("CMake >= 3.17.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)
    def build_extension(self, ext):
        print(platform.system())
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = 'Debug' if self.debug else 'Release'
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable, 
                      "-DOPENJIJ_VERSION_INFO={}".format(self.distribution.get_version()),
                      "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
                      #"-DCMAKE_FIND_DEBUG_MODE=1",
                     ]
        build_args = []
        
        if platform.system() != "Windows":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
                ]
                build_args += ["--config", cfg]
        
        if platform.system() == 'Darwin':
            # disable macos openmp since addtional dependency is needed.
            if not {'True': True, 'False': False}[os.getenv('FORCE_USE_OMP', 'False')]:
                print("FORCE_USE_OMP=No")
                cmake_args += ['-DFORCE_USE_OMP=No']
            else:
                print("FORCE_USE_OMP=Yes")
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]
                #if platform.processor() == archs: 
                #    print("COPY libomp.")
                #    shutil.copytree("/usr/local/opt/libomp", "./libomp")
            #else:
                #print("COPY libomp.")
                #shutil.copytree("/usr/local/opt/libomp", "./libomp")
                
        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]       
        if "USE_TEST" in os.environ:  
            cmake_args += ["-DUSE_TEST=Yes", "-LA", "-LH"]
        
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


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


setup(
    name=NAME,
    version_config=True,
    author=AUTHOR,
    author_email='openjij@j-ij.com',
    url='https://openjij.github.io/OpenJij/',
    description='Framework for the Ising model and QUBO',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires = ">=3.7, <3.11",
    setup_requires=[
        'pytest-runner', 
        'setuptools-git-versioning', 
    ],
    install_requires=[
        'dimod >= 0.9.1',
        'numpy >= 1.18.4, !=1.21.0, !=1.21.1', 
        'scipy', 
        'requests',
        'jij-cimod >= 1.2.3', 
        'typing-extensions >= 3.10.0; python_version < "3.8.0"'
    ],
    tests_require = [
        'pytest',
        'pytest-mock',
        'pytest-cov', 
        'pytest-html',
        ],
    ext_modules=[CMakeExtension('cxxjij')],
    cmdclass=dict(build_ext=CMakeBuild, test=GoogleTestCommand,
                  pytest=PyTestCommand),
    packages=[  
        'openjij', 
        'openjij.model', 
        'openjij.sampler', 
        'openjij.sampler.chimera_gpu', 
        'openjij.utils',
        ],
    license='Apache License 2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    platforms=[
                'Operating System :: Microsoft :: Windows :: Windows 10',
                'Operating System :: MacOS :: MacOS X',
                'Operating System :: POSIX :: Linux',
                ],
    zip_safe=False
)
