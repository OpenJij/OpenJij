# Copyright 2022 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from packaging.version import LegacyVersion

try:
    from skbuild import setup
    from skbuild.cmaker import get_cmake_version
    from skbuild.exceptions import SKBuildError
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

setup_requires = [
    "numpy",
    "pybind11",
    "ninja; sys_platform != 'win32'",
]

if any(arg in sys.argv for arg in ("pytest", "test")):
    setup_requires.append("pytest-runner")

# Add CMake as a build requirement if cmake is not installed or is too low a version.
try:
    if LegacyVersion(get_cmake_version()) < LegacyVersion("3.22"):
        setup_requires.append("cmake")
except SKBuildError:
    setup_requires.append("cmake")

setup(
    setup_requires=setup_requires,
    install_requires=[
        "dimod >= 0.9.1, <0.12.0",
        "numpy >= 1.21.2",
        "scipy",
        "requests",
        "jij-cimod == 1.4.0a1",
        'typing-extensions >= 3.10.0; python_version < "3.8.0"',
    ],
    packages=[
        "openjij",
        "openjij.model",
        "openjij.sampler",
        "openjij.sampler.chimera_gpu",
        "openjij.utils",
    ],
    cmake_install_dir="openjij",
    include_package_data=True,
    zip_safe=False,
)
