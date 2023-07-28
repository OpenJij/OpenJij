# Copyright 2023 Jij Inc.

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

try:
    from skbuild import setup
except ImportError:
    from setuptools import setup

setup_requires = [
    "numpy",
    "pybind11",
    "cmake > 3.20",
    "scikit-build > 0.16.0"
]

if any(arg in sys.argv for arg in ("pytest", "test")):
    setup_requires.append("pytest-runner")

setup(
    setup_requires=setup_requires,
    packages=[
        "openjij",
        "openjij.model",
        "openjij.sampler",
        "openjij.sampler.chimera_gpu",
        "openjij.utils",
    ],
    cmake_install_dir="openjij",
    include_package_data=False,
    zip_safe=False,
)
