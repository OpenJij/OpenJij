//    Copyright 2023 Jij Inc.

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#ifdef __clang__
//   Eigen generates a bunch of implicit-copy-constructor-is-deprecated warnings
//   with -Wdeprecated under Clang, so disable that warning here:
#pragma GCC diagnostic ignored "-Wdeprecated"
#endif
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
//   Eigen generates a bunch of implicit-copy-constructor-is-deprecated warnings
//   with -Wdeprecated under GCC, so disable that warning here:
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#endif
