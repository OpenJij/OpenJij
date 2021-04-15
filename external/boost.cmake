include(FetchContent)
FetchContent_Declare(
    Boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_SUBMODULES libs/system libs/serialization libs/random
                   libs/function libs/config libs/headers libs/assert libs/core libs/integer
                   libs/type_traits libs/mpl libs/throw_exception libs/preprocessor libs/utility
                   libs/static_assert libs/smart_ptr libs/predef libs/move libs/io libs/iterator
                   libs/detail libs/spirit libs/optional libs/type_index libs/container_hash
                   libs/array libs/bind
                   tools/build tools/boost_install
    GIT_TAG        boost-1.75.0
)

FetchContent_GetProperties(Boost)
FetchContent_Populate(Boost)

