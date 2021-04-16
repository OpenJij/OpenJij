include(FetchContent)
FetchContent_Declare(
    boostorg
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG        boost-1.75.0
)

FetchContent_GetProperties(Boost)

if (NOT boostorg_POPULATED)
    FetchContent_Populate(boostorg)
    add_subdirectory(${boostorg_SOURCE_DIR} ${boostorg_BINARY_DIR})
endif ()

FetchContent_Populate(Boost)

