add_library(algorithm INTERFACE)
target_include_directories(algorithm
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# set(ALGORITHM_LIBRARY sa sqa)

# add_library(algorithm SHARED ${ALGORITHM_LIBRARY})
# add_library(algorithm_static STATIC ${ALGORITHM_LIBRARY})

