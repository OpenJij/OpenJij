set(ALGORITHM_LIBRARY sa sqa)

add_library(algorithm SHARED ${ALGORITHM_LIBRARY})
add_library(algorithm_static STATIC ${ALGORITHM_LIBRARY})

