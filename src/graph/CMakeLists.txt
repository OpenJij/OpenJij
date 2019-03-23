set(GRAPH_LIBRARY graph dense sparse square chimera)

add_library(graph SHARED ${GRAPH_LIBRARY})
add_library(graph_static STATIC ${GRAPH_LIBRARY})
