add_library(graph
    SHARED
    graph.cpp
    dense.cpp
    sparse.cpp
    square.cpp
    chimera.cpp
)

add_library(graph_static
    STATIC
    graph.cpp
    dense.cpp
    sparse.cpp
    square.cpp
    chimera.cpp
)
