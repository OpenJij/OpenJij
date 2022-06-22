from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import openjij.model.chimera_model, openjij.model.king_graph

from openjij.model.model import (
    BinaryPolynomialModel,
    BinaryQuadraticModel,
    bqm_from_numpy_matrix,
    make_BinaryPolynomialModel,
    make_BinaryPolynomialModel_from_hising,
    make_BinaryPolynomialModel_from_hubo,
    make_BinaryPolynomialModel_from_JSON,
    make_BinaryQuadraticModel,
    make_BinaryQuadraticModel_from_JSON,
)

__all__ = [
    "make_BinaryQuadraticModel",
    "make_BinaryQuadraticModel",
    "BinaryQuadraticModel",
    "bqm_from_numpy_matrix",
    "make_BinaryPolynomialModel",
    "make_BinaryPolynomialModel_from_JSON",
    "BinaryPolynomialModel",
    "make_BinaryPolynomialModel_from_hising",
    "make_BinaryPolynomialModel_from_hubo",
]
