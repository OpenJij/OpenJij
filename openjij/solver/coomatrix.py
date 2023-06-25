from __future__ import annotations

import dataclasses
import typing as typ

import numpy as np
import numpy.typing as npt

import openjij.cxxjij as cj


KeyType = typ.TypeVar("KeyType", int, str)


@dataclasses.dataclass
class COOMatrix:
    row: npt.NDArray[np.int64]
    col: npt.NDArray[np.int64]
    val: npt.NDArray[np.float64]
    size: int

    def get_cxxjij_graph(self) -> cj.SparseSymmetricGraph:
        return cj.SparseSymmetricGraph(
            row = self.row,
            col = self.col,
            values = self.val,
        )

    @classmethod
    def from_dict(
        cls,
        dict_coo: dict[tuple[KeyType, KeyType], float],
    ) -> tuple[COOMatrix, dict[KeyType, int], set[KeyType]]:
        """Create COO matrix from dict.

        Args:
            dict_coo (dict[(int, int), float] or dict[(str, str), float]): COO matrix.
            constant (float): Constant energy offset. Defaults to 0.0.

        Returns:
            tuple[COOMatrix, dict[int, int], set[int]]: COO matrix, variable map, ignored variables.
        """
        row = []
        col = []
        val = []
        variable_map: dict[KeyType, int] = {} 
        all_variables: set[KeyType] = set([]) 
        for (i, j), v in dict_coo.items():
            if v != 0.0:
                if i not in variable_map:
                    variable_map[i] = len(variable_map)
                if j not in variable_map:
                    variable_map[j] = len(variable_map)
                row.append(variable_map[i])
                col.append(variable_map[j])
                val.append(v)
            all_variables.add(i)
            all_variables.add(j)

        # Extract ignored variables
        ignored_variables = all_variables - set(variable_map.keys())

        coo_matrix = COOMatrix(
            row=np.array(row, dtype=np.int64),
            col=np.array(col, dtype=np.int64),
            val=np.array(val, dtype=np.float64),
            size=max(max(row), max(col)) + 1,
        )
        return coo_matrix, variable_map, ignored_variables

