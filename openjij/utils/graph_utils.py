from __future__ import annotations
import numpy as np


def qubo_to_ising(mat: np.ndarray):
    """Inplace-convert numpy matrix from qubo to ising.

    Args:
        mat (np.ndarray): numpy matrix
    """
    mat /= 4
    for i in range(mat.shape[0]):
        mat[i, i] += np.sum(mat[i, :])


def chimera_to_ind(r: int, c: int, z: int, L: int):
    """[summary]

    Args:
        r (int): row index
        c (int): column index
        z (int): in-chimera index (must be from 0 to 7)
        L (int): height and width of  chimera-units (total number of spins is :math:`L \\times L \\times 8`)

    Raises:
        ValueError: [description]

    Returns:
        int: corresponding Chimera index
    """
    if not (0 <= r < L and 0 <= c < L and 0 <= z < 8):
        raise ValueError(
            "0 <= r < L or 0 <= c < L or 0 <= z < 8. "
            "your input r={}, c={}, z={}, L={}".format(r, c, z, L)
        )
    return r * L * 8 + c * 8 + z
