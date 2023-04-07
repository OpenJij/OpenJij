from __future__ import annotations
try:
    from typing import Optional, Union
except ImportError:
    from typing_extensions import Optional, Union

import openjij as oj
from openjij.sampler.response import Response
from openjij.cxxjij.graph import (
    BinaryPolynomialModel,
    IsingPolynomialModel
)
from openjij.cxxjij.sampler import make_sa_sampler
from openjij.utils.cxx_cast import (
    cast_to_cxx_update_method,
    cast_to_cxx_random_number_engine,
    cast_to_cxx_temperature_schedule
)

def to_oj_response(
    variables: list[list[Union[int, float]]], 
    index_list: list[Union[int, str, tuple[int, ...]]],
    energies: list[float], 
    vartype: str
) -> Response:
    return Response.from_samples(
        samples_like=[dict(zip(index_list, v_list)) for v_list in variables], 
        vartype=vartype, 
        energy=energies
    )  

def base_sample_hubo(
    hubo: dict[tuple, float],
    vartype: Optional[str] = None,
    num_sweeps: int = 1000,
    num_reads: int = 1,
    num_threads: int = 1,
    beta_min: Optional[float] = None,
    beta_max: Optional[float] = None,
    update_method: str = "METROPOLIS",
    random_number_engine: str = "XORSHIFT",
    seed: Optional[int] = None,
    temperature_schedule: str = "GEOMETRIC",
) -> Response:
    
    if vartype == "BINARY":
        sampler = make_sa_sampler(
            BinaryPolynomialModel(
                key_list=list(hubo.keys()), 
                value_list=list(hubo.values())
            )
        )
    elif vartype == "SPIN":
        sampler = make_sa_sampler(
            IsingPolynomialModel(
                key_list=list(hubo.keys()), 
                value_list=list(hubo.values())
            )
        )
    else:
        raise ValueError("vartype must `BINARY` or `SPIN`")


    sampler.set_num_sweeps(num_sweeps=num_sweeps)
    sampler.set_num_reads(num_reads=num_reads)
    sampler.set_num_threads(num_threads=num_threads)
    sampler.set_update_method(
        update_method=cast_to_cxx_update_method(update_method)
    )
    sampler.set_random_number_engine(
        random_number_engine=cast_to_cxx_random_number_engine(random_number_engine)
    )
    sampler.set_temperature_schedule(
        temperature_schedule=cast_to_cxx_temperature_schedule(temperature_schedule)
    )

    if beta_min is not None:
        sampler.set_beta_min(beta_min=beta_min)
    else:
        sampler.set_beta_min_auto()

    if beta_max is not None:
        sampler.set_beta_max(beta_max=beta_max)
    else:
        sampler.set_beta_max_auto()

    if seed is not None:
        sampler.sample(seed=seed)
    else:
        sampler.sample()

    response = to_oj_response(
        sampler.get_samples(), 
        sampler.get_index_list(),
        sampler.calculate_energies(),
        vartype
    )

    return response

