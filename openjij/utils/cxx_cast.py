from openjij.cxxjij.algorithm import (
    UpdateMethod,
    RandomNumberEngine
)
from openjij.cxxjij.utility import TemperatureSchedule


def cast_to_cxx_update_method(update_method: str) -> UpdateMethod:
    if update_method == "METROPOLIS":
        return UpdateMethod.METROPOLIS
    elif update_method == "HEAT_BATH":
        return UpdateMethod.HEAT_BATH
    else:
        raise RuntimeError(f"Invalid update_method={update_method}")


def cast_to_cxx_random_number_engine(
    random_number_engine: str,
) -> RandomNumberEngine:
    if random_number_engine == "XORSHIFT":
        return RandomNumberEngine.XORSHIFT
    elif random_number_engine == "MT":
        return RandomNumberEngine.MT
    elif random_number_engine == "MT_64":
        return RandomNumberEngine.MT_64
    else:
        raise RuntimeError(f"Invalid random_number_engine={random_number_engine}")


def cast_to_cxx_temperature_schedule(
    temperature_schedule: str,
) -> TemperatureSchedule:
    if temperature_schedule == "GEOMETRIC":
        return TemperatureSchedule.GEOMETRIC
    elif temperature_schedule == "LINEAR":
        return TemperatureSchedule.LINEAR
    else:
        raise RuntimeError(f"Invalid temperature_schedule={temperature_schedule}")
