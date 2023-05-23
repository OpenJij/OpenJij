from __future__ import annotations

import typing as typ
from collections import UserList
import dataclasses
import enum
import uuid

import numpy as np
import numpy.typing as npt


class VarType(enum.Enum):
    SPIN = enum.auto()
    BINARY = enum.auto()
    INTEGER = enum.auto()
    REAL = enum.auto()


@dataclasses.dataclass
class Sample:
    energy: float
    result: npt.NDArray[np.float64]
    vartype: VarType
    num_occurances: int = 1
    _info: dict[str, typ.Any] = dataclasses.field(default_factory=dict)
    run_id: str = dataclasses.field(default_factory=lambda : str(uuid.uuid4()))

    @property
    def state(self) -> npt.NDArray[np.float64]:
        return self.result


class SampleSet(UserList):
    def __init__(self, samples: list[Sample]) -> None:
        super().__init__(samples)
        self.data: list[Sample] = samples

    @property
    def energies(self) -> list[float]:
        return [sample.energy for sample in self.data]
    
    @property
    def states(self) -> list[npt.NDArray[np.float64]]:
        return [sample.state for sample in self.data]

    def lowest(self) -> 'SampleSet':
        min_index = np.argmin(self.energies)
        return SampleSet([self.data[min_index]])

    def separate(self) -> dict[str, 'SampleSet']:
        separeted_samples: dict[str, SampleSet] = {}
        for sample in self.data:
            if sample.run_id in separeted_samples:
                separeted_samples[sample.run_id].append(sample)
            else:
                separeted_samples[sample.run_id] = SampleSet([sample])
        return separeted_samples

    def compresse(self, ignore_run_id: bool = False) -> 'SampleSet':
        if ignore_run_id:
            return _compresse_sampleset(self)
        else:
            separeted_samples = self.separate()
            compressed_samples: list[Sample] = []
            for run_id in separeted_samples:
                compressed_samples.extend(_compresse_sampleset(separeted_samples[run_id]))
            return SampleSet(compressed_samples)

    @classmethod
    def from_states_and_energies(cls, states: typ.Sequence[npt.NDArray[np.float64]], energies: typ.Sequence[float], vartype: VarType, info_list: typ.Optional[list[dict[str, typ.Any]]] = None) -> 'SampleSet':
        if info_list is None:
            info_list = [{} for _ in range(len(states))]
        run_id = str(uuid.uuid4())
        return cls([Sample(energy=energy, result=state, vartype=vartype, run_id=run_id, _info=info) for state, energy, info in zip(states, energies, info_list)])


def _compresse_sampleset(sample_set: SampleSet) -> SampleSet:
    # choose the samples that have the same energy
    compressed_samples: dict[int, Sample] = {}
    samples: list[Sample] = []
    energies: list[float] = []
    for sample in sample_set:
        try:
            # if the energy is already in the list, add the number of occurances
            energy_index = energies.index(sample.energy)
            result_is_same = len(sample.result) == len(samples[energy_index].result) and np.all(sample.result == samples[energy_index].result)
            if result_is_same:
                if energy_index not in compressed_samples:
                    compressed_samples[energy_index] = sample
                compressed_samples[energy_index].num_occurances += 1
        except ValueError:
            # if the energy is not in the list, add the energy to the list
            energies.append(sample.energy)
            samples.append(sample)

    return SampleSet(list(compressed_samples.values()))


