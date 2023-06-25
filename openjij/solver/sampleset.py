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

    def __repr__(self) -> str:
        vartype_str = super().__repr__()
        return vartype_str[1:].split(':')[0]


KeyType = typ.TypeVar("KeyType", int, str)


@dataclasses.dataclass
class Sample(typ.Generic[KeyType]):
    """Sample class for storing a sample.

    Attributes:
        energy (float): Energy of the sample.
        result (np.ndarray): Sample.
        vartype (VarType): Variable type of the sample.
        num_occurrences (int): Number of occurrences of the sample. default is 1.
        _info (dict[str, typing.Any]): Information of the sample.
    """
    energy: float
    value: dict[KeyType, float]
    vartype: VarType
    num_occurrences: int = 1
    _info: dict[str, typ.Any] = dataclasses.field(default_factory=dict)
    run_id: str = dataclasses.field(default_factory=lambda : str(uuid.uuid4()))



class SampleSet(UserList):

    """set of Sample object.

    Attributes:
        data (list[Sample]): list of Sample object.
    """

    def __init__(self, samples: list[Sample]) -> None:
        super().__init__(samples)
        self.data: list[Sample] = samples
        self.info: dict[str, typ.Any] = {}

    @property
    def energies(self) -> list[float]:
        return [sample.energy for sample in self.data]
    
    @property
    def states(self) -> list[dict[KeyType, float]]:
        return [sample.value for sample in self.data]

    def lowest(self) -> 'SampleSet':
        min_index = np.argmin(self.energies)
        return SampleSet([self.data[min_index]])

    def separate(self) -> dict[str, 'SampleSet']:
        """Separate samples by run_id.

        Returns:
            dict[str, SampleSet]: separated samples.
        
        Examples:
            >>> import openjij as oj
            >>> sampleset1 = oj.solver.SampleSet.from_array([[1, -1], [1, 1]], [0.0, 1.0], oj.solver.VarType.SPIN, run_id="run1")
            >>> sampleset2 = oj.solver.SampleSet.from_array([[1, -1], [1,-1]], [0.0, 1.0], oj.solver.VarType.SPIN, run_id="run2")
            >>> sample12 = sampleset1.concat(sampleset2)
            >>> sample12.separate()
            {'run1': [Sample(energy=0.0, result=[1, -1], vartype=VarType.SPIN, num_occurrences=1, _info={}, run_id='run1'), Sample(energy=1.0, result=[1, 1], vartype=VarType.SPIN, num_occurrences=1, _info={}, run_id='run1')], 'run2': [Sample(energy=0.0, result=[1, -1], vartype=VarType.SPIN, num_occurrences=1, _info={}, run_id='run2'), Sample(energy=1.0, result=[1, -1], vartype=VarType.SPIN, num_occurrences=1, _info={}, run_id='run2')]}

        """
        separeted_samples: dict[str, SampleSet] = {}
        for sample in self.data:
            if sample.run_id in separeted_samples:
                separeted_samples[sample.run_id].append(sample)
            else:
                separeted_samples[sample.run_id] = SampleSet([sample])
        return separeted_samples

    def concat(self, other: SampleSet) -> 'SampleSet':
        return SampleSet(self.data + other.data)

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
    def from_array(
            cls,
            states: typ.Sequence[typ.Sequence[float]],
            energies: typ.Sequence[float], vartype: VarType,
            info_list: typ.Optional[list[dict[str, typ.Any]]] = None,
            run_id: typ.Optional[str] = None,
            variable_labels: typ.Optional[typ.Sequence[KeyType]] = None
        ) -> 'SampleSet':
        """Create SampleSet from array.
        """
        if variable_labels is None:
            var_labels = list(range(len(states[0])))
        else:
            var_labels = variable_labels

        # Check length of states and variable_labels
        if len(states[0]) != len(var_labels):
            raise ValueError("length of states and variable_labels must be same.")

        samples = [{var_labels[i]: float(s) for i, s in enumerate(state)} for state in states]

        return cls.from_dict_value(samples, energies, vartype, info_list, run_id)


    @classmethod
    def from_dict_value(
            cls,
            states: typ.Sequence[dict[KeyType, float]],
            energies: typ.Sequence[float], vartype: VarType,
            info_list: typ.Optional[list[dict[str, typ.Any]]] = None,
            run_id: typ.Optional[str] = None
    ) -> 'SampleSet':
        
        # Check length of states and energies
        if len(states) != len(energies):
            raise ValueError("length of states and energies must be same.")

        num_samples = len(states)

        if info_list is None:
            info_list = [{} for _ in range(num_samples)]

        if len(info_list) < num_samples:
            info_list = list(info_list) + [{} for _ in range(num_samples - len(info_list))]
        
        if run_id is None:
            run_id = str(uuid.uuid4())

        return cls([Sample(
                energy=energy,
                value=state,
                vartype=vartype,
                run_id=run_id, _info=info)
                for state, energy, info in zip(states, energies, info_list)])


    def __getitem__(self, index) -> Sample:
        return super().__getitem__(index)    

def _compresse_sampleset(sample_set: SampleSet) -> SampleSet:
    """Compress the sampleset.

    Args:
        sample_set (SampleSet): sampleset to be compressed.
    
    Returns:
        SampleSet: compressed sampleset.
    
    Examples:
        >>> import openjij as oj
        >>> sampleset1 = oj.solver.SampleSet.from_array([[1, -1], [1, 1], [1, -1]], [0.0, 1.0, 0.0], oj.solver.VarType.SPIN, run_id="run1")
        >>> compressed = _compresse_sampleset(sampleset1)
        >>> compressed.states
        [[1, -1], [1, 1]]
        >>> compressed.energies
        [0.0, 1.0]
        >>> compressed.num_occurrences
        [2, 1]

    """
    # choose the samples that have the same energy
    compressed_samples: dict[int, Sample] = {}
    samples: list[Sample] = []
    energies: list[float] = []
    for sample in sample_set.data:
        # 1. Check same energy is already in the list
        if sample.energy in energies:
            energy_index = energies.index(sample.energy)
            result_is_same = len(sample.value) == len(samples[energy_index].value)
            result_is_same = result_is_same and np.all(sample.value == samples[energy_index].value)

            # if the energy is already in the list, add the number of occurances
            if result_is_same:
                if energy_index not in compressed_samples:
                    compressed_samples[energy_index] = sample
                compressed_samples[energy_index].num_occurrences += 1
            else:
            # if the energy is not in the list, add the energy to the list
                energies.append(sample.energy)
                samples.append(sample)
                compressed_samples[len(energies) - 1] = sample
        else:
            # if the energy is not in the list, add the energy to the list
            energies.append(sample.energy)
            samples.append(sample)
            compressed_samples[len(energies) - 1] = sample

    return SampleSet(list(compressed_samples.values()))


