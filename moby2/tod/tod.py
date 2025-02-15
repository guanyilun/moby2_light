import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class TODInfo:
    array_data: Any
    sample_index: int = 0

    def copy(self):
        return TODInfo(self.array_data, self.sample_index)


@dataclass
class TOD:
    ctime: np.ndarray
    data: np.ndarray
    det_uid: np.ndarray
    nsamps: int
    info: TODInfo
    sample_offset: int = 0

    @classmethod
    def from_npz(cls, filename):
        data_z = np.load(filename, allow_pickle=True)
        data = data_z['arr_0'].item()
        return cls(data['ctime'], data['data'], np.arange(data['data'].shape[0]), len(data['ctime']), TODInfo(data['array_data']))