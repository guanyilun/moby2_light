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
    def from_npz_old(cls, filename):
        data_z = np.load(filename, allow_pickle=True)
        data = data_z['arr_0'].item()
        return cls(data['ctime'], data['data'], np.arange(data['data'].shape[0]), len(data['ctime']), TODInfo(data['array_data']))
    @classmethod
    def from_npz(cls, filename):
        data_z = np.load(filename, allow_pickle=True)
        data = data_z['data'].item()
        nsamps = 400*60*6
        return cls(data['ctime'][:nsamps], data['data'][:, :nsamps], np.arange(data['data'].shape[0]), len(data['ctime'][:, :nsamps]), TODInfo(data['array_data']))

    @classmethod
    def from_npz_sims(cls, dir, todname, amp, halflife):
        data_z = np.load('{}/{}.npz'.format(dir, todname), allow_pickle=True)
        data = data_z['data'].item()
        data_z_sim = np.load('{}/sim_{}_amp{}_h{}.npz'.format(dir, todname, amp, halflife), allow_pickle=True)
        nsamps = 400*60*6
        data_sim = data_z_sim['data'].item()
        for d in range(len(data_sim['det_uid'])):
            data['data'][data_sim['det_uid'][d], :nsamps] = data_sim['data'][d, :nsamps]
        return cls(data['ctime'][:nsamps], data['data'][:, :nsamps], np.arange(data['data'].shape[0]), len(data['ctime'][:nsamps]), TODInfo(data['array_data']))


def detrend_tod(tod=None, dets=None, data=None):
    """
    Subtract a line through the first and last datapoints of each tod,
    preserving the data mean.

    Instead of a tod, you can pass in a data array through data=... .
    
    Returns (y0, y1), which are vectors of the values removed from the
    first and last samples, respectively.
    """
    if data is None:
        data = tod.data
    one_d = data.ndim == 1
    if one_d:
        data.shape = (1,-1)
    if dets is None:
        dets = np.ones(data.shape[0], 'bool')
    if np.asarray(dets).dtype == 'bool':
        dets = np.nonzero(dets)[0]
    y0, y1 = data[dets,0], data[dets,-1]
    slopes = (y1-y0) / (data.shape[1]-1)
    y0, y1 = -(y1-y0)/2, (y1-y0)/2
    x = np.arange(data.shape[1]).astype('float')
    for di, _y0, _slope in zip(dets, y0, slopes):
        data[di] -= x*_slope + _y0
    if one_d:
        data.shape = -1
        return np.array((y0[0], y1[0]))
    return np.array((y0, y1))


def retrend_tod(trends, tod=None, dets=None, data=None):
    """
    Put the trends back in.
    """
    if data is None:
        data = tod.data
    one_d = data.ndim == 1
    if one_d:
        data = data.reshape(1,-1)  # better view for us
        trends = np.array(trends).reshape(1,2)  # sanity check?
    if dets is None:
        dets = np.ones(data.shape[0], 'bool')
    if np.asarray(dets).dtype == 'bool':
        dets = np.nonzero(dets)[0]
    y0, y1 = trends
    slopes = (y1-y0) / (data.shape[1]-1)
    y0, y1 = -(y1-y0)/2, (y1-y0)/2
    x = np.arange(data.shape[1]).astype('float')
    for di, _y0, _slope in zip(dets, y0, slopes):
        data[di] += x*_slope + _y0


def remove_mean(tod=None, dets=None, data=None):
    """
    Perform
        data[dets] -= data[dets].mean(axis=1)

    If data is not provided, tod.data is used.
    """
    if data is None:
        data = tod.data
    if dets is None:
        dets = np.arange(data.shape[0])
    data[dets] -= data[dets].mean(axis=1, keepdims=True)