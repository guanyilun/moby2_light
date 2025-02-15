import numpy as np


class TODSnippet:
    def __init__(self, tod=None, det_uid=None, tslice=None):
        """Create a TOD snippet with a subset of dets and a slice of samples

        Parameters
        ----------
        tod: base TOD object
        det_uid: list of dets of interests
        tslice: a slice object to get a subset of samples

        """
        if tod is None: return
        self.det_uid = det_uid
        self.data = tod.data[det_uid,tslice]
        self.tslice = tslice
        self.info = tod.info.copy()
    def demean(self):
        """Remove the mean of the snippet"""
        self._mean = self.data.mean(axis=1)
        self.data -= self._mean[:,None]
        return self
    def deslope(self):
        self._slope = (self.data[:,-1] - self.data[:,0]) / self.data.shape[-1]
        self.data -= self._slope[:,None] * np.arange(self.data.shape[-1])
        return self
    def __repr__(self):
        return f"TODSnippet(ndet={len(self.det_uid)},tslice={self.tslice})"
    def peaks(self):
        """Get peak amplitudes"""
        return np.max(np.abs(self.data), axis=1)

        
def get_glitch_snippets(tod, dets, cv):
    """Generate TODSnippet from a given CutsVector

    Parameters
    ----------
    tod: TOD object
    dets: list of dets of interests
    rng: time index range of the snippet of form [i_l, i_h]
    rm: whether to remove mean in each snippets

    Returns
    -------
    [TODSnippet] (list of TODSnippet)

    """
    snippets = []
    tslices = cv2slices(cv)
    for s in tslices:
        snippets.append(TODSnippet(tod, dets, s).demean())
    return snippets


def cv2slices(cv):
    """Convert CutsVector to a list ot time slices"""
    return [slice(v[0],v[1],None) for v in cv]


def affected_snippets_from_cv(tod, cuts, cv, dets):
    """Get snippets from a given cut vector while only maintaining
    those dets that are affected in each range in the cv.

    Parameters
    ----------
    tod: base TOD object
    cuts: TODCuts object containing the base cuts to extract affected dets from
    cv: a CutsVector object that specifies ranges of interests
    dets: a narrow-down list of dets to look at

    """
    dets_events = dets_affected_in_cv(cuts, cv, dets)
    snippets = []
    for d, s in zip(dets_events, cv2slices(cv)):
        snippets.append(TODSnippet(tod, d, s))
    return snippets

def glitch_det_count(cuts, dets=None):
    """Count number of dets affected as a time series

    Parameters
    ----------
    cuts: TODCuts object

    """
    if dets is None:
        return np.sum([c.get_mask() for c in cuts.cuts], axis=0)
    else:
        return np.sum([c.get_mask() for i, c in enumerate(cuts.cuts)
                       if cuts.det_uid[i] in dets], axis=0)


def pcuts2mask(cuts):
    """Convert partial cuts to a 2d boolean mask"""
    mask = np.stack([c.get_mask() for c in cuts.cuts], axis=0)
    return mask


def is_cut(cv, t):
    """check if a specific time is cut in a det (provided CutVector)

    Parameters
    ----------
    cv: CutsVector
    t: time index

    Returns
    -------
    True if t is cut else False

    """
    for c in cv:
        if c[0] <= t <= c[1]:
            return True
    return False

def dets_affected_at_t(cuts, t):
    """Find dets affected by the given cuts at a specific time t

    Parameters
    ----------
    cuts: TODCuts object
    t: time index

    Returns
    -------
    [det_uid]

    """
    return [d for (d, cv) in zip(cuts.det_uid, cuts.cuts) if is_cut(cv, t)]

    
def fill_cv(data, cv, fill_value=0, inplace=True):
    """Fill an array-like data with a CutsVector, by default
    it acts on the last axis.

    """
    ss = cv2slices(cv)
    if not inplace: data = data.copy()
    for s in ss:
        data[...,s] = fill_value
    return data

def dets_affected_in_cv(cuts, cv, dets):
    """Find detectors affected in each range in a CutsVector

    Parameters
    ----------
    cuts (TODCuts): base TODCuts object to gather the dets affected info
    cv (CutsVector): specify the ranges of samples to find dets affected
    dets (boolean array): an narrowed-down list of dets to look at

    """
    get_dets = lambda x: np.where((np.sum(x,axis=1)>0)*dets)[0]
    return slices_map(get_dets, pcuts2mask(cuts), cv2slices(cv))


def slices_map(func, data, slices):
    """Apply a function on the data inside given slices, by default
    the slice is applied on the last axis

    """
    res = []
    for s in slices:
        res.append(func(data[...,s]))
    return res