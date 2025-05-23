import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
import time
import numpy.typing as npt
from scipy import stats
from numba import jit, prange

from . import filters


@dataclass
class GlitchParams:
    """Parameters for glitch detection algorithm.
    
    Attributes:
        n_sig: Number of sigma for glitch detection threshold
        t_glitch: Characteristic time of glitch (seconds)
        min_separation: Minimum separation between glitches (samples)
        max_glitch: Maximum number of glitches to detect
        high_pass_fc: High-pass filter cutoff frequency (Hz)
        buffer: Number of samples to buffer around detected glitches
    """
    n_sig: float = 10.0
    t_glitch: float = 0.002
    min_separation: int = 30
    max_glitch: int = 50000
    high_pass_fc: float = 5.0
    buffer: int = 6


@jit(nopython=True)
def cuts_vector_from_mask(mask: npt.NDArray[np.bool_]) -> Tuple[npt.NDArray[np.int32], int]:
    """Numba-optimized implementation of cuts vector from mask conversion.

    Args:
        mask: Boolean array where True indicates cut samples
    
    Returns:
        Tuple of (cuts array, number of cuts)
        cuts array has shape (n_cuts * 2,) and contains alternating start/end indices
    """
    n_mask = len(mask)
    # First pass: count cuts
    n_cuts = 0
    cutting = False
    
    for i in range(n_mask):
        if cutting and not mask[i]:
            cutting = False
        elif not cutting and mask[i]:
            cutting = True
            n_cuts += 1
            
    if n_cuts == 0:
        return np.empty(0, dtype=np.int32), 0
        
    # Second pass: store cuts
    cuts = np.empty(n_cuts * 2, dtype=np.int32)
    oindex = 0
    cutting = False
    
    for i in range(n_mask):
        if cutting and not mask[i]:
            cutting = False
            cuts[oindex] = i
            oindex += 1
        elif not cutting and mask[i]:
            cutting = True
            cuts[oindex] = i
            oindex += 1
            
    if cutting:
        cuts[oindex] = n_mask
        oindex += 1
        
    assert oindex == n_cuts * 2
    return cuts, n_cuts


@jit(nopython=True)
def cuts_to_mask(cuts: npt.NDArray[np.int32], n_cuts: int, n_samples: int) -> npt.NDArray[np.bool_]:
    """Convert cuts array back to mask.
    
    Args:
        cuts: Array containing alternating start/end indices
        n_cuts: Number of cuts
        n_samples: Total number of samples
        
    Returns:
        Boolean mask array
    """
    mask = np.zeros(n_samples, dtype=np.bool_)
    for i in range(n_cuts):
        start = cuts[i * 2]
        end = cuts[i * 2 + 1]
        mask[start:end] = True
    return mask


def get_glitch_cuts(
    data: Optional[npt.NDArray] = None,
    dets: Optional[npt.NDArray[np.int32]] = None,
    tod: Optional[Any] = None,
    params: Dict[str, Any] = {},
    threads: int = 1
) -> "TODCuts":
    print("\nStarting glitch detection pipeline...")
    t_start = time.time()
    # Initialize glitch parameters
    glitch_params = GlitchParams()
    for key, value in params.items():
        setattr(glitch_params, key, value)
    
    # Get data and detector indices
    if data is None:
        if tod is None:
            raise ValueError("Either data or tod must be provided")
        data = tod.data
    
    if dets is None:
        dets = np.arange(data.shape[0], dtype=np.int32)
    else:
        dets = np.asarray(dets, dtype=np.int32)
    
    print(f"Processing {len(dets)} detectors with {data.shape[1]} samples each")
    # Create frequency domain filter
    t0 = time.time()
    filt_vec = (
        filters.sine2_high_pass(tod, fc=glitch_params.high_pass_fc) *
        filters.gaussian_filter(tod, time_sigma=glitch_params.t_glitch)
    )
    filt_vec = np.asarray(filt_vec, dtype=np.float32)
    print(f"Filter creation took {time.time() - t0:.2f} seconds")
    
    print("\nStarting vectorized glitch detection...")
    glitch_cuts = detect_glitches_vectorized(
        data=data,
        dets=dets,
        filt_vec=filt_vec,
        n_sig=glitch_params.n_sig,
        max_glitch=glitch_params.max_glitch,
        min_separation=glitch_params.min_separation
    )
    
    print("\nCreating cuts object...")
    # Create cuts object
    if tod is not None:
        cuts = TODCuts.for_tod(tod, assign=False)
    else:
        cuts = TODCuts(data.shape[0], data.shape[1])
    
    # Assign cuts for each detector
    for det, cut in zip(dets, glitch_cuts):
        cuts.cuts[det] = cut
    
    # Add buffer regions
    cuts.buffer(glitch_params.buffer)
    total_time = time.time() - t_start
    
    print(f"\nTotal glitch detection pipeline completed in {total_time:.2f} seconds")
    return cuts


def detect_glitches_vectorized(
    data: np.ndarray,
    dets: np.ndarray,
    filt_vec: np.ndarray,
    n_sig: float,
    max_glitch: int,
    min_separation: int,
    detrend: bool = True,
    retrend: bool = False,
) -> List["CutsVector"]:
    t0 = time.time()
    data = data[dets].copy()
    n_det, n_samp = data.shape

    print(f"\nDetrending data...")
    # 1. Vectorized Detrending
    if detrend:
        win = min(1000, n_samp // 2)
        x0 = np.mean(data[:, :win].real, axis=1, keepdims=True)
        x1 = np.mean(data[:, -win:].real, axis=1, keepdims=True)
        trend_slope = (x1 - x0) / (n_samp - 1.0 - win)
        j_indices = np.arange(n_samp)
        t_detrend = time.time()
        print(f"Detrending took {t_detrend - t0:.2f} seconds")
        trend = x0 + trend_slope * (j_indices - win/2)
        data -= trend

    print("\nPerforming FFT operations...")
    t_fft_start = time.time()
    # 2. FFT Operations using numpy
    data_fft = np.fft.fft(data, axis=1)
    data_fft *= filt_vec[np.newaxis, :]
    data = np.fft.ifft(data_fft, axis=1).real
    t_fft = time.time()
    print(f"FFT operations took {t_fft - t_fft_start:.2f} seconds")

    if retrend:
        print("\nRetrending data...")
        data += trend

    print("\nCalculating thresholds...")
    t_thresh_start = time.time()
    # 3. Vectorized Thresholding with SciPy
    iqr = stats.iqr(data, axis=1, scale='normal')  # Proper IQR calculation
    thresh = (iqr * n_sig)[:, np.newaxis]
    cut_mask = np.abs(data) > thresh
    t_thresh = time.time()
    print(f"Thresholding took {t_thresh - t_thresh_start:.2f} seconds")

    print("\nProcessing masks...")
    t_mask_start = time.time()
    # 4. Numba-accelerated mask processing
    final_masks = process_masks_batched(cut_mask.astype(bool), min_separation, max_glitch, n_samp)
    t_mask = time.time()
    print(f"Mask processing took {t_mask - t_mask_start:.2f} seconds")

    # Convert to CutsVector with max_glitch check
    return [CutsVector.new_always_cut(n_samp) if m.sum() > max_glitch*2
            else CutsVector.from_mask(m) for m in final_masks]


@jit(nopython=True)
def process_masks_batched(
    cut_masks: np.ndarray, min_sep: int, max_glitch: int, n_samp: int
) -> List[np.ndarray]:
    n_det = cut_masks.shape[0]
    final_masks = []
    for i in prange(n_det):
        mask = cut_masks[i]
        cuts, n_cuts = cuts_vector_from_mask(mask)
        if n_cuts > 0:
            merged_cuts = merge_cuts(cuts, n_cuts, min_sep)
            merged_mask = cuts_to_mask(merged_cuts, len(merged_cuts)//2, n_samp)
            if len(merged_cuts)//2 > max_glitch:
                merged_mask[:] = True
        else:
            merged_mask = mask
        final_masks.append(merged_mask)
    return final_masks


@jit(nopython=True)
def merge_cuts(cuts: np.ndarray, n_cuts: int, min_sep: int) -> np.ndarray:
    if n_cuts == 0:
        return cuts
    merged = [cuts[0], cuts[1]]
    for i in range(1, n_cuts):
        current_end = merged[-1]
        next_start = cuts[2*i]
        next_end = cuts[2*i + 1]
        if (next_start - current_end) < min_sep:
            merged[-1] = next_end
        else:
            merged.append(next_start)
            merged.append(next_end)
    return np.array(merged, dtype=np.int32)


class CutsVector:
    def __init__(self, cuts_in=None, nsamps=None, ncuts=0):
        self.nsamps = nsamps
        if cuts_in is not None:
            # Store cuts in the same format as numba functions expect
            # Convert from (n,2) to (n*2,) if needed
            if isinstance(cuts_in, list):
                cuts_in = np.array(cuts_in)
            if len(cuts_in.shape) == 2:
                self.cuts = cuts_in.ravel()
                self.n_cuts = len(cuts_in)
            else:
                self.cuts = cuts_in
                self.n_cuts = len(cuts_in) // 2
        else:
            self.cuts = np.zeros(ncuts * 2, dtype=np.int32)
            self.n_cuts = ncuts

    def __len__(self):
        return self.n_cuts

    def __getitem__(self, key):
        # Reshape to (n,2) format for compatibility
        return self.cuts.reshape(-1, 2)[key]

    def get_mask(self, nsamps=None, invert=False):
        if nsamps is None:
            nsamps = self.nsamps
        if len(self.cuts) == 0:
            mask = np.zeros(nsamps, dtype=bool)
        else:
            mask = cuts_to_mask(self.cuts, self.n_cuts, nsamps)
        return ~mask if invert else mask

    @classmethod
    def from_mask(cls, mask):
        mask = np.asarray(mask, dtype=bool)
        cuts, _ = cuts_vector_from_mask(mask)
        return cls(cuts_in=cuts, nsamps=len(mask))

    def get_complement(self):
        if len(self.cuts) == 0:
            return self.__class__([[0, self.nsamps]])
            
        # Convert to (n,2) format for computation
        cuts_2d = self.cuts.reshape(-1, 2)
        complement = np.empty((len(cuts_2d)+1, 2), dtype=np.int32)
        complement[0] = [0, cuts_2d[0,0]]
        if len(cuts_2d) > 1:
            complement[1:-1] = np.column_stack((cuts_2d[:-1,1], cuts_2d[1:,0]))
        complement[-1] = [cuts_2d[-1,1], self.nsamps]

        
        valid = complement[:,0] < complement[:,1]
        return self.__class__(complement[valid], nsamps=self.nsamps)

    def get_collapsed(self):
        if len(self.cuts) == 0:
            return self
            
        # Convert to (n,2) format for computation
        cuts = np.sort(self.cuts.reshape(-1, 2).copy(), axis=0)
        idx = np.ones(len(cuts), dtype=bool)
        idx[1:] = cuts[1:,0] > cuts[:-1,1]
        
        collapsed = cuts[idx]
        if self.nsamps is not None:
            np.clip(collapsed, 0, self.nsamps, out=collapsed)
            
        valid = collapsed[:,0] < collapsed[:,1]
        return self.__class__(collapsed[valid], self.nsamps)

    def get_resampled(self, resample=1):
        return self.__class__((self.cuts.reshape(-1, 2) // resample).ravel(), self.nsamps)

    def get_buffered(self, left, right=None):
        if len(self.cuts) == 0:
            return self.__class__(nsamps=self.nsamps)
            
        if right is None:
            right = left
            
        # Convert to (n,2) format for computation
        cuts_2d = self.cuts.reshape(-1, 2)
        buffered = cuts_2d + np.array([-left, right])
        return self.__class__(buffered, self.nsamps).get_collapsed()

    @classmethod
    def new_always_cut(cls, nsamps):
        return cls([[0, nsamps]], nsamps)


# Decorator to iterate operations on all dets.
def iterate_dets(f):
    f0 = f
    def iterated(self, det, *args, **kwargs):
        if not hasattr(det, '__len__') or  \
                (hasattr(det, 'ndim') and det.ndim == 0):
            return f0(self, det, *args, **kwargs)
        return [f0(self, d, *args, **kwargs) for d in det]
    # Preserve docstring
    iterated.__doc__ =  f.__doc__
    return iterated


class TODCuts:
    def __init__(self, ndets=None, nsamps=None, det_uid=None, sample_offset=0):
        if ndets is None:
            ndets = len(det_uid)
        if det_uid is None:
            det_uid = np.arange(ndets)
            
        self.cuts = [CutsVector([], nsamps) for _ in range(ndets)]
        self.nsamps = nsamps
        self.sample_offset = sample_offset
        self.det_uid = np.array(det_uid)

    @classmethod 
    def read_from_path(cls, path: str) -> "TODCuts":
        """Parse TODCuts from text format.

        The format contains a header section with metadata followed by detector cuts:
            format = 'TODCuts'
            format_version = 2  
            n_det = 1760
            n_samp = 259864
            samp_offset = 0
            END
               0: (0,192) (55020,55021) (224081,224082)
               1: (0,192) (55020,55021) (224081,224082)
               ...

        Args:
            text_content (str): String containing the cuts data
            
        Returns:
            TODCuts object initialized from the text data
        """
        with open(path, 'r') as f:
            text_content = f.read()

        # Parse header section
        header = {}
        lines = text_content.strip().split('\n')
        for line in lines:
            if line.strip() == 'END':
                break 
            if '=' in line:
                key, value = line.split('=', 1)
                header[key.strip()] = value.strip().strip("'")

        # Create cuts object
        cuts = cls(ndets=int(header['n_det']), 
                  nsamps=int(header['n_samp']),
                  sample_offset=int(header['samp_offset']))

        # Skip header lines including END line
        data_lines = lines[len(header)+1:]
        
        # Parse cuts for each detector
        for line in data_lines:
            if not line.strip():
                continue
            # Parse detector number and cuts
            det_str, cuts_str = line.strip().split(':', 1)
            det_idx = int(det_str)
            
            # Parse cut ranges
            cut_pairs = []
            for cut in cuts_str.strip().split():
                start, end = map(int, cut.strip('()').split(','))
                if start < end:  # Only add valid cut ranges
                    cut_pairs.append([start, end])
                
            cuts.cuts[det_idx] = CutsVector(cut_pairs, cuts.nsamps)
        return cuts

    def copy(self, resample=1, det_uid=None, cut_missing=False):
        nsamps = int(self.nsamps / resample)
        sample_offset = None if self.sample_offset is None else int(self.sample_offset * resample)
        
        if det_uid is None:
            det_uid = self.det_uid
            det_idx = np.arange(len(det_uid))
        else:
            det_idx = np.full(len(det_uid), -1)
            det_map = {uid: i for i, uid in enumerate(self.det_uid)}
            for i, uid in enumerate(det_uid):
                det_idx[i] = det_map.get(uid, -1)
            if not cut_missing and (det_idx == -1).any():
                raise RuntimeError("Specify cut_missing=True to include absent detectors")

        out = self.__class__(det_uid=det_uid, nsamps=nsamps, sample_offset=sample_offset)
        out.cuts = [CutsVector.new_always_cut(out.nsamps) if i < 0 
                   else self.cuts[i].get_resampled(resample) for i in det_idx]
        return out

    def extract(self, sample_offset, nsamps, cut_missing=False):
        delta = sample_offset - self.sample_offset
        
        pad_left = pad_right = np.zeros((0,2), int)
        if delta < 0 or sample_offset + nsamps > self.sample_offset + self.nsamps:
            if not cut_missing:
                raise ValueError("CutsVector superset not permitted unless cut_missing=True")
            if delta < 0:
                pad_left = np.array([(0, -delta)])
            if sample_offset + nsamps > self.sample_offset + self.nsamps:
                pad_right = np.array([(self.nsamps + self.sample_offset - sample_offset, nsamps)])

        cvects = []
        for c in self.cuts:
            # Reshape cuts to (-1, 2) format before subtraction
            if len(c.cuts) > 0:
                cuts_2d = c.cuts.reshape(-1, 2) - delta
            else:
                cuts_2d = np.zeros((0, 2), dtype=int)
                
            c1 = np.vstack((pad_left, cuts_2d, pad_right))
            ce = CutsVector(cuts_in=c1, nsamps=nsamps).get_collapsed()
            cvects.append(ce)
            
        out = self.__class__(det_uid=self.det_uid, nsamps=nsamps, sample_offset=sample_offset)
        out.cuts = cvects
        return out

    def get_mask(self):
        return np.array([not self.is_always_cut(i) for i in range(len(self.cuts))])

    def get_cut(self, det_uid=False):
        cut_idx = (~self.get_mask()).nonzero()[0]
        return self.det_uid[cut_idx] if det_uid else cut_idx

    def get_uncut(self, det_uid=False):
        uncut_idx = self.get_mask().nonzero()[0]
        return self.det_uid[uncut_idx] if det_uid else uncut_idx

    @iterate_dets
    def is_always_cut(self, det):
        if self.nsamps is None:
            raise ValueError('nsamps not set for cuts object.')
        c = self.cuts[det]
        return len(c) == 1 and c[0,0] == 0 and c[0,1] >= self.nsamps

    @iterate_dets
    def set_always_cut(self, det_idx):
        if self.nsamps is None:
            raise ValueError('nsamps not set for cuts object.')
        self.cuts[det_idx] = CutsVector.new_always_cut(self.nsamps)

    @iterate_dets
    def add_cuts(self, det_index, new_cuts, mask=False):
        if mask:
            new_cuts = CutsVector.from_mask(new_cuts)
        new_cuts = np.asarray(new_cuts, dtype=int, order='C')
        # Handle empty cuts
        if len(self.cuts[det_index].cuts) == 0:
            self.cuts[det_index] = CutsVector(new_cuts, self.nsamps)
            return
            
        # Simple union implementation since libactpol.merge_cuts is not available
        merged = np.unique(np.vstack((self.cuts[det_index].cuts.reshape(-1, 2), new_cuts.reshape(-1, 2))), axis=0)
        self.cuts[det_index] = CutsVector(merged, self.nsamps).get_collapsed()

    def merge_tod_cuts(self, tod_cuts, cut_missing=False):
        tod_cuts = tod_cuts.extract(self.sample_offset, self.nsamps, cut_missing=cut_missing)
        assert self.sample_offset == tod_cuts.sample_offset
        
        if len(tod_cuts.det_uid) == 0:
            return
            
        det_map = {uid: i for i, uid in enumerate(tod_cuts.det_uid)}
        for i0, det_uid in enumerate(self.det_uid):
            if det_uid in det_map:
                self.add_cuts(i0, tod_cuts.cuts[det_map[det_uid]])
            elif cut_missing:
                self.set_always_cut(i0)

    def get_complement(self):
        out = self.__class__(det_uid=self.det_uid, nsamps=self.nsamps, sample_offset=self.sample_offset)
        out.cuts = [c.get_complement() for c in self.cuts]
        return out
        
    def buffer(self, nbuf):
        for i, c in enumerate(self.cuts):
            self.cuts[i] = c.get_buffered(nbuf)

    @classmethod
    def for_tod(cls, tod, assign=True):
        obj = cls(tod.data.shape[0], tod.data.shape[1], tod.det_uid, tod.sample_offset)
        if assign:
            obj.cuts = [CutsVector.new_always_cut(obj.nsamps) for _ in range(obj.ndets)]
        return obj


def fill_one_cuts(data: np.ndarray, cuts: np.ndarray, noise: Optional[np.ndarray] = None, 
                 fit_region: int = 10, extrapolate: int = 0) -> None:
    """Fill cut regions in data array using linear fits from surrounding regions.
    
    Parameters:
    -----------
    data : np.ndarray
        1D float32 array containing the data to be processed
    cuts : np.ndarray
        Array of cut regions with shape (n_cuts, 2) specifying [start, end) indices
    noise : np.ndarray, optional
        Optional 1D float32 array of noise values to add to cuts
    fit_region : int
        Number of samples to use for fitting on each side of cut
    extrapolate : int
        Whether to extrapolate when only one side has data (0=False, 1=True)
    
    Notes:
    ------
    This function modifies the data array in-place.
    """
    # Validate input
    if len(cuts) == 0:
        return

    nsamps = len(data)
    noise_i = 0

    # Validate noise array if provided
    if noise is not None and len(cuts) > 0:
        total_cut_samples = sum(cut[1] - cut[0] for cut in cuts)
        if len(noise) < total_cut_samples:
            raise ValueError("Required more noise samples than were provided")

    # Process each cut region
    last_cut = 0
    for i, (start, end) in enumerate(cuts):
        # Skip if cut is beyond data length
        if start >= nsamps:
            break

        # Define regions for fitting
        left_edge = max(last_cut, start - fit_region)  # Don't overlap with previous cut
        right_edge = min(nsamps, end + fit_region)     # Don't go beyond end of data

        # Check for next cut region to avoid overlap
        if i + 1 < len(cuts) and cuts[i+1, 0] < right_edge:
            right_edge = cuts[i+1, 0]

        # Get samples before and after the cut
        left_region = np.arange(left_edge, start)
        right_region = np.arange(end, right_edge)

        # Determine fill method based on available data
        has_left = len(left_region) > 0
        has_right = len(right_region) > 0
        cut_region = np.arange(start, end)
        
        if len(cut_region) == 0:
            continue  # Empty cut region, nothing to do

        # Calculate fill values using linear fit when possible
        if has_left and has_right:
            # We have data on both sides - do linear fit or interpolation
            x_values = np.concatenate([left_region, right_region])
            y_values = np.concatenate([data[left_region], data[right_region]])
            
            # Simple linear fit
            if len(x_values) > 1:
                slope, intercept = np.polyfit(x_values, y_values, 1)
                fill_values = slope * cut_region + intercept
            else:
                fill_values = np.full_like(cut_region, y_values[0], dtype=np.float32)
        
        elif (has_left and len(left_region) >= 2 and extrapolate) or (has_right and len(right_region) >= 2 and extrapolate):
            # Only one side has data but we can extrapolate
            x_fit = left_region if has_left else right_region
            y_fit = data[x_fit]
            
            # Linear extrapolation
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            fill_values = slope * cut_region + intercept

        else:
            # Use constant fill from nearest edge
            if has_left:
                fill_values = np.full_like(cut_region, data[start-1], dtype=np.float32)
            elif has_right:
                fill_values = np.full_like(cut_region, data[end], dtype=np.float32)
            else:
                fill_values = np.zeros_like(cut_region, dtype=np.float32)

        # Add noise if provided
        if noise is not None:
            n_needed = len(cut_region)
            # Calculate noise scaling based on residuals around fit
            noise_scale = 1.0
            if 'slope' in locals() and (has_left or has_right):
                residuals = []
                if has_left:
                    residuals.append(data[left_region] - (slope * left_region + intercept))
                if has_right:
                    residuals.append(data[right_region] - (slope * right_region + intercept))
                
                if residuals:
                    residuals = np.concatenate(residuals)
                    noise_scale = np.std(residuals)
                        
            # Apply scaled noise
            fill_values += noise_scale * noise[noise_i:noise_i + n_needed]
            noise_i += n_needed

        # Ensure continuity at boundaries for smooth transitions
        if has_left:
            fill_values[0] = data[start-1]
        if has_right and len(fill_values) > 0:
            fill_values[-1] = data[end]
            
        # Update the data array in-place
        data[cut_region] = fill_values
        
        # Track last cut end for next iteration
        last_cut = end

def fill_cuts(tod=None, cuts=None, data=None,
              neighborhood=40, do_all=False, filterScaling=1.0,
              extrapolate=False, sample_index=0, no_noise=False):
    """Replace certain samples of a TOD with a straight line plus some
    white noise.  The samples to fill are specified by a TODCuts
    object, which is passed through the cuts= argument, or obtained
    from tod.cuts.  The data array to fill is passed through the
    data= argument, or else tod.data is used.
    The cut samples will be replaced with a straight line and white
    noise, with the linear fit and RMS taken from a "neighborhood" of
    samples on either side of the cut region.  The white noise can be
    modulated by the filterScaling argument, or turned off entirely by
    passing no_noise=True.
    The extrapolate argument affects how cut regions at the beginning
    or end of a timestream are handled; False means that the linear
    fill will have its slope forced to zero.
    """
    # Just passing in tod is enough.
    if cuts is None and tod is not None:
        cuts = tod.cuts
    if data is None and tod is not None:
        data = tod.data
        si = tod.info.sample_index
    else:
        si = sample_index
    
    nsamps = data.shape[-1]
        
    # Check alignment between cuts and data.
    if si != cuts.sample_offset:
        print(f"TOD was loaded from sample {si} but cuts have sample offset {cuts.sample_offset}.")
        print(f"The first {abs(si - cuts.sample_offset)} samples will be handled differently.")
    else:
        assert (data.shape[-1] == cuts.nsamps) or (data.shape[-1] == cuts.nsamps+1)
    
    # Validate detector count matches
    assert(data.shape[0] == len(cuts.det_uid))
    if tod is not None:
        assert(np.all(tod.det_uid == cuts.det_uid))
    
    det_list = range(data.shape[0]) if do_all else cuts.get_uncut()
    
    for deti in det_list:
        mask = cuts.cuts[deti].get_mask()
        offset = cuts.sample_offset - si
        total_mask = np.ones(nsamps, dtype=bool)
        total_mask[max(0, offset):min(nsamps, cuts.nsamps+offset)] = \
            mask[max(0, -offset):min(nsamps-offset, cuts.nsamps)]
        cuts_list = CutsVector.from_mask(total_mask)
        
        # Convert to array of [start, end] pairs
        cuts_array = np.array(cuts_list, dtype=np.int32).reshape(-1, 2)
        
        if no_noise: 
            noise = None
        else:
            # Calculate total number of samples to be cut
            n_cut = sum([end - start for start, end in cuts_array])
            # Generate noise samples
            noise = np.random.normal(size=n_cut) * filterScaling
        
        # Fill cuts for this detector
        fill_one_cuts(data[deti], cuts_array, noise, neighborhood, int(extrapolate))