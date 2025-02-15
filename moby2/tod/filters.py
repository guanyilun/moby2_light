"""
Frequency domain filters for signal processing.

This module provides a collection of frequency-domain filter functions that return real or complex
vectors suitable for direct multiplication by an FFT (Fast Fourier Transform).
"""

from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray


def gen_freqs(n_data: int, t_sample: float) -> NDArray:
    """Generate frequency vector for a specified length and time period.
    
    Args:
        n_data: Number of elements in vector
        t_sample: Sample time
    
    Returns:
        NDArray: Array of frequencies
    """
    dn = 2  # Central frequency sign control (2 for positive, 1 for negative)
    return 1./(n_data * t_sample) * np.hstack((
        np.arange(0., (n_data + dn)//2),
        np.arange(-(n_data + dn)//2 + dn, 0)
    ))


def gen_freqs_tod(tod) -> NDArray:
    """Generate frequencies from time-ordered data.
    
    Args:
        tod: Time-ordered data object with ctime attribute
        
    Returns:
        NDArray: Array of frequencies
    """
    n = tod.ctime.shape[0]
    return gen_freqs(n, (tod.ctime[-1] - tod.ctime[0]) / (n-1))


def power_law_filter(tod, power: float = -2.0, knee: float = 1) -> NDArray:
    """Generate a power law filter.
    
    Args:
        tod: Time-ordered data object
        power: Power law exponent
        knee: Knee frequency
        
    Returns:
        NDArray: Filter coefficients
    """
    f = gen_freqs_tod(tod)
    filt = np.power(np.abs(f), power)
    filt[0] = 0.0
    return filt


def low_freq_wiener_filter(tod, power: float = -2.0, f_knee: float = 1.0) -> NDArray:
    """Generate a low-frequency Wiener filter.
    
    Args:
        tod: Time-ordered data object
        power: Power law exponent
        f_knee: Knee frequency
        
    Returns:
        NDArray: Filter coefficients
    """
    f = gen_freqs_tod(tod)
    s = np.power(np.abs(f)/f_knee, power)
    s[0] = s[1]
    n = np.ones_like(f)
    return s/(s + n)


def rc_filter(tod, fc: float = 2.0) -> NDArray:
    """Generate an RC (resistance-capacitance) filter.
    
    Args:
        tod: Time-ordered data object
        fc: Cutoff frequency
        
    Returns:
        NDArray: Filter coefficients
    """
    f = gen_freqs_tod(tod)
    return 1/np.sqrt(1 + np.power(f/fc, 2))


def sine2_high_pass(
    tod = None,
    fc: float = 1.0,
    df: float = 0.1,
    nsamps: Optional[int] = None,
    sample_time: Optional[float] = None
) -> NDArray:
    """Generate a sine-squared high-pass filter.
    
    Args:
        tod: Time-ordered data object (optional)
        fc: Cutoff frequency
        df: Frequency width of transition region
        nsamps: Number of samples (if tod not provided)
        sample_time: Sample time (if tod not provided)
        
    Returns:
        NDArray: Filter coefficients
    """
    fc, df = np.abs(fc), np.abs(df)
    f = gen_freqs_tod(tod) if tod is not None else gen_freqs(nsamps, sample_time)
    
    filt = np.zeros_like(f)
    filt[np.abs(f) > fc + df/2.] = 1.0
    sel = (np.abs(f) > fc - df/2.) & (np.abs(f) < fc + df/2.)
    filt[sel] = np.sin(np.pi/2./df*(np.abs(f[sel]) - fc + df/2.))**2
    return filt


def sine2_low_pass(
    tod = None,
    fc: float = 1.0,
    df: float = 0.1,
    nsamps: Optional[int] = None,
    sample_time: Optional[float] = None
) -> NDArray:
    """Generate a sine-squared low-pass filter.
    
    Args:
        tod: Time-ordered data object (optional)
        fc: Frequency where power is half (Hz)
        df: Width of filter (Hz)
        nsamps: Number of samples (if tod not provided)
        sample_time: Sample time (if tod not provided)
        
    Returns:
        NDArray: Filter coefficients
    """
    fc, df = np.abs(fc), np.abs(df)
    f = gen_freqs_tod(tod) if tod is not None else gen_freqs(nsamps, sample_time)
    
    filt = np.zeros_like(f)
    filt[np.abs(f) < fc - df/2.] = 1.0
    sel = (np.abs(f) > fc - df/2.) & (np.abs(f) < fc + df/2.)
    filt[sel] = np.sin(np.pi/2*(1 - 1/df*(np.abs(f[sel]) - fc + df/2.)))**2
    return filt


def high_pass_butterworth(
    tod,
    fc: float = 1.0,
    order: int = 1,
    gain: float = 1.0
) -> NDArray:
    """Generate a Butterworth high-pass filter.
    
    Args:
        tod: Time-ordered data object
        fc: Cutoff frequency
        order: Order of the filter
        gain: Filter gain
        
    Returns:
        NDArray: Filter coefficients
    """
    f = 1j * gen_freqs_tod(tod) / fc
    filt = np.ones(len(f), dtype=complex)
    
    for k in range(1, order + 1):
        sk = np.exp(1j * (2*k + order - 1) * np.pi / (2*order))
        filt = filt * f / (1 - f*sk)
    
    return np.abs(gain * filt)


def low_pass_butterworth(
    tod,
    fc: float = 1.0,
    order: int = 1,
    gain: float = 1.0
) -> NDArray:
    """Generate a Butterworth low-pass filter.
    
    Args:
        tod: Time-ordered data object
        fc: Cutoff frequency
        order: Order of the filter
        gain: Filter gain
        
    Returns:
        NDArray: Filter coefficients
    """
    f = 1j * gen_freqs_tod(tod) / fc
    filt = np.ones(len(f), dtype=complex)
    
    for k in range(1, order + 1):
        sk = np.exp(1j * (2*k + order - 1) * np.pi / (2*order))
        filt = filt / (f - sk)
    
    return np.abs(gain * filt)


def gaussian_filter(
    tod,
    time_sigma: Optional[float] = None,
    frec_sigma: Optional[float] = None,
    gain: float = 1.0,
    f_center: float = 0.0
) -> NDArray:
    """Generate a Gaussian filter.
    
    Args:
        tod: Time-ordered data object
        time_sigma: Time domain standard deviation
        frec_sigma: Frequency domain standard deviation
        gain: Filter gain
        f_center: Center frequency
        
    Returns:
        NDArray: Filter coefficients
    """
    if time_sigma is not None and frec_sigma is not None:
        print("WARNING: cannot specify both time and frequency sigmas. Using time_sigma.")
        
    if time_sigma is not None:
        sigma = 1.0 / (2*np.pi*time_sigma)
    elif frec_sigma is not None:
        sigma = frec_sigma
    else:
        sigma = 1.0

    f = gen_freqs_tod(tod)
    return gain * np.exp(-0.5*(np.abs(f) - f_center)**2/sigma**2)