"""
TOD (Time-Ordered Data) processing package.
"""

from .cuts import get_glitch_cuts, TODCuts, CutsVector
from .tod import TOD, detrend_tod, remove_mean