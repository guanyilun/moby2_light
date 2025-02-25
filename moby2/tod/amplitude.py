import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import random
from scipy.interpolate import splrep, BSpline
import scipy as sp

import moby2
from moby2.scripting import products
from moby2.tod import cuts

import cutslib as cl
import cutslib.glitch as gl
from cutslib.visual import array_plots, get_position, tod3D

import pandas as pd

from scipy.optimize import curve_fit

from astropy.coordinates import SkyCoord



def amp_fit(det, amp, pos_ra, pos_dec, off):

    #det: a Nx2 (ra,dec) array of detector position
    #pos_ra: ra position of the source 
    #pos_dec: dec position of the source


    c1 = SkyCoord(det[:,0], det[:,1], unit="deg")
    c2 = SkyCoord(pos_ra, pos_dec, unit ="deg")
    r_diff = c1.separation(c2).deg
    
    r_flat = r_diff.flatten()

    f = np.loadtxt('/home/eh9397/20230903_beams_s1v3/s19_pa5_f150_night_beam_profile_instant.txt')
    rad = f.T[0] #radius in degrees
    i = f.T[1]
    b = np.interp(r_flat, rad, i)

    val = amp * b + off
    return val


