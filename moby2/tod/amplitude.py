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


def excise_peaks(snip_data, buff):
    new_snip = snip_data.copy()
    for i in range(len(snip_data)):
        det_vals = snip_data[i]
        try:
            #buffer around peak value

            kernel_size = 8
            kernel = np.ones(kernel_size) / kernel_size

            kernel_smooth = 50
            kernel_2 = np.ones(kernel_smooth) / kernel_smooth

            mean_vals1 = np.convolve(det_vals, kernel, mode='same')
            smooth_vals = np.convolve(mean_vals1, kernel_2, mode='same')
            vals_use = mean_vals1-smooth_vals
            std_use = np.std(vals_use[50:-50])

            peak_idx = sp.signal.find_peaks(vals_use, prominence = std_use*4)[0][0]

            x_interp = [i for i in range(peak_idx-buff, peak_idx+buff)]
            y_interp = np.interp(x_interp, [peak_idx-(buff+1), peak_idx+buff], [mean_vals1[peak_idx-(buff+1)], mean_vals1[peak_idx+buff]])
 
            new_snip[i][x_interp] = y_interp
        except Exception as e:
            print('failed to excise peak')
            print(e)
            continue
    return new_snip

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


