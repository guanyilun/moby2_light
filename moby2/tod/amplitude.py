
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import random
from scipy.interpolate import splrep, BSpline
import scipy as sp

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
    dir = '.'
    f = np.loadtxt(dir+'/s19_pa5_f150_night_beam_profile_instant.txt')
    rad = f.T[0] #radius in degrees
    i = f.T[1]
    b = np.interp(r_flat, rad, i)
    
    val = amp * b + off
    return val
    
def dplanck(f, T):
    a=2
    """The derivative of the planck spectrum with respect to temperature, evaluated
    at frequencies f and temperature T, in units of Jy/sr/K."""
    c = 299792458.0
    h = 6.62606957e-34
    k = 1.3806488e-23
    x = h*f/(k*T)
    dIdT  = 2*x**4 * k**3*T**2/(h**2*c**2) / (4*np.sinh(x/2)**2) * 1e26
    return dIdT

def calibrate(vals):
    #convert to mJy/sr
    if type(vals) != np.ndarray:
        vals = np.array(vals)
    
    vals *= dplanck(149e9, 2.72548)/1e3
    
    nsr = 211.47
    beam_cal = nsr * 1e-9
    
    vals *= beam_cal
    return vals

def get_all_amps(tod_name_sim, tod_sim, snippets, amp, halflife, dir):
    source_amps = []
    source_ctimes = []
    
    for i in range(len(snippets)):
        snip_idx = i
        snip = snippets[snip_idx]
        slice_inds = snip.tslice
        
        excised = excise_peaks(snip.data, 15)
        snip_data = snip.data - excised
        d_final = snip_data.flatten()
        d_final = calibrate(d_final)
        
        pos_csv = 'pos_{}_amp{}_h{}_tstart_{}_tend_{}_plus1.csv'.format(tod_name_sim, amp, halflife, slice_inds.start, slice_inds.stop)
        pos_df = pd.read_csv('{}/{}'.format(dir, pos_csv))
        
        pos_df.ra *= 180/np.pi
        pos_df.dec *= 180/np.pi
        
        n_dets = len(snip_data)
        n_samps = len(snip_data[0])
        peaks = []
        
        #save the ra/dec at the peak for each detector
        ra_peaks = np.zeros(n_dets)
        dec_peaks = np.zeros(n_dets)
        
        #save 3D array with det = [N_dets, N_samps, RA/DEC]
        det = np.zeros((n_dets,n_samps,2))
    
        for j in range (n_dets):
        
    
            pos_df_t = pos_df.loc[pos_df['det_id'] == np.asarray(tod_sim.det_uid_original)[snip.det_uid[j]]]
            ras = pos_df_t['ra'].values
            decs = pos_df_t['dec'].values
        
            val_peak = np.amax(snip_data[j])
            peaks.append(val_peak)
            id_peak = np.where((snip_data[j]) == val_peak)[0][0]
            ra_peaks[j] = ras[id_peak]
            dec_peaks[j] = decs[id_peak]
    
            for n in range(n_samps):
                  det[j][n][0] = ras[n]
                  det[j][n][1] = decs[n]
    
    
        # get guess parameters for fit parameters (amplitude, ra, dec, offset)
        
        guess_amp = np.amax(peaks)
        
        id_max_peak = np.where(peaks == guess_amp)[0]
        # guess_ra = ra_peaks[id_max_peak][0]
        # guess_dec = dec_peaks[id_max_peak][0]
        
        guess_ra = np.mean(pos_df['ra'].values)
        guess_dec = np.mean(pos_df['dec'].values)
        
        guess_off = np.mean(snip_data[:100])
        
        det_final = det.reshape(-1, det.shape[-1])

        
        popt, pcov = curve_fit(amp_fit, det_final, d_final, p0=[guess_amp, guess_ra, guess_dec, guess_off], sigma=np.ones(len(d_final))*100, absolute_sigma=True)
        
        fit_amp =  popt[0]
        fit_ra = popt[1]
        fit_dec = popt[2]
        fit_off = popt[3]
        
        avg_time = np.average([tod_sim.ctime[[snip.tslice.start]][0], tod_sim.ctime[[snip.tslice.stop]][0]])
        source_amps.append(fit_amp)
        source_ctimes.append(avg_time)
    return source_amps, source_ctimes


def flare (time, amp, h):
    k = np.log(2)/h

    val = amp*np.exp(-k*(time))
    return val

# CALIBRATION
def dplanck(f, T):
    a=2
    """The derivative of the planck spectrum with respect to temperature, evaluated
    at frequencies f and temperature T, in units of Jy/sr/K."""
    c = 299792458.0
    h = 6.62606957e-34
    k = 1.3806488e-23
    x = h*f/(k*T)
    dIdT  = 2*x**4 * k**3*T**2/(h**2*c**2) / (4*np.sinh(x/2)**2) * 1e26
    return dIdT

def calibrate(vals):
    #convert to mJy/sr
    if type(vals) != np.ndarray:
        vals = np.array(vals)
    vals *= dplanck(149e9, 2.72548)/1e3
    nsr = 211.47
    beam_cal = nsr * 1e-9

    vals *= beam_cal
    return vals
