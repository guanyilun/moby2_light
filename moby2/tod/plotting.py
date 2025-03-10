from matplotlib import pyplot as plt
import numpy as np


def array_plot(snippet, colour_affected_det = 'red', colour_other_det = '#9900ff',
               display = 'show', save_name = 'newfig.png', sim = False, tod_sim = None):
  
    '''
    Plots the detector array indicating the affected detectors from the glitch.

    Input: snippet: snippet of glitch computed with moby2.tod.glitch.affected_snippets_from_cv,
    colour_affected_det: colour to make the affected detectors, colour_other_det: colour to make
    the other detectors, display: either 'show' or 'save' depending on if you would like to only
    see the plot or also save it, save_name: name including path to save the plot, sim: Is this 
    a simulation? True or False, tod_sim: the TOD of the simulation loaded with from_npz_sims.
    Output: the focal plane plot
    '''

    #get the detector positions for all detectors
    array_x = snippet.info.array_data['sky_x']
    array_y = snippet.info.array_data['sky_y']

    #get the positions for detectors affected by the glitch
    if sim:
      x = array_x[np.asarray(tod_sim.det_uid_original)[snippet.det_uid]]
      y = array_y[np.asarray(tod_sim.det_uid_original)[snippet.det_uid]]
    else:
      x = array_x[snippet.det_uid]
      y = array_y[snippet.det_uid]

    plt.ioff()
    plt.figure(figsize=(6, 6))
    plt.scatter(array_x, array_y, c = colour_other_det, s=120)
    plt.scatter(x, y, c = colour_affected_det, s=120)

    if display == 'show':
        plt.ion()
        plt.show()
    elif display == 'save':
        plt.savefig(save_name)
        plt.close()


def tod_plot(snippet, colour = 'purple', alpha = 0.2,
               display = 'show', save_name = 'newfig.png'):
    '''
    Plots the TOD of the glitch.

    Input: snippet: snippet of glitch computed with moby2.tod.glitch.affected_snippets_from_cv,
    colour: colour to plot the TODs, alpha: the opacity to plot each detector TOD,
    display: either 'show' or 'save' depending on if you would like to only
    see the plot or also save it, save_name: name including path to save the plot.
    Output: the TOD plot
    '''

    #demean and detrend the TOD
    s_t = snippet.demean()
    s_t = s_t.deslope()
    data_t = s_t.data

    plt.ioff()
    plt.figure(figsize=(6, 4))
    plt.plot(data_t.T, color = colour, alpha = alpha)
    plt.xlabel('Time Sample', fontsize = 18)
    plt.ylabel('Amplitude [$\\mu$K]', fontsize = 18)

    if display == 'show':
        plt.ion()
        plt.show()
    elif display == 'save':
        plt.savefig(save_name)
        plt.close()
