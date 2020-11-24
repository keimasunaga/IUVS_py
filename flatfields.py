import astropy.io.fits as fits
from scipy.io import readsav
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from variables import saveloc
from variables import iuvdataloc as dataloc
from PyUVS.data import get_files, beta_flip
from iuvtools.time import get_timeDt
from iuvtools.data import get_wv

def get_star_files(orbit_number, directory, segment='star', level='l1b', channel='fuv'):
    """
    Convenience function for apoapse data. In addition to returning file paths to the data, it determines how many
    swaths were taken, which swath each file belongs to since there are often 2-3 files per swath, whether the MCP
    voltage settings were for daytime or nighttime, the mirror step between integrations, and the beta-angle orientation
    of the APP.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    directory : str
        Absolute path to your IUVS level 1B data directory which has the orbit blocks, e.g., "orbit03400, orbit03500,"
        etc.

    Returns
    -------
    swath_info : dict
        A dictionary containing filepaths to the requested data files, the number of swaths, the swath number
        for each data file, whether or not the file is a dayside file, and whether the APP was beta-flipped
        during this orbit.

    """

    # get list of FITS files for given orbit number
    files, n_files = get_files(orbit_number, directory=directory, level=level, segment=segment, channel=channel,
                               count=True)

    # set initial counters
    n_swaths = 0
    prev_ang = 999

    # arrays to hold final file paths, etc.
    filepaths = []
    daynight = []
    swath = []
    sDt = []
    eDt = []
    flipped = 'unknown'

    # loop through files...
    for i in range(n_files):

        # open FITS file
        hdul = fits.open(files[i])

        # skip single integrations, they are more trouble than they're worth
        #if hdul['primary'].data.ndim == 2:
        #    continue

        # determine if beta-flipped
        if flipped == 'unknown':
            flipped = beta_flip(hdul)

        # store filepath
        filepaths.append(files[i])

        # determine if dayside or nightside
        if hdul['observation'].data['mcp_volt'] > 700:
            daynight.append(False)
        else:
            daynight.append(True)

        # calcualte mirror direction
        mirror_dir = np.sign(hdul['integration'].data['mirror_deg'][-1] - hdul['integration'].data['mirror_deg'][0])
        if prev_ang == 999:
            prev_ang *= mirror_dir

        # check the angles by seeing if the mirror is still scanning in the same direction
        ang0 = hdul['integration'].data['mirror_deg'][0]
        if ((mirror_dir == 1) & (prev_ang > ang0)) | ((mirror_dir == -1) & (prev_ang < ang0)):
            # increment the swath count
            n_swaths += 1

        # store swath number
        swath.append(n_swaths - 1)

        # change the previous angle comparison value
        prev_ang = hdul['integration'].data['mirror_deg'][-1]

        # start and end datetime for each swath (Added by K. Masunaga)
        timeDt = get_timeDt(hdul)
        sDt.append(timeDt[0])
        eDt.append(timeDt[-1])
    # make a dictionary to hold all this shit
    swath_info = {
        'files': np.array(filepaths),
        'n_swaths': n_swaths,
        'swath_number': np.array(swath),
        'dayside': np.array(daynight),
        'beta_flip': flipped,
        'sDt': sDt,
        'eDt': eDt
    }

    # return the dictionary
    return swath_info


class StarInfo:
    '''
    Initiate ApoapseInfo object that collects information about apoapse observation of a given orbit number.

    Parameters
    ----------
    orbit_numer : int
        The orbit number for which to obtain the infomation.
    level : str
        Data level. Defaults to l1b.
    channel : str
        MUV and/or FUV channel. Defaults to fuv.

    Returns
    -------
    ApoapseInfo object
    '''

    def __init__(self, orbit_number, level='l1b', channel='fuv'):
        self.segment = 'star'
        self.orbit_number = orbit_number
        self.level = level
        self.channel = channel
        swath_info = get_star_files(self.orbit_number, directory=dataloc, level=self.level, channel=channel)
        self.files = swath_info['files']
        self.n_files = int(len(self.files))
        self.n_swaths = swath_info['n_swaths']
        self.swath_number = swath_info['swath_number']
        self.dayside = swath_info['dayside']
        self.beta_flip = swath_info['beta_flip']
        self.sDt = swath_info['sDt']
        self.eDt = swath_info['eDt']

    def get_hdul(self, ith_file):
        '''
        Returns header data units list (hdul) (i.e., open fits file)

        Parameters
        ----------
        ith_file : int
            hdul of ith file in self.files to be returned
            Note: it is not the same as self.swath_number
                  because swath number may be [0, 0, 1, 1, ...] due to dayside/nightside seprated files.

        Returns
        -------
        hdul:
            open fits file
        '''
        hdul = fits.open(self.files[ith_file])
        return hdul



class FlatFieldLya:
    def __init__(self, fname, orbit_number, wv0=121.6, wv_width=2.5):
        self.fname = fname
        self.wv0 = wv0
        self.wv_width = wv_width
        wv, idx_near_wv0 = self.get_wvinfo(orbit_number)
        self.wv = wv[idx_near_wv0]
        norm, norm_stdv, spa_cent = self.get_norm_data()
        self.norm = norm[:, idx_near_wv0]
        self.norm_stdv = norm_stdv[:, idx_near_wv0]
        self.spa_cent = spa_cent

    def get_norm_data(self):
        filepath = os.path.join(saveloc, 'misc_items/flatfield/'+self.fname)
        sav = readsav(filepath)
        spa_centroid = sav['spa_centroid']
        normed_tot = sav['normed_tot']
        normed_stdv_tot = sav['normed_stdv_tot']
        return normed_tot, normed_stdv_tot, spa_centroid

    def get_wvinfo(self, orbit_number):
        starinfo = StarInfo(orbit_number)
        hdul = starinfo.get_hdul(0)   ## Check if it's ok to use the first swath
        wv = get_wv(hdul)
        wv_avg = np.mean(wv, axis=0)
        idx_near_wv0 = np.where((wv_avg>self.wv0-self.wv_width) & (wv_avg<self.wv0+self.wv_width))[0]
        hdul.close()
        return wv_avg, idx_near_wv0

    def fill_nan_near_wv0(self, wvlim=[120.5, 123]):
        self.norm[:,(self.wv>wvlim[0])&(self.wv<wvlim[1])] = np.nan
        self.norm_stdv[:,(self.wv>wvlim[0])&(self.wv<wvlim[1])] = np.nan

    def plot_spa_dist(self, ax, subplots=False):
        if subplots:
            pass
        else:
            evenly_spaced_interval = np.linspace(0, 1, len(self.wv))
            colors = [cm.RdYlBu_r(x) for x in evenly_spaced_interval]
            [ax.errorbar(self.spa_cent, self.norm[:,i], yerr=self.norm_stdv[:,i], linestyle = 'None', marker='o', color=colors[i]) for i in range(len(self.wv))]

    def append_data(self, other):
        self.norm = np.append(self.norm, other.norm, axis=0)
        self.norm_stdv = np.append(self.norm_stdv, other.norm_stdv, axis=0)
        self.spa_cent = np.append(self.spa_cent, other.spa_cent)
