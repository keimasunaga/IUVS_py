import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits

from PyUVS.data import get_apoapse_files as get_swath_info
from PyUVS.graphics import angle_meshgrid, H_colormap
from PyUVS.variables import data_directory as dataloc
from PyUVS.variables import slit_width_deg

class ApoapseInfo:
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
        self.segment = 'apoapse'
        self.orbit_number = orbit_number
        self.level = level
        self.channel = channel
        swath_info = get_swath_info(self.orbit_number, directory=dataloc, level=self.level, channel=channel)
        self.files = swath_info['files']
        self.n_swaths = swath_info['n_swaths']
        self.swath_number = swath_info['swath_number']
        self.dayside = swath_info['dayside']
        self.beta_flip = swath_info['beta_flip']

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


class ApoapseSwath:
    def __init__(self, hdul, swath_number=0, wv0=121.6, wv_width=2.5):
        self.hdul = hdul
        self.swath_number = swath_number
        self.wv0 = wv0
        self.wv_width = wv_width

    def get_img(self):
        '''
        Retruns an apoapse swath image at a given wavelength integrated over a given wavelength range.

        Parameters
        ----------
        wv0 : float
            Line center wavelength in nm. (i.e., 121.6 for Ly-alpha)
        wv_width : float
            (Half) Width of wavelength in which counts to be intergated.
            (i.e, counts in wv0 ± wv_width are integrated)

        Returns
        -------
        img : array
            An image of a swath.
        '''
        data = self.hdul['primary'].data  # brighntness, dim = (n_integral, n_space, n_spectral)
        counts_3d = self.hdul['detector_dark_subtracted'].data # counts, dim = (n_integral, n_space, n_spectral)
        cal = data/counts_3d
        cal_1d = cal[0][0]
        wv_2d = self.hdul['observation'].data['wavelength'][0] # wavelength in nm, dim = (n_space, n_spectral)
        counts_sum = np.array([[np.sum(counts_1d*(np.abs(wv_1d - self.wv0) < self.wv_width)) for counts_1d, wv_1d in zip(counts_2d, wv_2d)] for counts_2d in counts_3d])
        cal_intp = np.array([np.interp(self.wv0, wv_1d, cal_1d) for wv_1d in wv_2d])
        img = counts_sum*cal_intp[None, :]
        return img

    def get_xygrids(self):
        x, y = angle_meshgrid(self.hdul)
        x += slit_width_deg * self.swath_number
        y = (120 - y) + 60
        return x, y

    def plot(self, ax=None, **kwargs):
        img = self.get_img()
        print(img.shape)

        x, y = self.get_xygrids()

        if ax is None:
            mesh = plt.pcolormesh(x, y, img, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, img, **kwargs)

        return mesh


def get_apoapseinfo(orbit_number, level='l1b', channel='fuv'):
    '''
    Returns an apoapse info object created by the ApoapseInfo class.

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
    apoinfo : ApoapseInfo object
        An apoapse information object.
    '''
    apoinfo = ApoapseInfo(orbit_number, channel=channel)
    return apoinfo


def plot_apoapse_image(orbit_number, wv0=121.6, wv_width=2.5, ax=None, **kwargs):
    '''
    Plots an apoapse image integrated over given wavelength for a given orbit number.

    Parameters
    ----------
    orbit_numer : int
        The orbit number for which to plot the image.
    wv0 : float
            Line center wavelength in nm. (i.e., 121.6 for Ly-alpha)
    wv_width : float
        (Half) Width of wavelength in which counts to be intergated.
        (i.e, counts in wv0 ± wv_width are integrated)

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        A mesh object that pcolormesh returns.
    '''
    # Get apopase information object and filenames
    apoinfo = get_apoapseinfo(orbit_number)
    fnames = apoinfo.files

    for ith_file, iswath_number in enumerate(apoinfo.swath_number):
        hdul = apoinfo.get_hdul(ith_file)
        aposwath = ApoapseSwath(hdul, iswath_number, wv0, wv_width)
        mesh = aposwath.plot(**kwargs)

    ax = plt.gca()
    ax.set_title('Orbit ' + str(orbit_number) + ' Apoapse ' + str(wv0) + ' nm')
    ax.set_xlabel('Spatial angle [deg]')
    ax.set_ylabel('Integrations')

    return mesh

mesh = plot_apoapse_image(7050, cmap=H_colormap(), norm=mpl.colors.LogNorm(vmin=1e-1, vmax=20))
cb = plt.colorbar(mesh)
cb.set_label('Brightness [kR]')
