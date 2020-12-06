"""
IUVS l2b data module

author: K. Masunaga (kei.masunaga@lasp.colorado.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
import os

from PyUVS.graphics import angle_meshgrid, H_colormap, NO_colormap
from PyUVS.variables import slit_width_deg
from PyUVS.data import get_apoapse_files as get_swath_info
from PyUVS.variables import data_directory as dataloc
from chaffin.paths import anc_dir

class ApoapseInfo:
    def __init__(self, orbit_number, level='l2b', channel='fuv'):
        self.segment = 'apoapse'
        self.orbit_number = orbit_number
        self.level = level
        self.channel = channel
        swath_info = get_swath_info(self.orbit_number, directory=dataloc, level=self.level, channel=channel)
        self.files = swath_info['files']
        self.n_files = int(len(self.files))
        self.n_swaths = swath_info['n_swaths']
        self.swath_number = swath_info['swath_number']
        self.dayside = swath_info['dayside']
        self.beta_flip = swath_info['beta_flip']
        self.sDt = swath_info['sDt']
        self.eDt = swath_info['eDt']

    def get_hdul(self, idx_swath_number):
        hdul = fits.open(self.files[idx_swath_number])
        return hdul


class ApoapseSwath:
    def __init__(self, hdul, linename='Lya', swath_number=0):
        self.hdul = hdul
        self.swath_number = swath_number
        self.linename = linename

    def get_img(self, flatfield_correct=True):
        data = self.hdul['primary'].data
        if self.linename == 'Lya':
            self.img = data[:,:,0]
        if self.linename == 'OI1304':
            self.img = data[:,:,1]
        if self.linename == 'OI1356':
            self.img = data[:,:,2]
        if self.linename == 'CII1336':
            self.img = data[:,:,3]
        if self.linename == 'Solar':
            self.img = data[:,:,4]
        if flatfield_correct:
            self.apply_flatfield()
        return self.img

    def apply_flatfield(self):
        spatial_binning = np.concatenate([self.hdul['Binning'].data['SPAPIXLO'][0],
                                         [self.hdul['Binning'].data['SPAPIXHI'][0][-1]+1]])
        if self.linename == 'Lya':
            #slit_flatfield = np.load(os.path.join(anc_dir, 'kei_flatfield_polynomial_25Nov2020.npy'))
            slit_flatfield = np.load(os.path.join(anc_dir, 'bad_flatfield_23Sep2020.npy'))
        else:
            warnings.warn('Binning is not periapsis--- using a very rough FUV flatfield.')
            slit_flatfield = np.load(os.path.join(anc_dir, 'bad_flatfield_23Sep2020.npy'))
        flatfield = np.array([np.mean(slit_flatfield[p0:p1]) for p0,p1 in zip(spatial_binning[:-1], spatial_binning[1:])])
        nint = self.img.shape[0]
        for iint in range(nint):
            self.img[iint, :] /= flatfield

    def get_xygrids(self):
        x, y = angle_meshgrid(self.hdul)
        x += slit_width_deg * self.swath_number
        y = (120 - y) + 60
        return x, y

    def get_cmap(self):
        if self.linename == 'Lya':
            cmap = H_colormap()
        if self.linename == 'OI1304':
            cmap = NO_colormap()
        if self.linename == 'OI1356':
            pass
        if self.linename == 'CII1336':
            pass
        if self.linename == 'Solar':
            pass
        return cmap

    def plot(self, ax=None, flatfield_correct=True, **kwargs):
        img = self.get_img(flatfield_correct=flatfield_correct)
        x, y = self.get_xygrids()

        if ax is None:
            mesh = plt.pcolormesh(x, y, img, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, img, **kwargs)

        return mesh


def get_apoapseinfo(orbit_number, level='l2b', channel='fuv'):
    """
    Returns an apoapse info object created by the ApoapseInfo class.

    Parameters
    ----------
    orbit_numer : int
        The orbit number for which to obtain the infomation.
    level : str
        Data level. Default is l2b.
        (l2b is only tested for the moment. Have to check later if l1b also works.)
    channel : str
        MUV and/or FUV channel.
        (fuv is only tested for the moment. Have to check later if muv also works.)
    Returns
    -------
    apoinfo : ApoapseInfo object
        An apoapse information object.
    """
    apoinfo = ApoapseInfo(orbit_number, channel=channel)
    return apoinfo


def plot_apoapse_image(orbit_number, linename, ax=None, **kwargs):
    """
    Plots an apoapse image for a given orbit number.

    Parameters
    ----------
    orbit_numer : int
        The orbit number for which to plot the image.
    linename : str
        Emission line name for which to plot the image.
    ax : matplotlib Artist
        The axis object in which to draw the image.

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        A mesh object that pcolormesh returns.
    """
    # Get apopase information object and filenames
    apoinfo = get_apoapseinfo(orbit_number)
    fnames = apoinfo.files

    for iswath, ifname in enumerate(fnames):
        hdul = apoinfo.get_hdul(iswath)
        aposwath = ApoapseSwath(hdul, linename, iswath)
        mesh = aposwath.plot(**kwargs)

    ax = plt.gca()
    ax.set_title('Orbit ' + str(orbit_number) + ' Apoapse ' + linename)
    ax.set_xlabel('Spatial angle [deg]')
    ax.set_ylabel('Integrations')

    return mesh


class SzaGeo:
    def __init__(self, hdul, swath_number=0):
        self.hdul = hdul
        self.swath_number = swath_number
        self.data = hdul['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE']
        self.mrh_alt = hdul['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT']

    def get_xygrids(self):
        x, y = angle_meshgrid(self.hdul)
        x += slit_width_deg * self.swath_number
        y = (120 - y) + 60
        return x, y

    def plot(self, ax=None, **kwargs):
        img = np.where(self.mrh_alt[:,:,4] == 0, self.data, np.nan)
        x, y = self.get_xygrids()
        if ax is None:
            mesh = plt.pcolormesh(x, y, img, vmin=0, vmax=180, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, img, vmin=0, vmax=180, **kwargs)

        return mesh

def plot_apoapse_sza_geo(orbit_number, ax=None, **kwargs):
    apoinfo = get_apoapseinfo(orbit_number)
    fnames = apoinfo.files
    for iswath, ifname in enumerate(fnames):
        hdul = apoinfo.get_hdul(iswath)
        szageo = SzaGeo(hdul, iswath)
        mesh = szageo.plot(cmap=plt.get_cmap('magma_r', 18))

    ax = plt.gca()
    ax.set_title('Orbit ' + str(orbit_number) + ' SZA')
    ax.set_xlabel('Spatial angle [deg]')
    ax.set_ylabel('Integrations')
    return mesh


class LocalTimeGeo:
    def __init__(self, hdul, swath_number=0):
        self.hdul = hdul
        self.swath_number = swath_number
        self.data = hdul['PixelGeometry'].data['PIXEL_LOCAL_TIME']
        self.mrh_alt = hdul['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT']

    def get_xygrids(self):
        x, y = angle_meshgrid(self.hdul)
        x += slit_width_deg * self.swath_number
        y = (120 - y) + 60
        return x, y

    def plot(self, ax=None, **kwargs):
        # Check what the third dimension of mrh_alt!!
        img = np.where(self.mrh_alt[:,:,4] == 0, self.data, np.nan)
        x, y = self.get_xygrids()
        if ax is None:
            mesh = plt.pcolormesh(x, y, img, vmin=0, vmax=24, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, img, vmin=0, vmax=24, **kwargs)

        return mesh

def plot_apoapse_lt_geo(orbit_number, ax=None, **kwargs):
    apoinfo = get_apoapseinfo(orbit_number)
    fnames = apoinfo.files
    for iswath, ifname in enumerate(fnames):
        hdul = apoinfo.get_hdul(iswath)
        ltgeo = LocalTimeGeo(hdul, iswath)
        mesh = ltgeo.plot(cmap=plt.get_cmap('twilight_shifted', 24))

    ax = plt.gca()
    ax.set_title('Orbit ' + str(orbit_number) + ' Local Time')
    ax.set_xlabel('Spatial angle [deg]')
    ax.set_ylabel('Integrations')
    return mesh

class LatLonGeo:
    def __init__(self, hdul, swath_number=0):
        self.hdul = hdul
        self.swath_number = swath_number
        self.lat = hdul['PixelGeometry'].data['PIXEL_CORNER_LAT']
        self.lon = hdul['PixelGeometry'].data['PIXEL_CORNER_LON']
        self.mrh_alt = hdul['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT']

    def get_xygrids(self):
        x, y = angle_meshgrid(self.hdul)
        x += slit_width_deg * self.swath_number
        y = (120 - y) + 60
        return x, y

    def plot_lat(self, ax=None, **kwargs):
        # Check what the third dimension of mrh_alt!!
        img = np.where(self.mrh_alt[:,:,4] == 0, self.lat[:,:,4], np.nan)
        x, y = self.get_xygrids()
        if ax is None:
            mesh = plt.pcolormesh(x, y, img, vmin=-90, vmax=90, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, img, vmin=-90, vmax=90, **kwargs)

        return mesh

    def plot_lon(self, ax=None, **kwargs):
        # Check what the third dimension of mrh_alt!!
        img = np.where(self.mrh_alt[:,:,4] == 0, self.lon[:,:,4], np.nan)
        x, y = self.get_xygrids()
        if ax is None:
            mesh = plt.pcolormesh(x, y, img, vmin=0, vmax=360, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, img, vmin=0, vmax=360, **kwargs)

        return mesh


def test():
    plt.close('all')
    fig = plt.figure(figsize=(24,6))

    ax1 = fig.add_subplot(121)
    mesh1 = plot_apoapse_image(3780, 'Lya', ax1, vmin=0, vmax=10)
    cb1 = plt.colorbar(mesh1)
    cb1.set_label('Brightness [kR]')

    ax2 = fig.add_subplot(122)
    mesh2 = plot_apoapse_image(3780, 'OI1304', ax2, vmin=0, vmax=1.5)
    cb2 = plt.colorbar(mesh2)
    cb2.set_label('Brightness [kR]')

    plt.show()
