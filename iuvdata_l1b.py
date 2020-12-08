import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
import spiceypy as spice

from common.tools import quaternion_rotation
from PyUVS.data import get_apoapse_files as get_swath_info
from PyUVS.graphics import angle_meshgrid, H_colormap
from PyUVS.variables import data_directory as dataloc
from PyUVS.variables import slit_width_deg
from chaffin.integration import fit_line


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

    def __init__(self, orbit_number, level='l1b', channel='fuv', product_type='production'):
        self.segment = 'apoapse'
        self.orbit_number = orbit_number
        self.level = level
        self.channel = channel
        self.product_type = product_type
        swath_info = get_swath_info(self.orbit_number, directory=dataloc, level=self.level, channel=channel, product_type=product_type)
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


class ApoapseSwath:
    '''
    ApoapseSwath object hanles primary data in a single apoapse swath file (i.e., hdul).

    Parameters
    ----------
    hdul : Open fits file.
        Header Data Units list opend by fits.open().
    swath_number : int
        Swath_number of the hdul data for the orbit. Defaults to 0.
    wv0 : float
        Line center wavelength in nm. (i.e., 121.6 for Ly-alpha)
    wv_width : float
        (Half) Width of wavelength in which counts to be intergated.
        (i.e, counts in wv0 ± wv_width are integrated)

    Returns
    -------
    ApoapseSwath Object
    '''
    def __init__(self, hdul, swath_number=0, wv0=121.6, wv_width=2.5, set_img=True, counts=False):
        self.hdul = hdul
        self.swath_number = swath_number
        self.wv0 = wv0
        self.wv_width = wv_width
        if set_img:
            self.img = self.fit_line()#self.get_img(counts=counts)
            self.xgrids, self.ygrids = self.get_xygrids()
        else:
            self.img = None
            self.xgrids, self.ygrids = None, None

    def get_img(self, counts=False):
        '''
        Retruns an apoapse swath image at a given wavelength integrated over a given wavelength range.

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
        if counts:
            img = counts_sum
        else:
            img = counts_sum*cal_intp[None, :]
        return img

    def fit_line(self, flatfield_correct=False, correct_muv=False):
        return fit_line(self.hdul, self.wv0, flatfield_correct=flatfield_correct, correct_muv=correct_muv)

    def get_xygrids(self):
        x, y = angle_meshgrid(self.hdul)
        x += slit_width_deg * self.swath_number
        y = (120 - y) + 60
        return x, y

    def plot(self, ax=None, **kwargs):
        if ax is None:
            mesh = plt.pcolormesh(self.xgrids, self.ygrids, self.img, **kwargs)
        else:
            mesh = ax.pcolormesh(self.xgrids, self.ygrids, self.img, **kwargs)
        return mesh

    def sub_obj(self, other):
        obj = ApoapseSwath(hdul=None, swath_number=self.swath_number, wv0=self.wv0, wv_width=self.wv_width, set_img=False)
        img_sub = self.img - other.img
        obj.img = img_sub
        obj.xgrids, obj.ygrids = self.xgrids, self.ygrids
        return obj

def get_apoapseinfo(orbit_number, level='l1b', channel='fuv', product_type='production'):
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
    apoinfo = ApoapseInfo(orbit_number, channel=channel, product_type=product_type)
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
    for ith_file, iswath_number in enumerate(apoinfo.swath_number):
        hdul = apoinfo.get_hdul(ith_file)
        aposwath = ApoapseSwath(hdul, iswath_number, wv0, wv_width)
        mesh = aposwath.plot(**kwargs)

    ax = plt.gca()
    ax.set_title('Orbit ' + str(orbit_number) + ' Apoapse ' + str(wv0) + ' nm')
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
    for ith_file, iswath_number in enumerate(apoinfo.swath_number):
        hdul = apoinfo.get_hdul(ith_file)
        szageo = SzaGeo(hdul, iswath_number)
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
    for ith_file, iswath_number in enumerate(apoinfo.swath_number):
        hdul = apoinfo.get_hdul(ith_file)
        ltgeo = LocalTimeGeo(hdul, iswath_number)
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

class FieldAngleGeo:
    def __init__(self, hdul, swath_number=0):
        # Read hdul and variables
        self.hdul = hdul
        self.swath_number = swath_number
        self.et = hdul['Integration'].data['et']
        self.pixel_uvec_from_sc = hdul['PixelGeometry'].data['PIXEL_VEC'] ## pixel unit vector from sc
        self.los_length = hdul['PixelGeometry'].data['PIXEL_CORNER_LOS']
        self.sc_pos_iau = hdul['SpacecraftGeometry'].data['V_SPACECRAFT']
        self.mrh_alt = hdul['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT']
        # Calc pixel vector
        self.pixel_vec_from_sc_iau = self.pixel_uvec_from_sc * self.los_length[:, None, :, :]
        self.pixel_vec_from_pla_iau = self.sc_pos_iau[:, :, None, None] + self.pixel_vec_from_sc_iau
        self._to_mso(self.pixel_vec_from_pla_iau)
        self.data = None

    def _to_mso(self, pixel_vec_from_pla_iau):
        mats = [spice.pxform('IAU_MARS', 'MAVEN_MSO', iet) for iet in self.et]
        q = [spice.m2q(imats) for imats in mats]
        self.pixel_vec_from_pla_mso = np.array([[quaternion_rotation(q[it], pixel_vec_from_pla_iau[it, :, ibin, 4]) for ibin in range(pixel_vec_from_pla_iau.shape[2])] for it in range(pixel_vec_from_pla_iau.shape[0])])
        length_pixel_vec_from_pla_mso = np.sqrt(self.pixel_vec_from_pla_mso[:,:,0]**2 + self.pixel_vec_from_pla_mso[:,:,1]**2 + self.pixel_vec_from_pla_mso[:,:,2]**2)
        self.pixel_uvec_from_pla_mso = self.pixel_vec_from_pla_mso/length_pixel_vec_from_pla_mso[:,:, None]

    def calc_cone_angle(self, field_mso):
        if field_mso is not None:
            self.field_mso = field_mso
            if np.size(self.field_mso) == 3:
                self.data = np.degrees(np.arccos(np.dot(self.pixel_uvec_from_pla_mso, self.field_mso)/np.sqrt(np.dot(self.field_mso, self.field_mso))))
            else:
                self.data = np.array([np.degrees(np.arccos(np.dot(self.pixel_uvec_from_pla_mso[it], self.field_mso[it])/np.sqrt(np.dot(self.field_mso[it], self.field_mso[it])))) for it in range(self.pixel_uvec_from_pla_mso.shape[0])])
        else:
            self.field_mso = None

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


def test():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    mesh = plot_apoapse_image(7050, ax=ax1, cmap=H_colormap(), norm=mpl.colors.LogNorm(vmin=1e-1, vmax=20))
    cb = plt.colorbar(mesh)
    cb.set_label('Brightness [kR]')

    ax2 = fig.add_subplot(212)
    plot_apoapse_lt_geo(7050, ax=ax2)
    plt.show()
