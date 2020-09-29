import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

from variables import saveloc
from PyUVS.geometry import beta_flip
from PyUVS.graphics import H_colormap
from iuvdata import ApoapseInfo, ApoapseSwath
from common.tools import RunTime
from PyUVS.time import find_segment_et
from PyUVS.spice import load_iuvs_spice
from iuvtools.time import get_et
from iuvtools.geometry import get_sc_sza
from iuvtools.info import get_solar_lon

class PixelGlobeAll:
    def __init__(self, orbit_number, xlength=8000, ylength=8000, pixres=100):
        self.orbit_number = orbit_number
        # Bin setting
        self.xlength = xlength
        self.ylength = ylength
        self.pixres = pixres #20 #[km/pixel]
        self.xsize = int(self.xlength/self.pixres)
        self.ysize = int(self.ylength/self.pixres)
        # Empty bins to be filled with data
        self.databin = np.zeros((self.ysize, self.xsize))
        self.szabin = np.zeros((self.ysize, self.xsize))
        self.ltbin = np.zeros((self.ysize, self.xsize))
        self.latbin = np.zeros((self.ysize, self.xsize))
        self.lonbin = np.zeros((self.ysize, self.xsize))
        self.altbin = np.zeros((self.ysize, self.xsize))
        self.ndat = np.zeros((self.ysize, self.xsize))
        self.sza_sc = None
        self.alt_sc = None
        # Flip info
        self.flip = None

    def get_apoinfo(self):
        apoinfo = ApoapseInfo(self.orbit_number)
        return apoinfo

    def get_primary_dims(self, hdul):
        # determine dimensions, and if it's a single integration, skip it
        dims = hdul['primary'].shape
        if len(dims) == 3:
            n_int = dims[0]
            n_spa = dims[1]
            return n_int, n_spa
        else:
            print('No data or single integration, load skipped')
            return None, None

    def mesh_data(self, hdul):

        if self.flip is None:
            self.flip = beta_flip(hdul)

        n_int, n_spa = self.get_primary_dims(hdul)

        if n_int is not None:

            #if self.dayside(hdul):

            #self.flip = beta_flip(hdul)
            aposwath = ApoapseSwath(hdul)
            primary_arr = aposwath.get_img()
            alt_arr = hdul['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT']
            sza_arr = hdul['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE']
            lt_arr = hdul['PixelGeometry'].data['PIXEL_LOCAL_TIME']
            lat_arr = hdul['PixelGeometry'].data['PIXEL_CORNER_LAT']
            lon_arr = hdul['PixelGeometry'].data['PIXEL_CORNER_LON']

            # this is copied directly from Sonal; someday I'll figure it out and comment...
            # essentially it finds the place where the pixel position vector intersects the 400x400 grid
            # and places the pixel value in that location
            for i in range(n_int):
                vspc = hdul['spacecraftgeometry'].data[i]['v_spacecraft']
                vspcnorm = vspc/np.linalg.norm(vspc)
                vy = hdul['spacecraftgeometry'].data[i]['vy_instrument']
                vx = np.cross(vy, vspcnorm)

                for j in range(n_spa):
                    primary = primary_arr[i,j]
                    alt = alt_arr[i,j,4]
                    sza = sza_arr[i,j]
                    lt = lt_arr[i,j]
                    lat = lat_arr[i,j,4]
                    lon = lon_arr[i,j,4]

                    #sza = np.where(alt_arr[i,j, 4] == 0, sza, np.nan)
                    #lt = np.where(alt_arr[i,j,4] == 0, lt, np.nan)
                    #lat = np.where(alt_arr[i,j,4] == 0, lat, np.nan)
                    #lon = np.where(alt_arr[i,j,4] == 0, lon, np.nan)

                    for m in range(4):
                        try:
                            vpix = hdul['pixelgeometry'].data[i]['pixel_vec']
                            vpixcorner = (np.squeeze(vpix[:,j,m]) + np.squeeze(vpix[:,j,4]))/2
                            vdiff = vspc - (np.dot(vspc,vpixcorner)*vpixcorner)
                            x = int(np.dot(vdiff,vx)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /self.pixres+self.xsize/2)
                            y = int(np.dot(vdiff,vy)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /self.pixres+self.ysize/2)
                            if (x >= 0) & (y >= 0):
                                    self.databin[y,x] += primary
                                    self.altbin[y,x] += alt
                                    self.szabin[y,x] += sza
                                    self.ltbin[y,x] = lt # cannot average lt because average does not give a correct value i.e., (0+24)/2 = 12
                                    self.latbin[y,x] += lat
                                    self.lonbin[y,x] = lon # cannot average lon because average does not give a correct value i.e., (0+360)/2 = 180
                                    self.ndat[y,x] += 1
                        except:
                            continue

    def mesh_mean(self):
        self.databin = self.databin/self.ndat
        self.altbin = self.altbin/self.ndat
        self.szabin = self.szabin/self.ndat
        #self.ltbin = self.ltbin/self.ndat
        self.latbin = self.latbin/self.ndat
        #self.lonbin = self.lonbin/self.ndat

    def get_xygrids(self):
        #make coordinate grids for each pixel in kilometers
        x, y = np.meshgrid(np.linspace(-self.xsize/2*self.pixres, self.xsize/2*self.pixres, self.xsize), np.linspace(-self.ysize/2*self.pixres, self.ysize/2*self.pixres, self.ysize))
        return x, y

    def get_data(self):
        #calculate the average
        z = self.databin
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def plot(self, ax=None, nanalt=None, nansza=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_data()
        if nanalt is not None and nansza is not None:
            z = np.where((self.altbin<=nanalt) & (self.szabin<=nansza), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, **kwargs)
        return mesh

    def set_other_orbit(self, other_orbit_number):
        self.other_orbit_number = other_orbit_number
        self.other_datapath = saveloc  + 'quicklook/apoapse_l2b/Lyman-alpha/globe/orbit_' + '{:05d}'.format(other_orbit_number//100 * 100) + '/npy/orbit_' + '{:05d}'.format(self.other_orbit_number) + '.npy'

    def get_other_xygrids(self):
        dic = np.load(self.other_datapath, allow_pickle=True).item()
        x, y = dic['x'], dic['y']
        return x, y

    def get_other_data(self):
        dic = np.load(self.other_datapath, allow_pickle=True).item()
        z = dic['z']
        return z

    def plot_other(self, ax=None, nanalt=None, nansza=None, **kwargs):
        x, y = self.get_other_xygrids()
        z = self.get_other_data()
        if nanalt is not None and nansza is not None:
            z = np.where((self.altbin<=nanalt) & (self.szabin<=nansza), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, **kwargs)
        return mesh

    def get_diff_data(self):
        z = self.get_data()
        dic = np.load(self.other_datapath, allow_pickle=True).item()
        z_other = dic['z']
        z_diff = z - z_other
        return z_diff

    def plot_diff(self, ax=None, nanalt=None, nansza=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_diff_data()
        if nanalt is not None and nansza is not None:
            z = np.where((self.altbin<=nanalt) & (self.szabin<=nansza), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, **kwargs)
        return mesh


    def get_alt(self):
        #calculate the average
        z = self.altbin
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def get_sza(self):
        #calculate the average
        z = self.szabin
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def get_lt(self):
        #calculate the average
        z = self.ltbin
        z = np.where(self.ndat == 0, np.nan, z)
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def get_lat(self):
        #calculate the average
        z = self.latbin
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def get_lon(self):
        #calculate the average
        z = self.lonbin
        z = np.where(self.ndat == 0, np.nan, z)
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def plot_sza(self, ax=None, nanalt=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_sza()
        if nanalt is not None:
            z = np.where((self.altbin<=nanalt), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=0, vmax=180, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=0, vmax=180, **kwargs)
        return mesh

    def plot_lt(self, ax=None, nanalt=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_lt()
        if nanalt is not None:
            z = np.where((self.altbin<=nanalt), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=0, vmax=24, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=0, vmax=24, **kwargs)
        return mesh

    def plot_lat(self, ax=None, nanalt=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_lat()
        if nanalt is not None:
            z = np.where((self.altbin<=nanalt), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=-90, vmax=90, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=-90, vmax=90, **kwargs)
        return mesh

    def plot_lon(self, ax=None, nanalt=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_lon()
        if nanalt is not None:
            z = np.where((self.altbin<=nanalt), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=0, vmax=360, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=0, vmax=360, **kwargs)
        return mesh

def quicklook_apoapse_globe_diff(orbit_number):
    glb = PixelGlobeAll(orbit_number)
    glb.set_other_orbit(orbit_number - 1)
    apoinfo = glb.get_apoinfo()
    other_data_ok = os.path.isfile(glb.other_datapath)
    if apoinfo.n_files > 0 and other_data_ok:
        print('----- Processing orbit #', orbit_number, ' -----')
        et_apo = find_segment_et(orbit_number)
        et = []
        sc_sza = []
        Ls = []
        for ith_file, iswath_number in enumerate(apoinfo.swath_number):
            hdul = apoinfo.get_hdul(ith_file)
            glb.mesh_data(hdul)
            et.append(get_et(hdul))
            sc_sza.append(get_sc_sza(hdul))
            Ls.append(get_solar_lon(hdul))

        glb.mesh_mean()

        ## save obs info
        et = np.concatenate(et).ravel()
        et_lim = [et[0], et[-1]]
        sc_sza = np.concatenate(sc_sza).ravel()
        sc_sza_lim = [sc_sza[0], sc_sza[-1]]
        sc_sza_apo = np.interp(et_apo, et, sc_sza)
        Ls_lim = [Ls[0], Ls[1]]
        Ls_mean = np.nanmean(Ls)

        ## calc mean and median brightness
        nanalt = 500
        nansza = 90
        diff = glb.get_diff_data()
        x, y = glb.get_xygrids()
        alt = glb.get_alt()
        sza = glb.get_sza()
        lt = glb.get_lt()
        lat = glb.get_lat()
        lon = glb.get_lon()

        diff2 = np.where((glb.altbin<=nanalt) & (glb.szabin<=nansza), diff, np.nan)
        diff_mean = np.nanmean(diff2)
        diff_med = np.nanmedian(diff2)

        data_median = 2*np.nanmedian(glb.databin)
        data_mean = np.nanmean(glb.databin)
        data_max = np.nanmax(glb.databin)
        print(data_median, data_mean, data_max)

        plt.close()
        fig, ax = plt.subplots(4,3, figsize=(18, 18))
        mesh00 = glb.plot(ax=ax[0,0], cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=data_median))
        mesh10 = glb.plot_other(ax=ax[1,0], cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=data_median))
        mesh20 = glb.plot_diff(ax=ax[2,0], cmap='coolwarm', vmin=-2, vmax=2)
        mesh30 = glb.plot_diff(ax=ax[3,0], cmap=H_colormap(), vmin=0, vmax=2)

        mesh01 = glb.plot(ax=ax[0,1], nanalt=400, nansza=110, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=data_median))
        mesh11 = glb.plot_other(ax=ax[1,1], nanalt=400, nansza=110, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=data_median))
        mesh21 = glb.plot_diff(ax=ax[2,1], nanalt=400, nansza=110, cmap='coolwarm', vmin=-2, vmax=2)
        mesh31 = glb.plot_diff(ax=ax[3,1], nanalt=400, nansza=110, cmap=H_colormap(), vmin=0, vmax=2)
        mesh02 = glb.plot_sza(ax=ax[0,2], nanalt=0, cmap=plt.get_cmap('magma_r', 18))
        mesh12 = glb.plot_lt(ax=ax[1,2], nanalt=0, cmap=plt.get_cmap('twilight_shifted', 24))
        mesh22 = glb.plot_lat(ax=ax[2,2], nanalt=0, cmap=plt.get_cmap('coolwarm', 18))
        mesh32 = glb.plot_lon(ax=ax[3,2], nanalt=0, cmap=plt.get_cmap('twilight', 36))
        #mesh = np.array([[mesh00, mesh10], [mesh20, mesh30], [mesh01, mesh11], [mesh21, mesh31]])
        mesh = np.array([[mesh00, mesh01, mesh02], [mesh10, mesh11, mesh12], [mesh20, mesh21, mesh22], [mesh30, mesh31, mesh32]])
        ax[0,0].set_title('Orbit ' + str(orbit_number))
        ax[1,0].set_title('Previous orbit')
        ax[2,0].set_title('Diff')
        ax[3,0].set_title('Diff')
        ax[0,1].set_title('Orbit ' + str(orbit_number))
        ax[1,1].set_title('Previous orbit')
        ax[2,1].set_title('Diff')
        ax[3,1].set_title('Diff')
        ax[0,2].set_title('SZA [deg]')
        ax[1,2].set_title('Local Time [h]')
        ax[2,2].set_title('Latitude [deg]')
        ax[3,2].set_title('Longitude [deg]')

        # set xy labels and aspect ratio
        [[jax.set_xlabel('[km]') for jax in iax] for iax in ax]
        [[jax.set_ylabel('[km]') for jax in iax] for iax in ax]
        [[jax.set_aspect(1) for jax in iax] for iax in ax]
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = np.array([[make_axes_locatable(jax) for jax in iax] for iax in ax])
        cax = np.array([[jdivider.append_axes("right", size="5%", pad=0.05) for jdivider in idivider] for idivider in divider])

        #plt.colorbar(im, cax=cax)
        cb = np.array([[plt.colorbar(jmesh, cax=jcax) for jmesh, jcax in zip(imesh, icax)] for imesh, icax in zip(mesh, cax)])

        cb[0,0].set_label('Brightness [kR]')
        cb[1,0].set_label('Brightness [kR]')
        cb[2,0].set_label('Brightness [kR]')
        cb[3,0].set_label('Brightness [kR]')
        cb[0,1].set_label('Brightness [kR]')
        cb[1,1].set_label('Brightness [kR]')
        cb[2,1].set_label('Brightness [kR]')
        cb[3,1].set_label('Brightness [kR]')
        cb[0,2].set_label('SZA [deg]')
        cb[1,2].set_label('Local Time [h]')
        cb[2,2].set_label('Latitude [deg]')
        cb[3,2].set_label('Longitude [deg]')

        # save figure
        pngpath = saveloc + 'quicklook/apoapse_l2b/Lyman-alpha/globe_diff/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/'
        fname_save = 'orbit_' + '{:05d}'.format(orbit_number)
        if not os.path.exists(pngpath):
            os.makedirs(pngpath)
        plt.savefig(pngpath + fname_save)
        plt.tight_layout()
        # save data
        dic = {'et_apo':et_apo, 'et_lim':et_lim, 'sc_sza_apo':sc_sza_apo, 'sc_sza_lim':sc_sza_lim, 'Ls_mean':Ls_mean, 'Ls_lim':Ls_lim,
               'diff_mean':diff_mean, 'diff_med':diff_med}
        dicpath = saveloc + 'quicklook/apoapse_l2b/Lyman-alpha/globe_diff/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/npy/'
        dname_save = 'orbit_' + '{:05d}'.format(orbit_number)
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
        np.save(dicpath + dname_save, dic)

if __name__ == '__main__':
    load_iuvs_spice()
    start_orbit = int(sys.argv[1])
    n_orbit = int(sys.argv[2])
    orbit_arr = np.arange(n_orbit) + start_orbit
    for iorbit_number in np.arange(n_orbit) + start_orbit:
        quicklook_apoapse_globe_diff(iorbit_number)
        plt.close()
