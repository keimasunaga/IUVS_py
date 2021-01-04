import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
from scipy.linalg import hadamard, subspace_angles
from scipy.stats import circmean, circstd


from variables import saveloc
from PyUVS.geometry import beta_flip
from PyUVS.graphics import H_colormap
from iuvdata_l1b import ApoapseInfo, ApoapseSwath, FieldAngleGeo
from common.tools import RunTime
from common import circular
from PyUVS.time import find_segment_et, et2datetime
from PyUVS.spice import load_iuvs_spice
from iuvtools.time import get_et
from iuvtools.geometry import get_sc_sza, PixelTransCoord
from iuvtools.info import get_solar_lon
from pfptools.sw_drivers_jh import get_sw_driver_apo
from pfptools.euv_drivers_yd import get_euv_driver_apo
from iuvtools.data import primary_is_nan, echelle_place_ok


def get_angle_vectors_2d(v1, v2, degree=True):
    x1, y1 = v1[0], v1[1]
    x2, y2 = v2[0], v2[1]
    dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2      # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    if degree:
        return np.rad2deg(angle)
    else:
        return angle

class PixelGlobeAll:
    def __init__(self, orbit_number, xlength=8000, ylength=8000, pixres=100):
        self.orbit_number = orbit_number
        # Bin setting
        self.xlength = xlength
        self.ylength = ylength
        self.pixres = pixres #20 #[km/pixel]
        self.xsize = int(self.xlength/self.pixres)
        self.ysize = int(self.ylength/self.pixres)
        self.xdist = np.linspace(-self.xsize/2*self.pixres, self.xsize/2*self.pixres, self.xsize)
        self.ydist = np.linspace(-self.ysize/2*self.pixres, self.ysize/2*self.pixres, self.ysize)
        self.xmesh, self.ymesh = np.meshgrid(self.xdist, self.ydist)

        # Empty bins to be filled with data
        self.databin = np.zeros((self.ysize, self.xsize))
        self.szabin = np.zeros((self.ysize, self.xsize))
        self.ltbin = np.zeros((self.ysize, self.xsize))
        self.latbin = np.zeros((self.ysize, self.xsize))
        self.lonbin = np.zeros((self.ysize, self.xsize))
        self.altbin = np.zeros((self.ysize, self.xsize))
        self.szaxybin = np.zeros((self.ysize, self.xsize)) # angle from vec_sun in instrment xy plane (-180, 180)
        self.fieldanglebin = np.zeros((self.ysize, self.xsize))

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

    def mesh_data(self, hdul, fieldangle_obj=None):

        if self.flip is None:
            self.flip = beta_flip(hdul)

        n_int, n_spa = self.get_primary_dims(hdul)

        if n_int is not None:

            #if self.dayside(hdul):

            #self.flip = beta_flip(hdul)
            aposwath = ApoapseSwath(hdul)
            primary_arr = aposwath.fit_line()
            alt_arr = hdul['PixelGeometry'].data['PIXEL_CORNER_MRH_ALT']
            sza_arr = hdul['PixelGeometry'].data['PIXEL_SOLAR_ZENITH_ANGLE']
            lt_arr = hdul['PixelGeometry'].data['PIXEL_LOCAL_TIME']
            lat_arr = hdul['PixelGeometry'].data['PIXEL_CORNER_LAT']
            lon_arr = hdul['PixelGeometry'].data['PIXEL_CORNER_LON']
            pixtrans = PixelTransCoord(hdul)#['pixelgeometry'].data[i]['pixel_vec']
            vpix_from_pla_all = pixtrans.pixel_vec_from_pla_iau

            # this is copied directly from Sonal; someday I'll figure it out and comment...
            # essentially it finds the place where the pixel position vector intersects the 400x400 grid
            # and places the pixel value in that location
            for i in range(n_int):
                vpix = hdul['pixelgeometry'].data[i]['pixel_vec']
                vsun = hdul['spacecraftgeometry'].data[i]['v_sun']
                vspc = hdul['spacecraftgeometry'].data[i]['v_spacecraft']
                vspcnorm = vspc/np.linalg.norm(vspc)
                vy = hdul['spacecraftgeometry'].data[i]['vy_instrument']
                vx = np.cross(vy, vspcnorm)
                mat_to_inst  = np.array([vx, vy, vspcnorm])
                vsun_norm = vsun/np.linalg.norm(vsun)
                vsun_inst = np.matmul(mat_to_inst, vsun_norm)
                vsun_inst_2d = np.array([vsun_inst[0], vsun_inst[1]])/np.sqrt(vsun_inst[0]**2 + vsun_inst[1]**2)
                vpix_from_pla = vpix_from_pla_all[i]

                for j in range(n_spa):
                    primary = primary_arr[i,j]
                    alt = alt_arr[i,j,4]
                    sza = sza_arr[i,j]
                    lt = lt_arr[i,j]
                    lat = lat_arr[i,j,4]
                    lon = lon_arr[i,j,4]
                    vpix_from_pla_inst = np.matmul(mat_to_inst, vpix_from_pla[:,j,4])
                    vpix_from_pla_inst_2d = np.array([vpix_from_pla_inst[0], vpix_from_pla_inst[1]])/np.sqrt(vpix_from_pla_inst[0]**2 + vpix_from_pla_inst[1]**2)
                    r_xy = np.sqrt(vpix_from_pla_inst[0]**2 + vpix_from_pla_inst[1]**2)
                    vpix_from_pla_inst_norm = vpix_from_pla_inst/r_xy
                    sza_xy = get_angle_vectors_2d(vsun_inst, vpix_from_pla_inst_norm)#np.rad2deg(np.arctan2(np.dot(vsun_inst, np.cross(vsun_inst, vpix_from_pla_inst_norm)), np.dot(vsun_inst, vpix_from_pla_inst_norm)))

                    if fieldangle_obj.field_mso is not None:
                        field_angle = fieldangle_obj.data[i,j]
                    else:
                        field_angle = None

                    for m in range(4):
                        try:
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
                                    self.szaxybin[y,x] = sza_xy # cannot average theta because average does not give a correct value i.e., (-180+180)/2 = 0
                                    if field_angle is not None:
                                        self.fieldanglebin[y,x] += field_angle
                        except:
                            continue

    def mesh_mean(self):
        self.databin = self.databin/self.ndat
        self.altbin = self.altbin/self.ndat
        self.szabin = self.szabin/self.ndat
        #self.ltbin = self.ltbin/self.ndat
        self.latbin = self.latbin/self.ndat
        #self.lonbin = self.lonbin/self.ndat
        #self.thetabin = self.thetabin/self.ndat
        self.fieldanglebin = self.fieldanglebin/self.ndat

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

    def plot(self, ax=None, alt_lim=None, nansza=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_data()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, **kwargs)
        return mesh

    def set_other_orbit(self, other_orbit_number):
        self.other_orbit_number = other_orbit_number
        self.other_datapath = saveloc  + 'quicklook/apoapse_l1b/Lyman-alpha/globe/orbit_' + '{:05d}'.format(other_orbit_number//100 * 100) + '/npy/orbit_' + '{:05d}'.format(self.other_orbit_number) + '.npy'

    def get_other_xygrids(self):
        dic = np.load(self.other_datapath, allow_pickle=True).item()
        x, y = dic['x'], dic['y']
        return x, y

    def get_other_data(self):
        dic = np.load(self.other_datapath, allow_pickle=True).item()
        z = dic['z']
        return z

    def plot_other(self, ax=None, alt_lim=None, nansza=None, **kwargs):
        x, y = self.get_other_xygrids()
        z = self.get_other_data()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
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

    def plot_diff(self, ax=None, alt_lim=None, nansza=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_diff_data()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
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

    def plot_sza(self, ax=None, alt_lim=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_sza()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=0, vmax=180, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=0, vmax=180, **kwargs)
        return mesh

    def plot_lt(self, ax=None, alt_lim=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_lt()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=0, vmax=24, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=0, vmax=24, **kwargs)
        return mesh

    def plot_lat(self, ax=None, alt_lim=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_lat()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=-90, vmax=90, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=-90, vmax=90, **kwargs)
        return mesh

    def plot_lon(self, ax=None, alt_lim=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_lon()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=0, vmax=360, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=0, vmax=360, **kwargs)
        return mesh

    def get_sza_xy(self):
        #calculate the average
        z = self.szaxybin
        z = np.where(self.ndat == 0, np.nan, z)
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def plot_sza_xy(self, ax=None, alt_lim=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_sza_xy()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=-180, vmax=180, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=-180, vmax=180, **kwargs)
        return mesh


    def get_fieldangle(self):
        #calculate the average
        z = self.fieldanglebin
        z = np.where(self.ndat == 0, np.nan, z)
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def plot_fieldangle(self, ax=None, alt_lim=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_fieldangle()
        if alt_lim is not None:
            z = np.where((self.altbin>=alt_lim[0])&(self.altbin<=alt_lim[1]), z, np.nan)
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, vmin=0, vmax=180, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, vmin=0, vmax=180, **kwargs)
        return mesh

def save_globe_data(orbit_number):
    glb = PixelGlobeAll(orbit_number)
    glb.set_other_orbit(orbit_number - 1)
    apoinfo = glb.get_apoinfo()

    if apoinfo.n_files > 0:
        et_apo = find_segment_et(orbit_number)
        Dt_apo = et2datetime(et_apo)
        print('DTAPO',Dt_apo)

        et = []
        sc_sza = []
        Ls = []
        dic_euv = get_euv_driver_apo(orbit_number)
        dic_sw = get_sw_driver_apo(orbit_number)
        if dic_sw is not None:
            nsw = dic_sw['npsw']
            vsw = dic_sw['vvec']
            bsw = dic_sw['bsw'][0:-1]
            fsw = nsw * (vsw * 1e5)
            esw = -np.cross(vsw, bsw)
        else:
            esw = None

        nan_ok = True
        echelle_ok = True
        for ith_file, iswath_number in enumerate(apoinfo.swath_number):
            hdul = apoinfo.get_hdul(ith_file)

            if primary_is_nan(hdul):
                nan_ok = False
                continue
            #if echelle_place_ok(hdul) is False:
            #    echelle_ok = False
            #    continue

            fieldangle = FieldAngleGeo(hdul, iswath_number)
            fieldangle.calc_cone_angle(esw)
            glb.mesh_data(hdul, fieldangle)
            et.append(get_et(hdul))
            sc_sza.append(get_sc_sza(hdul))
            Ls.append(get_solar_lon(hdul))

        if nan_ok and echelle_ok:

            glb.mesh_mean()

            ## save obs info
            et = np.concatenate(et).ravel()
            Dt_lim = [et2datetime(et[0]), et2datetime(et[-1])]
            sc_sza = np.concatenate(sc_sza).ravel()
            sc_sza_lim = [sc_sza[0], sc_sza[-1]]
            sc_sza_apo = np.interp(et_apo, et, sc_sza)
            Ls_lim = [Ls[0], Ls[1]]
            Ls_mean = circmean(Ls, high=360, low=0)

            dic_iuvs = {'orbit_number':orbit_number, 'file_version':apoinfo.file_version,
                        'length':glb.xlength, 'pixres':glb.pixres, 'npixel':glb.xsize,
                        'x':glb.xdist, 'y':glb.ydist,
                        'data':glb.databin, 'sza':glb.szabin, 'lat':glb.latbin, 'lon':glb.lonbin, 'lt':glb.ltbin,
                        'sza_xy':glb.szaxybin, 'efield_angle':glb.fieldanglebin, 'alt':glb.altbin,
                        'Dt_apo':Dt_apo, 'Dt_lim':Dt_lim,
                        'sc_sza_apo':sc_sza_apo, 'sc_sza_lim':sc_sza_lim, 'beta_flip':glb.flip,
                        'Ls_mean':Ls_mean, 'Ls_lim':Ls_lim}

            dic_save = {'dic_iuvs':dic_iuvs, 'dic_sw':dic_sw, 'dic_euv':dic_euv}
            dicpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza_all/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/npy/'
            dname_save = 'orbit_' + '{:05d}'.format(orbit_number)
            if not os.path.exists(dicpath):
                os.makedirs(dicpath)
            np.save(dicpath + dname_save, dic_save)
        hdul.close()



from PyUVS.graphics import H_colormap

from orbit_info import get_Dt_apo
from iuvtools.info import get_sza_apo
from common import circular
from scipy.stats import circmean
import pandas as pd

def get_dic_df(dic_iuvs, bool_alt, bool_szaxy):

    data = dic_iuvs['data']
    sza = dic_iuvs['sza']
    lat = dic_iuvs['lat']
    lon = dic_iuvs['lon']
    lt = dic_iuvs['lt']
    sza_xy = dic_iuvs['sza_xy']
    efield_angle = dic_iuvs['efield_angle']
    alt = dic_iuvs['alt']

    # save data
    data_mean = []
    sza_mean = []
    lat_mean = []
    lon_mean = []
    #szaxy_mean = []
    lt_mean = []
    efield_angle_mean = []

    data_std = []
    sza_std = []
    lat_std = []
    lon_std = []
    #szaxy_std = []
    lt_std = []
    efield_angle_std = []

    data_med = []
    sza_med = []
    lat_med = []
    lon_med = None
    #szaxy_med = []
    lt_med = None
    efield_angle_med = []

    nbin = []
    #import pdb; pdb.set_trace()
    for ith,ibool_szaxy in enumerate(bool_szaxy):
        # calc data for bool_alt
        data_temp = np.where(bool_alt & ibool_szaxy, data, np.nan)
        nbin.append(np.size(data_temp[~np.isnan(data_temp)]))
        data_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, data, np.nan)))
        sza_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, sza, np.nan)))
        lat_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, lat, np.nan)))
        lon_temp = np.where(bool_alt & ibool_szaxy, lon, np.nan)
        lon_mean.append(circmean(lon_temp[~np.isnan(lon_temp)], high=360, low=0))
        lt_temp = np.where(bool_alt & ibool_szaxy, lt, np.nan)
        lt_mean.append(circmean(lt_temp[~np.isnan(lt_temp)], high=24, low=0))
        efield_angle_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, efield_angle, np.nan)))

        data_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, data, np.nan)))
        sza_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, sza, np.nan)))
        lat_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, lat, np.nan)))
        lon_std.append(circstd(lon_temp[~np.isnan(lon_temp)], high=360, low=0))
        lt_std.append(circstd(lt_temp[~np.isnan(lt_temp)], high=24, low=0))
        efield_angle_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, efield_angle, np.nan)))

        data_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, data, np.nan)))
        sza_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, sza, np.nan)))
        lat_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, lat, np.nan)))
        efield_angle_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, efield_angle, np.nan)))


    data_df = pd.DataFrame({'mean':data_mean, 'std':data_std, 'med':data_med, 'nbin':nbin})
    sza_df = pd.DataFrame({'mean':sza_mean, 'std':sza_std, 'med':sza_med, 'nbin':nbin})
    lat_df = pd.DataFrame({'mean':lat_mean, 'std':lat_std, 'med':lat_med, 'nbin':nbin})
    lon_df = pd.DataFrame({'mean':lon_mean, 'std':lon_std, 'med':lon_med, 'nbin':nbin})
    lt_df = pd.DataFrame({'mean':lt_mean, 'std':lt_std, 'med':lt_med, 'nbin':nbin})
    efield_angle_df = pd.DataFrame({'mean':efield_angle_mean, 'std':efield_angle_std, 'med':efield_angle_med, 'nbin':nbin})
    dic_df = {'data_df':data_df, 'sza_df':sza_df, 'lat_df':lat_df, 'lon_df':lon_df, 'lt_df':lt_df, 'efield_angle_df':efield_angle_df}
    return dic_df

def get_dic_df_disk(dic_iuvs, bool_alt, bool_sza):

    data = dic_iuvs['data']
    sza = dic_iuvs['sza']
    lat = dic_iuvs['lat']
    lon = dic_iuvs['lon']
    lt = dic_iuvs['lt']
    sza_xy = dic_iuvs['sza_xy']
    efield_angle = dic_iuvs['efield_angle']
    alt = dic_iuvs['alt']

    # save data
    data_mean = []
    sza_mean = []
    lat_mean = []
    lon_mean = []
    #szaxy_mean = []
    lt_mean = []
    efield_angle_mean = []

    data_std = []
    sza_std = []
    lat_std = []
    lon_std = []
    #szaxy_std = []
    lt_std = []
    efield_angle_std = []

    data_med = []
    sza_med = []
    lat_med = []
    lon_med = None
    #szaxy_med = []
    lt_med = None
    efield_angle_med = []

    nbin = []
    #import pdb; pdb.set_trace()
    for ith,ibool_szaxy in enumerate(bool_sza):
        # calc data for bool_alt
        data_temp = np.where(bool_alt & ibool_szaxy, data, np.nan)
        nbin.append(np.size(data_temp[~np.isnan(data_temp)]))
        data_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, data, np.nan)))
        sza_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, sza, np.nan)))
        lat_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, lat, np.nan)))
        lon_temp = np.where(bool_alt & ibool_szaxy, lon, np.nan)
        lon_mean.append(circmean(lon_temp[~np.isnan(lon_temp)], high=360, low=0))
        lt_temp = np.where(bool_alt & ibool_szaxy, lt, np.nan)
        lt_mean.append(circmean(lt_temp[~np.isnan(lt_temp)], high=24, low=0))
        efield_angle_mean.append(np.nanmean(np.where(bool_alt & ibool_szaxy, efield_angle, np.nan)))

        data_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, data, np.nan)))
        sza_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, sza, np.nan)))
        lat_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, lat, np.nan)))
        lon_std.append(circstd(lon_temp[~np.isnan(lon_temp)], high=360, low=0))
        lt_std.append(circstd(lt_temp[~np.isnan(lt_temp)], high=24, low=0))
        efield_angle_std.append(np.nanstd(np.where(bool_alt & ibool_szaxy, efield_angle, np.nan)))

        data_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, data, np.nan)))
        sza_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, sza, np.nan)))
        lat_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, lat, np.nan)))
        efield_angle_med.append(np.nanmedian(np.where(bool_alt & ibool_szaxy, efield_angle, np.nan)))


    data_df = pd.DataFrame({'mean':data_mean, 'std':data_std, 'med':data_med, 'nbin':nbin})
    sza_df = pd.DataFrame({'mean':sza_mean, 'std':sza_std, 'med':sza_med, 'nbin':nbin})
    lat_df = pd.DataFrame({'mean':lat_mean, 'std':lat_std, 'med':lat_med, 'nbin':nbin})
    lon_df = pd.DataFrame({'mean':lon_mean, 'std':lon_std, 'med':lon_med, 'nbin':nbin})
    lt_df = pd.DataFrame({'mean':lt_mean, 'std':lt_std, 'med':lt_med, 'nbin':nbin})
    efield_angle_df = pd.DataFrame({'mean':efield_angle_mean, 'std':efield_angle_std, 'med':efield_angle_med, 'nbin':nbin})
    dic_df = {'data_df':data_df, 'sza_df':sza_df, 'lat_df':lat_df, 'lon_df':lon_df, 'lt_df':lt_df, 'efield_angle_df':efield_angle_df}
    return dic_df




def save_globe_data_region(orbit_number, savefig=True):
    # Load saved data
    dicpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza_all/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/npy/'
    dname_save = 'orbit_' + '{:05d}'.format(orbit_number) + '.npy'
    if os.path.isfile(dicpath+dname_save):
        dic = np.load(dicpath + dname_save, allow_pickle=True).item()
        dic_sw = dic['dic_sw']
        dic_iuvs = dic['dic_iuvs']

        # get iuvs data
        data = dic_iuvs['data']
        sza = dic_iuvs['sza']
        lat = dic_iuvs['lat']
        lon = dic_iuvs['lon']
        lt = dic_iuvs['lt']
        sza_xy = dic_iuvs['sza_xy']
        efield_angle = dic_iuvs['efield_angle']
        alt = dic_iuvs['alt']
        Dt_apo = dic_iuvs['Dt_apo']
        timestring_apo = Dt_apo.strftime("%Y-%m-%d %H:%M:%S")
        sc_sza_apo = dic_iuvs['sc_sza_apo']
        Ls_mean = dic_iuvs['Ls_mean']
        x = dic_iuvs['x']
        y = dic_iuvs['y']
        file_version = dic_iuvs['file_version']

        # set bool for limb regions
        bool_alt_limb = (alt>100)&(alt<=200)  ## boolen to select limb region (100-200km)
        bool_alt_limb2 = (alt>200)&(alt<=300) ## boolen to select limb region (200-300km)
        bool_szaxy = [np.abs(sza_xy)>150, (sza_xy>=-150)&(sza_xy<-90), (sza_xy>=-90)&(sza_xy<-30),
                     (sza_xy>=-30)&(sza_xy<30), (sza_xy>=30)&(sza_xy<90), (sza_xy>=90)&(sza_xy<150)]
        dic_df_limb = get_dic_df(dic_iuvs, bool_alt_limb, bool_szaxy)
        dic_df_limb2 = get_dic_df(dic_iuvs, bool_alt_limb2, bool_szaxy)

        # set bool for disk regions
        bool_alt_disk = (alt==0) ## boolen to select disk region
        bool_sza = [sza<=30, (sza>30)&(sza<=60), (sza>60)&(sza<=90),
                   (sza>90)&(sza<=120), (sza>120)&(sza<=150), (sza>150)&(sza<=180)]
        dic_df_disk = get_dic_df_disk(dic_iuvs, bool_alt_disk, bool_sza)

        dic_save = {'dic_df_limb':dic_df_limb, 'dic_df_limb2':dic_df_limb2, 'dic_df_disk':dic_df_disk}

        savepath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza_all/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/regions/'
        savename = 'orbit_' + '{:05d}'.format(orbit_number)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        np.save(savepath+savename, dic_save)

        if savefig:
            # Input NaN into bins outside the limited altitude range for geometry plots
            alt_lim_geo = (alt>=0)&(alt<=200)
            sza = np.where(alt_lim_geo, sza, np.nan)
            lat = np.where(alt_lim_geo, lat, np.nan)
            lon = np.where(alt_lim_geo, lon, np.nan)
            sza_xy = np.where(alt_lim_geo, sza_xy, np.nan)
            lt = np.where(alt_lim_geo, lt, np.nan)
            efield_angle = np.where(alt_lim_geo, efield_angle, np.nan)
            # plot
            plt.close()
            fig, ax = plt.subplots(4,2,figsize=(10,15))
            ax[3,1].remove()
            fig.suptitle('Orbit ' + '{:05d}'.format(orbit_number)+ ' (' + file_version+')'+ '\n'+timestring_apo + '\n SZA_SC=' + '{:.1f}'.format(sc_sza_apo) + '\n Ls=' + '{:.1f}'.format(Ls_mean), y=0.95)
            #fig.suptitle('Orbit ' + '{:05d}'.format(orbit_number)+ '\n'+timestring_apo + '\n SZA_SC=' + '{:.1f}'.format(sc_sza_apo) + '\n Ls=' + '{:.1f}'.format(Ls_mean), y=0.95)
            ax[0,0].set_title('Brightness')
            mesh00 = ax[0,0].pcolormesh(x, y, data, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=10))
            divider00 = make_axes_locatable(ax[0,0])
            cax00 = divider00.append_axes("right", size="5%", pad=0.05)
            cb00 = plt.colorbar(mesh00, cax=cax00)

            ax[1,0].set_title('Solar Zenith Angle')
            mesh10 = ax[1,0].pcolormesh(x, y, sza, vmin=0, vmax=180, cmap=plt.get_cmap('magma_r', 18))
            divider10 = make_axes_locatable(ax[1,0])
            cax10 = divider10.append_axes("right", size="5%", pad=0.05)
            cb10 = plt.colorbar(mesh10, cax=cax10)

            ax[2,0].set_title('SZA_xy')
            mesh20 = ax[2,0].pcolormesh(x, y, sza_xy, vmin=-180, vmax=180, cmap=plt.get_cmap('coolwarm', 36))
            divider20 = make_axes_locatable(ax[2,0])
            cax20 = divider20.append_axes("right", size="5%", pad=0.05)
            cb20 = plt.colorbar(mesh20, cax=cax20)

            ax[3,0].set_title('Angle wrt E-field')
            mesh30 = ax[3,0].pcolormesh(x, y, efield_angle, vmin=0, vmax=180, cmap=plt.get_cmap('bwr_r', 18))
            divider30 = make_axes_locatable(ax[3,0])
            cax30 = divider30.append_axes("right", size="5%", pad=0.05)
            cb30 = plt.colorbar(mesh30, cax=cax30)

            ax[0,1].set_title('Local Time')
            mesh01 = ax[0,1].pcolormesh(x, y, lt, vmin=0, vmax=24, cmap=plt.get_cmap('twilight_shifted', 24))
            divider01 = make_axes_locatable(ax[0,1])
            cax01 = divider01.append_axes("right", size="5%", pad=0.05)
            cb01 = plt.colorbar(mesh01, cax=cax01)

            ax[1,1].set_title('Longitude')
            mesh11 = ax[1,1].pcolormesh(x, y, lon, vmin=0, vmax=360, cmap=plt.get_cmap('twilight', 36))
            divider11 = make_axes_locatable(ax[1,1])
            cax11 = divider11.append_axes("right", size="5%", pad=0.05)
            cb11 = plt.colorbar(mesh11, cax=cax11)

            ax[2,1].set_title('Latitude')
            mesh21 = ax[2,1].pcolormesh(x, y, lat, vmin=-90, vmax=90, cmap=plt.get_cmap('coolwarm', 18))
            #divider21 = make_axes_locatable(ax[2,1])
            #cax21 = divider21.append_axes("right", size="5%", pad=0.05)
            cb21 = plt.colorbar(mesh21, ax=ax[2,1], pad=0.05)#cax=cax21)

            [[jax.set_xlabel('[km]') for jax in iax] for iax in ax]
            [[jax.set_ylabel('[km]') for jax in iax] for iax in ax]
            [[jax.set_aspect(1) for jax in iax] for iax in ax]

            cb00.set_label('[kR]',rotation=270, labelpad=10)
            cb10.set_label('[degree]',rotation=270, labelpad=10)
            cb20.set_label('[degree]',rotation=270, labelpad=10)
            cb30.set_label('[degree]',rotation=270, labelpad=10)
            cb01.set_label('[hour]',rotation=270, labelpad=10)
            cb11.set_label('[degree]',rotation=270, labelpad=10)
            cb21.set_label('[degree]',rotation=270, labelpad=10)

            fig.subplots_adjust(hspace=0.3, wspace=0.5)

            figpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza_all/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/fig/'
            savename = 'orbit_' + '{:05d}'.format(orbit_number)
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            plt.savefig(figpath+savename)
    else:
        print('No file found at orbit #'+'{:05d}'.format(orbit_number)+ ', skipping')


if __name__ == '__main__':
    load_iuvs_spice(True)
    sorbit = int(sys.argv[1])
    norbit = int(sys.argv[2])
    eorbit = sorbit + norbit
    orbit_arr = range(sorbit, eorbit)#[849]
    error_orbit = []
    for iorbit_number in orbit_arr:
        #try:
        print('{:05d}'.format(iorbit_number))
        save_globe_data(iorbit_number)
        save_globe_data_region(iorbit_number)
        #except:
        #print('Error caused, skipping orbit'+str(iorbit_number))
        #error_orbit.append(iorbit_number)
        plt.close()
