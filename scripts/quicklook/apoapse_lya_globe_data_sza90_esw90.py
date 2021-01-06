import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
from scipy.linalg import hadamard, subspace_angles
from scipy.stats import circmean, circstd
import pandas as pd

from maven_iuvs.geometry import beta_flip
from maven_iuvs.graphics import H_colormap
from maven_iuvs.time import find_segment_et, et2datetime
from maven_iuvs.spice import load_iuvs_spice

from variables import saveloc, spiceloc
from iuvdata_l1b import ApoapseInfo, ApoapseSwath, FieldAngleGeo
from common.tools import RunTime
from common import circular
from iuvtools.time import get_et
from iuvtools.geometry import get_sc_sza, PixelTransCoord
from iuvtools.info import get_solar_lon
from pfptools.sw_drivers_jh import get_sw_driver_apo
from pfptools.euv_drivers_yd import get_euv_driver_apo
from iuvtools.data import primary_is_nan, echelle_place_ok
from orbit_info import get_Dt_apo
from iuvtools.info import get_sza_apo
from common import circular


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

        # efield_angle at 0 (origin). This is used to judge if the efield lies in the instrument plane.
        efield_angle_origin = efield_angle[np.where(np.abs(y) == np.abs(y).min()), np.where(np.abs(x) == np.abs(x).min())]
        if (efield_angle_origin >= 70) & (efield_angle_origin <= 110): # 90+/-20deg

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

            dic_save = {'dic_df_limb':dic_df_limb, 'dic_df_limb2':dic_df_limb2, 'dic_df_disk':dic_df_disk, 'efield_angle_origin':efield_angle_origin}

            savepath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90_esw90/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/regions/'
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

                figpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90_esw90/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/fig/'
                savename = 'orbit_' + '{:05d}'.format(orbit_number)
                if not os.path.exists(figpath):
                    os.makedirs(figpath)
                plt.savefig(figpath+savename)
    else:
        print('No file found at orbit #'+'{:05d}'.format(orbit_number)+ ', skipping')


if __name__ == '__main__':
    load_iuvs_spice(spiceloc, load_long_allTrue)
    sorbit = int(sys.argv[1])
    norbit = int(sys.argv[2])
    eorbit = sorbit + norbit
    orbit_arr = range(sorbit, eorbit)#[849]
    error_orbit = []
    for iorbit_number in orbit_arr:
        print('{:05d}'.format(iorbit_number))
        save_globe_data_region(iorbit_number)
        plt.close()
