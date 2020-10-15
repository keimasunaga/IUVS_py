import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys

from PyUVS.graphics import H_colormap
from variables import saveloc


def get_globe_data_region(orbit_number, region=0, alt_default=True, median_br=True):

    savepath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/regions/'
    savename = 'orbit_' + '{:05d}'.format(orbit_number) + '.npy'
    if os.path.isfile(savepath+savename):
        dic_load = np.load(savepath+savename, allow_pickle=True).item()
        if alt_default:
            dic = dic_load['dic_df']
        else:
            dic = dic_load['dic_df2']

        data_df = dic['data_df']
        sza_df = dic['sza_df']
        efield_angle_df = dic['efield_angle_df']
        lat_df = dic['lat_df']
        lon_df = dic['lon_df']
        lt_df = dic['lt_df']

        data = data_df[data_df.index==region]
        data_out = data['med'].item() if median_br else data['mean'].item()
        sza = sza_df[sza_df.index==region]
        sza_out = sza['mean'].item()
        angle_efield = efield_angle_df[efield_angle_df.index==region]
        angle_efield_out = angle_efield['mean'].item()
        lat = lat_df[lat_df.index==region]
        lat_out = lat['mean'].item()
        lon = lon_df[lon_df.index==region]
        lon_out = lon['mean'].item()
        lt = lt_df[lt_df.index==region]
        lt_out = lt['mean'].item()

        dic_out = {'data':data_out, 'sza':sza_out, 'angle_efield':angle_efield_out,
                   'lat':lat_out, 'lon':lon_out, 'lt':lt_out}
        return dic_out

    else:
        return None


def get_iuv_sw_euv_data(orbit_number):
    dicpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/npy/'
    dname_save = 'orbit_' + '{:05d}'.format(orbit_number) + '.npy'
    dic = np.load(dicpath + dname_save, allow_pickle=True).item()
    dic_sw = dic['dic_sw']
    dic_iuvs = dic['dic_iuvs']
    dic_euv = dic['dic_euv']

    return dic_iuvs, dic_sw, dic_euv


def plot_overview(alt_default=True, selec_region=[0,1,2,3,4,5]):
    args = [i for i in sys.argv]
    if len(args)>1:
        alt_default = args[1]
        if args[1] == 'False':
            alt_default = False
        else:
            alt_default = True

    color = [[0.22719513, 0.24914381, 0.40675258, 1. ],
             [0.53262074, 0.35198344, 0.50904462, 1. ],
             [0.90686879, 0.62111989, 0.57090984, 1. ],
             [0.91383359, 0.82674759, 0.55570039, 1. ],
             [0.55800763, 0.76309657, 0.59473906, 1. ],
             [0.23115755, 0.46685701, 0.50050539, 1. ]]
    orbit_arr = np.arange(700, 5500)
    for ith, iregion in enumerate(selec_region):
        data = []
        sza = []
        angle = []
        lat = []
        lon = []
        lt = []
        fsw = []
        euv_lya = []
        Ls = []
        timeDt = []

        for iorbit in orbit_arr:
            dic = get_globe_data_region(iorbit, region=iregion, alt_default=alt_default, median_br=False)
            if dic is None:
                continue
            data.append(dic['data'])
            sza.append(dic['sza'])
            angle.append(dic['angle_efield'])
            lat.append(dic['lat'])
            lon.append(dic['lon'])
            lt.append(dic['lt'])

            dic_iuvs, dic_sw, dic_euv = get_iuv_sw_euv_data(iorbit)
            if dic_sw is not None:
                vsw = dic_sw['vpsw']
                nsw = dic_sw['npsw']
                fsw.append(nsw * vsw * 1e5)
            else:
                fsw.append(np.nan)

            if dic_euv is not None:
                euv_lya.append(dic_euv['euv'][2])
            else:
                euv_lya.append(np.nan)

            if dic_iuvs is not None:
                Ls.append(dic_iuvs['Ls_mean'])
                timeDt.append(dic_iuvs['Dt_apo'])
            else:
                Ls.append(np.nan)
                timeDt.append(np.nan)
        data = np.array(data)
        sza = np.array(sza)
        angle = np.array(angle)
        lat = np.array(lat)
        lon = np.array(lon)
        lt = np.array(lt)
        fsw = np.array(fsw)
        euv_lya = np.array(euv_lya)
        Ls = np.array(Ls)
        timeDt = np.array(timeDt)

        """idx = np.where(np.array(Ls)>180)[0]
        data = data[idx]
        angle = angle[idx]
        lat = lat[idx]
        lon = lon[idx]
        euv_lya = euv_lya[idx]
        fsw = fsw[idx]
        Ls = Ls[idx]"""

        if ith==0:
            fig = plt.figure(figsize=(20, 20))
            widths = [3, 1]
            gs = fig.add_gridspec(9,2, width_ratios=widths)#fig.add_gridspec(5, 2, height_ratios=heights)
            plt.subplots_adjust(hspace = 0.3)

        ax00 = fig.add_subplot(gs[0,0])
        ax00.plot(timeDt, data, 'o', markersize=1, label='region '+str(iregion), color=color[iregion])
        ax00.set_xlabel('time')
        ax00.set_ylabel('Brightness [kR]')
        ax00.set_ylim(0, 50)
        ax00.set_yticks(np.arange(0, 51, 10))

        ax10 = fig.add_subplot(gs[1,0])
        ax10.plot(timeDt, euv_lya, 'o', markersize=1, color=color[iregion])
        ax10.set_xlabel('time')
        ax10.set_ylabel('Ly-alpha irradiance [W/m2]')

        ax20 = fig.add_subplot(gs[2,0])
        ax20.plot(timeDt, fsw, 'o', markersize=1, color=color[iregion])
        ax20.set_xlabel('time')
        ax20.set_ylabel('Fsw [/cm2/s]')
        ax20.set_yscale('log')

        ax30 = fig.add_subplot(gs[3,0])
        ax30.plot(timeDt[angle>0], angle[angle>0], 'o', markersize=1, color=color[iregion])
        ax30.set_xlabel('time')
        ax30.set_ylabel('Angle wrt E-field [deg]')
        ax30.set_ylim(0, 180)
        ax30.set_yticks(np.arange(0, 181, 30))

        ax40 = fig.add_subplot(gs[4,0])
        ax40.plot(timeDt, Ls, 'o', markersize=1, color=color[iregion])
        ax40.set_xlabel('time')
        ax40.set_ylabel('Ls [deg]')
        ax40.set_ylim(-10, 370)
        ax40.set_yticks(np.arange(0, 361, 60))

        ax50 = fig.add_subplot(gs[5,0])
        ax50.plot(timeDt, sza, 'o', markersize=1, color=color[iregion])
        ax50.set_xlabel('time')
        ax50.set_ylabel('SZA [deg]')
        ax50.set_ylim(0, 180)
        ax50.set_yticks(np.arange(0, 181, 30))

        ax60 = fig.add_subplot(gs[6,0])
        ax60.plot(timeDt, lt, 'o', markersize=1, color=color[iregion])
        ax60.set_xlabel('time')
        ax60.set_ylabel('LT [deg]')
        ax60.set_ylim(0, 24)
        ax60.set_yticks(np.arange(0, 25, 4))

        ax70 = fig.add_subplot(gs[7,0])
        ax70.plot(timeDt, lon, 'o', markersize=1, color=color[iregion])
        ax70.set_xlabel('time')
        ax70.set_ylabel('Lon [deg]')
        ax70.set_ylim(-10, 370)
        ax70.set_yticks(np.arange(0, 361, 60))

        ax80 = fig.add_subplot(gs[8,0])
        ax80.plot(timeDt, lat, 'o', markersize=1, color=color[iregion])
        ax80.set_xlabel('time')
        ax80.set_ylabel('Lat [deg]')
        ax80.set_ylim(-90, 90)
        ax80.set_yticks(np.arange(-90, 91, 30))

        ax11 = fig.add_subplot(gs[1,1])
        ax11.plot(euv_lya, data, 'o', markersize=1, color=color[iregion])
        ax11.set_xlabel('Ly-alpha irradiance [W/m2]')
        ax11.set_ylabel('Brightness [kR]')
        ax11.set_ylim(0, 50)
        ax11.set_yticks(np.arange(0, 51, 10))

        ax21 = fig.add_subplot(gs[2,1])
        ax21.plot(fsw, data, 'o', markersize=1, color=color[iregion])
        ax21.set_xlabel('Fsw [/cm2/s]')
        ax21.set_ylabel('Brightness [kR]')
        ax21.set_xscale('log')
        ax21.set_ylim(0, 50)
        ax21.set_yticks(np.arange(0, 51, 10))

        ax31 = fig.add_subplot(gs[3,1])
        ax31.plot(angle[angle>0], data[angle>0], 'o', markersize=1, color=color[iregion])
        ax31.set_xlabel('Angle wrt E-field [deg]')
        ax31.set_ylabel('Brightness [kR]')
        ax31.set_xlim(0, 180)
        ax31.set_xticks(np.arange(0, 181, 30))
        ax31.set_ylim(0, 50)
        ax31.set_yticks(np.arange(0, 51, 10))

        ax41 = fig.add_subplot(gs[4,1])
        ax41.plot(Ls, data, 'o', markersize=1, color=color[iregion])
        ax41.set_xlabel('Ls [deg]')
        ax41.set_ylabel('Brightness [kR]')
        ax41.set_xlim(-10, 370)
        ax41.set_xticks(np.arange(0, 361, 60))
        ax41.set_ylim(0, 50)
        ax41.set_yticks(np.arange(0, 51, 10))

        ax51 = fig.add_subplot(gs[5,1])
        ax51.plot(sza, data, 'o', markersize=1, color=color[iregion])
        ax51.set_xlabel('SZA [deg]')
        ax51.set_ylabel('Brightness [kR]')
        ax51.set_xlim(0, 180)
        ax51.set_xticks(np.arange(0, 181, 30))
        ax51.set_ylim(0, 50)
        ax51.set_yticks(np.arange(0, 51, 10))

        ax61 = fig.add_subplot(gs[6,1])
        ax61.plot(lt, data, 'o', markersize=1, color=color[iregion])
        ax61.set_xlabel('LT [hour]')
        ax61.set_ylabel('Brightness [kR]')
        ax61.set_xlim(-0.5, 24.5)
        ax61.set_xticks(np.arange(0, 25, 4))
        ax61.set_ylim(0, 50)
        ax61.set_yticks(np.arange(0, 51, 10))

        ax71 = fig.add_subplot(gs[7,1])
        ax71.plot(lon, data, 'o', markersize=1, color=color[iregion])
        ax71.set_xlabel('Lon [deg]')
        ax71.set_ylabel('Brightness [kR]')
        ax71.set_xlim(-10, 370)
        ax71.set_xticks(np.arange(0, 361, 60))
        ax71.set_ylim(0, 50)
        ax71.set_yticks(np.arange(0, 51, 10))

        ax81 = fig.add_subplot(gs[8,1])
        ax81.plot(lat, data, 'o', markersize=1, color=color[iregion])
        ax81.set_xlabel('Lat [deg]')
        ax81.set_ylabel('Brightness [kR]')
        ax81.set_xlim(-90, 90)
        ax81.set_xticks(np.arange(-90, 91, 30))
        ax81.set_ylim(0, 50)
        ax81.set_yticks(np.arange(0, 51, 10))

    ax00.legend(loc=(1.04,0), markerscale=5)
    plt.tight_layout()
    #plt.show()

    if alt_default:
        os.makedirs(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/', exist_ok=True)
        plt.savefig(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/overview_globe_sza90_alt1.png', dpi=300)
    else:
        os.makedirs(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/', exist_ok=True)
        plt.savefig(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/overview_globe_sza90_alt2.png', dpi=300)



def plot_overview_altdiff():
    orbit_arr = np.arange(700, 5500)
    selec_region=[0,1,5]
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    for ith, iregion in enumerate(selec_region):
        data = []
        sza = []
        angle = []
        lat = []
        lon = []
        lt = []
        fsw = []
        euv_lya = []
        Ls = []
        timeDt = []

        data2 = []
        sza2 = []
        angle2 = []
        lat2 = []
        lon2 = []
        lt2 = []


        for iorbit in orbit_arr:
            dic = get_globe_data_region(iorbit, region=iregion, alt_default=True, median_br=False)
            dic2 = get_globe_data_region(iorbit, region=iregion, alt_default=False, median_br=False)

            if dic is None or dic2 is None:
                continue
            data.append(dic['data'])
            sza.append(dic['sza'])
            angle.append(dic['angle_efield'])
            lat.append(dic['lat'])
            lon.append(dic['lon'])
            lt.append(dic['lt'])

            data2.append(dic2['data'])
            sza2.append(dic2['sza'])
            angle2.append(dic2['angle_efield'])
            lat2.append(dic2['lat'])
            lon2.append(dic2['lon'])
            lt2.append(dic2['lt'])

            dic_iuvs, dic_sw, dic_euv = get_iuv_sw_euv_data(iorbit)
            if dic_sw is not None:
                vsw = dic_sw['vpsw']
                nsw = dic_sw['npsw']
                fsw.append(nsw * vsw * 1e5)
            else:
                fsw.append(np.nan)

            if dic_euv is not None:
                euv_lya.append(dic_euv['euv'][2])
            else:
                euv_lya.append(np.nan)

            if dic_iuvs is not None:
                Ls.append(dic_iuvs['Ls_mean'])
                timeDt.append(dic_iuvs['Dt_apo'])
            else:
                Ls.append(np.nan)
                timeDt.append(np.nan)

        data = np.array(data) - np.array(data2)
        sza = np.array(sza)
        angle = np.array(angle)
        lat = np.array(lat)
        lon = np.array(lon)
        lt = np.array(lt)
        fsw = np.array(fsw)
        euv_lya = np.array(euv_lya)
        Ls = np.array(Ls)
        timeDt = np.array(timeDt)


        if ith==0:
            fig = plt.figure(figsize=(20, 20))
            widths = [3, 1]
            gs = fig.add_gridspec(9,2, width_ratios=widths)#fig.add_gridspec(5, 2, height_ratios=heights)
            plt.subplots_adjust(hspace = 0.3)

        ax00 = fig.add_subplot(gs[0,0])
        ax00.plot(timeDt, data, 'o', markersize=1, label='region '+str(iregion), color=color[iregion])
        ax00.set_xlabel('time')
        ax00.set_ylabel('Brightness [kR]')
        ax00.set_ylim(-7.5, 7.5)
        ax00.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax10 = fig.add_subplot(gs[1,0])
        ax10.plot(timeDt, euv_lya, 'o', markersize=1, color=color[iregion])
        ax10.set_xlabel('time')
        ax10.set_ylabel('Ly-alpha irradiance [W/m2]')

        ax20 = fig.add_subplot(gs[2,0])
        ax20.plot(timeDt, fsw, 'o', markersize=1, color=color[iregion])
        ax20.set_xlabel('time')
        ax20.set_ylabel('Fsw [/cm2/s]')
        ax20.set_yscale('log')

        ax30 = fig.add_subplot(gs[3,0])
        ax30.plot(timeDt[angle>0], angle[angle>0], 'o', markersize=1, color=color[iregion])
        ax30.set_xlabel('time')
        ax30.set_ylabel('Angle wrt E-field [deg]')
        ax30.set_ylim(0, 180)
        ax30.set_yticks(np.arange(0, 181, 30))

        ax40 = fig.add_subplot(gs[4,0])
        ax40.plot(timeDt, Ls, 'o', markersize=1, color=color[iregion])
        ax40.set_xlabel('time')
        ax40.set_ylabel('Ls [deg]')
        ax40.set_ylim(-10, 370)
        ax40.set_yticks(np.arange(0, 361, 60))

        ax50 = fig.add_subplot(gs[5,0])
        ax50.plot(timeDt, sza, 'o', markersize=1, color=color[iregion])
        ax50.set_xlabel('time')
        ax50.set_ylabel('SZA [deg]')
        ax50.set_ylim(0, 180)
        ax50.set_yticks(np.arange(0, 181, 30))

        ax60 = fig.add_subplot(gs[6,0])
        ax60.plot(timeDt, lt, 'o', markersize=1, color=color[iregion])
        ax60.set_xlabel('time')
        ax60.set_ylabel('LT [deg]')
        ax60.set_ylim(0, 24)
        ax60.set_yticks(np.arange(0, 25, 4))

        ax70 = fig.add_subplot(gs[7,0])
        ax70.plot(timeDt, lon, 'o', markersize=1, color=color[iregion])
        ax70.set_xlabel('time')
        ax70.set_ylabel('Lon [deg]')
        ax70.set_ylim(-10, 370)
        ax70.set_yticks(np.arange(0, 361, 60))

        ax80 = fig.add_subplot(gs[8,0])
        ax80.plot(timeDt, lat, 'o', markersize=1, color=color[iregion])
        ax80.set_xlabel('time')
        ax80.set_ylabel('Lat [deg]')
        ax80.set_ylim(-90, 90)
        ax80.set_yticks(np.arange(-90, 91, 30))

        ax11 = fig.add_subplot(gs[1,1])
        ax11.plot(euv_lya, data, 'o', markersize=1, color=color[iregion])
        ax11.set_xlabel('Ly-alpha irradiance [W/m2]')
        ax11.set_ylabel('Brightness [kR]')
        ax11.set_ylim(-7.5, 7.5)
        ax11.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax21 = fig.add_subplot(gs[2,1])
        ax21.plot(fsw, data, 'o', markersize=1, color=color[iregion])
        ax21.set_xlabel('Fsw [/cm2/s]')
        ax21.set_ylabel('Brightness [kR]')
        ax21.set_xscale('log')
        ax21.set_ylim(-7.5, 7.5)
        ax21.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax31 = fig.add_subplot(gs[3,1])
        ax31.plot(angle[angle>0], data[angle>0], 'o', markersize=1, color=color[iregion])
        ax31.set_xlabel('Angle wrt E-field [deg]')
        ax31.set_ylabel('Brightness [kR]')
        ax31.set_xlim(0, 180)
        ax31.set_xticks(np.arange(0, 181, 30))
        ax31.set_ylim(-7.5, 7.5)
        ax31.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax41 = fig.add_subplot(gs[4,1])
        ax41.plot(Ls, data, 'o', markersize=1, color=color[iregion])
        ax41.set_xlabel('Ls [deg]')
        ax41.set_ylabel('Brightness [kR]')
        ax41.set_xlim(-10, 370)
        ax41.set_xticks(np.arange(0, 361, 60))
        ax41.set_ylim(-7.5, 7.5)
        ax41.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax51 = fig.add_subplot(gs[5,1])
        ax51.plot(sza, data, 'o', markersize=1, color=color[iregion])
        ax51.set_xlabel('SZA [deg]')
        ax51.set_ylabel('Brightness [kR]')
        ax51.set_xlim(0, 180)
        ax51.set_xticks(np.arange(0, 181, 30))
        ax51.set_ylim(-7.5, 7.5)
        ax51.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax61 = fig.add_subplot(gs[6,1])
        ax61.plot(lt, data, 'o', markersize=1, color=color[iregion])
        ax61.set_xlabel('LT [hour]')
        ax61.set_ylabel('Brightness [kR]')
        ax61.set_xlim(-0.5, 24.5)
        ax61.set_xticks(np.arange(0, 25, 4))
        ax61.set_ylim(-7.5, 7.5)
        ax61.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax71 = fig.add_subplot(gs[7,1])
        ax71.plot(lon, data, 'o', markersize=1, color=color[iregion])
        ax71.set_xlabel('Lon [deg]')
        ax71.set_ylabel('Brightness [kR]')
        ax71.set_xlim(-10, 370)
        ax71.set_xticks(np.arange(0, 361, 60))
        ax71.set_ylim(-7.5, 7.5)
        ax71.set_yticks(np.arange(-7.5, 7.6, 2.5))

        ax81 = fig.add_subplot(gs[8,1])
        ax81.plot(lat, data, 'o', markersize=1, color=color[iregion])
        ax81.set_xlabel('Lat [deg]')
        ax81.set_ylabel('Brightness [kR]')
        ax81.set_xlim(-90, 90)
        ax81.set_xticks(np.arange(-90, 91, 30))
        ax81.set_ylim(-7.5, 7.5)
        ax81.set_yticks(np.arange(-7.5, 7.6, 2.5))

    ax00.legend(loc=(1.04,0), markerscale=5)
    plt.tight_layout()
    #plt.show()


    os.makedirs(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/', exist_ok=True)
    plt.savefig(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/overview_globe_sza90_altdiff.png', dpi=300)



def plot_overview_altdiff_map():
    orbit_arr = np.arange(700, 5500)
    selec_region=[2,3,4]
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    for ith, iregion in enumerate(selec_region):
        data = []
        sza = []
        angle = []
        lat = []
        lon = []
        lt = []
        fsw = []
        euv_lya = []
        Ls = []
        timeDt = []

        data2 = []
        sza2 = []
        angle2 = []
        lat2 = []
        lon2 = []
        lt2 = []


        for iorbit in orbit_arr:
            dic = get_globe_data_region(iorbit, region=iregion, alt_default=True, median_br=False)
            dic2 = get_globe_data_region(iorbit, region=iregion, alt_default=False, median_br=False)

            if dic is None or dic2 is None:
                continue
            data.append(dic['data'])
            sza.append(dic['sza'])
            angle.append(dic['angle_efield'])
            lat.append(dic['lat'])
            lon.append(dic['lon'])
            lt.append(dic['lt'])

            data2.append(dic2['data'])
            sza2.append(dic2['sza'])
            angle2.append(dic2['angle_efield'])
            lat2.append(dic2['lat'])
            lon2.append(dic2['lon'])
            lt2.append(dic2['lt'])

            dic_iuvs, dic_sw, dic_euv = get_iuv_sw_euv_data(iorbit)
            if dic_sw is not None:
                vsw = dic_sw['vpsw']
                nsw = dic_sw['npsw']
                fsw.append(nsw * vsw * 1e5)
            else:
                fsw.append(np.nan)

            if dic_euv is not None:
                euv_lya.append(dic_euv['euv'][2])
            else:
                euv_lya.append(np.nan)

            if dic_iuvs is not None:
                Ls.append(dic_iuvs['Ls_mean'])
                timeDt.append(dic_iuvs['Dt_apo'])
            else:
                Ls.append(np.nan)
                timeDt.append(np.nan)

        data = np.array(data) - np.array(data2)
        sza = np.array(sza)
        angle = np.array(angle)
        lat = np.array(lat)
        lon = np.array(lon)
        lt = np.array(lt)
        fsw = np.array(fsw)
        euv_lya = np.array(euv_lya)
        Ls = np.array(Ls)
        timeDt = np.array(timeDt)


        if ith==0:
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(2,1)#fig.add_gridspec(5, 2, height_ratios=heights)
            #plt.subplots_adjust(hspace = 0.3)

        ax = fig.add_subplot(gs[0,0])
        ax.scatter(lon, lat, c=data, cmap=plt.get_cmap('coolwarm'), vmin=-5, vmax=5)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('lon [deg]')
        ax.set_ylabel('lat [deg]')

        ax2 = fig.add_subplot(gs[1,0])
        ax2.scatter(euv_lya, fsw, c=data, cmap=plt.get_cmap('coolwarm'), vmin=-5, vmax=5)
        ax2.set_xlim(2e-3, 4e-3)
        ax2.set_ylim(1e7, 1e10)
        ax2.set_yscale('log')
        ax2.set_xlabel('Lya irradiance [W/nm]')
        ax2.set_ylabel('SW flux [/cm2/s]')


    plt.tight_layout()
    #plt.show()


    os.makedirs(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/', exist_ok=True)
    plt.savefig(saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/overview/overview_map_globe_sza90_altdiff.png', dpi=300)
