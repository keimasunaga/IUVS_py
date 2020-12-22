import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, sys

from variables import saveloc
from PyUVS.graphics import H_colormap
from iuvtools.time import Dt2str


def quicklook_apoapse_median_diff(orbit_number):
    path = saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza_all/orbit_{:05d}'.format(orbit_number//100 * 100)+'/npy/'
    fname = 'orbit_{:05d}'.format(orbit_number) + '.npy'
    path_med = saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza_all/orbit_{:05d}'.format(orbit_number//100 * 100)+'/median_img/npy/'
    fname_med = 'orbit_{:05d}'.format(orbit_number) + '.npy'
    if os.path.isfile(path+fname) and os.path.isfile(path_med+fname_med):
        print(orbit_number)
        dic = np.load(path+fname, allow_pickle=True).item()
        dic_med = np.load(path_med+fname_med, allow_pickle=True).item()

        img = dic['dic_iuvs']['data']
        x = dic['dic_iuvs']['x']
        y = dic['dic_iuvs']['y']
        img_med = dic_med['img_median']
        diff = img - img_med

        alt = dic['dic_iuvs']['alt']
        sza_xy = dic['dic_iuvs']['sza_xy']
        sza = dic['dic_iuvs']['sza']
        lt = dic['dic_iuvs']['lt']
        lat = dic['dic_iuvs']['lat']
        lon = dic['dic_iuvs']['lon']
        Dt_apo = dic['dic_iuvs']['Dt_apo']
        sc_sza_apo = dic['dic_iuvs']['sc_sza_apo']
        Ls_mean = dic['dic_iuvs']['Ls_mean']
        efield_angle = dic['dic_iuvs']['efield_angle']
        sza_xy[(alt>0)|np.isnan(img)] = np.nan
        sza[(alt>0)|np.isnan(img)] = np.nan
        lt[(alt>0)|np.isnan(img)] = np.nan
        lon[(alt>0)|np.isnan(img)] = np.nan
        lat[(alt>0)|np.isnan(img)] = np.nan
        efield_angle[(alt>0)|np.isnan(img)] = np.nan
        med_max = np.nanmax(img_med)
        timestring_apo = Dt2str(Dt_apo)

        plt.close()
        fig, ax = plt.subplots(4,2,figsize=(9,12))
        fig.suptitle('Orbit ' + '{:05d}'.format(orbit_number)+ '\n'+timestring_apo + '\n SZA_SC=' + '{:.1f}'.format(sc_sza_apo) + '\n Ls=' + '{:.1f}'.format(Ls_mean), y=0.95)

        ax[0,0].set_title('Orbit ' + '{:05d}'.format(orbit_number))
        mesh0 = ax[0,0].pcolormesh(x, y, img, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=med_max))
        ax[0,0].set_aspect(1)
        cb0 = plt.colorbar(mesh0, ax=ax[0,0])
        cb0.set_label('Brightness [kR]', labelpad=10, rotation=270)

        ax[1,0].set_title('Median')
        mesh1 = ax[1,0].pcolormesh(x, y, img_med, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=med_max))
        cb1 = plt.colorbar(mesh1, ax=ax[1,0])
        ax[1,0].set_aspect(1)
        cb1.set_label('Brightness [kR]', labelpad=10, rotation=270)

        ax[2,0].set_title('Diff')
        mesh2 = ax[2,0].pcolormesh(x, y, diff, cmap='coolwarm', vmin=-1, vmax=1)#norm=mpl.colors.SymLogNorm(linthresh=1, vmin=-2, vmax=2))
        cb2 = plt.colorbar(mesh2, ax=ax[2,0])
        ax[2,0].set_aspect(1)
        cb2.set_label('Brightness [kR]', labelpad=10, rotation=270)

        ax[3,0].set_title('Angle wrt E-field')
        mesh3 = ax[3,0].pcolormesh(x, y, efield_angle, vmin=0, vmax=180, cmap=plt.get_cmap('bwr_r', 18))
        cb3 = plt.colorbar(mesh3, ax=ax[3,0])
        ax[3,0].set_aspect(1)
        cb3.set_label('[degree]', labelpad=10, rotation=270)


        ax[0,1].set_title('SZA')
        mesh4 = ax[0,1].pcolormesh(x, y, sza, vmin=0, vmax=180, cmap=plt.get_cmap('magma_r', 18))
        cb4 = plt.colorbar(mesh4, ax=ax[0,1])
        ax[0,1].set_aspect(1)
        cb4.set_label('[degree]', labelpad=10, rotation=270)

        ax[1,1].set_title('Local Time')
        mesh5 = ax[1,1].pcolormesh(x, y, lt, vmin=0, vmax=24, cmap=plt.get_cmap('twilight_shifted', 24))
        cb5 = plt.colorbar(mesh5, ax=ax[1,1])
        ax[1,1].set_aspect(1)
        cb5.set_label('[hour]', labelpad=10, rotation=270)

        ax[2,1].set_title('Longitude')
        mesh6 = ax[2,1].pcolormesh(x, y, lon, vmin=0, vmax=360, cmap=plt.get_cmap('twilight', 36))
        cb6 = plt.colorbar(mesh6, ax=ax[2,1])
        ax[2,1].set_aspect(1)
        cb6.set_label('[degree]', labelpad=10, rotation=270)

        ax[3,1].set_title('Latitude')
        mesh7 = ax[3,1].pcolormesh(x, y, lat, vmin=-90, vmax=90, cmap=plt.get_cmap('coolwarm', 18))
        cb7 = plt.colorbar(mesh7, ax=ax[3,1])
        ax[3,1].set_aspect(1)
        cb7.set_label('[degree]', labelpad=10, rotation=270)

        #divider3 = make_axes_locatable(ax[3,0])
        #cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        #cb3 = plt.colorbar(mesh3, cax=cax3)

        [[jax.set_xlabel('[km]') for jax in iax] for iax in ax]
        [[jax.set_ylabel('[km]') for jax in iax] for iax in ax]
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        figpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza_all/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/median_diff/fig/'
        savename = 'orbit_' + '{:05d}'.format(orbit_number)
        if not os.path.exists(figpath):
            os.makedirs(figpath)
        plt.savefig(figpath+savename)

if __name__ == '__main__':
    start_orbit = int(sys.argv[1])
    n_orbit = int(sys.argv[2])
    orbit_arr = np.arange(n_orbit) + start_orbit
    for iorbit_number in orbit_arr:
        quicklook_apoapse_median_diff(iorbit_number)
        plt.close()
