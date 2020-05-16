import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
from iuvdata_l1b import get_apoapseinfo, ApoapseSwath, SzaGeo, LocalTimeGeo, LatLonGeo
from PyUVS.graphics import H_colormap
from variables import saveloc

def quicklook_apoapse_diff(orbit_number, wv0=121.6, wv_width=2.5, savefig=True):

    apoinfo = get_apoapseinfo(orbit_number)
    apoinfo_pre = get_apoapseinfo(orbit_number-1)
    fig, ax = plt.subplots(4, 2, figsize=(24, 16))
    if np.size(apoinfo.files) > 0 and np.size(apoinfo_pre.files) > 0:
        if apoinfo.n_swaths == apoinfo_pre.n_swaths:
            for ith_file, iswath_number in enumerate(apoinfo.swath_number):
                ## Read data of the main orbit
                hdul = apoinfo.get_hdul(ith_file)
                aposwath = ApoapseSwath(hdul, iswath_number, wv0, wv_width)

                ## Read data of the previous orbit
                hdul_pre = apoinfo_pre.get_hdul(ith_file)
                aposwath_pre = ApoapseSwath(hdul_pre, iswath_number, wv0, wv_width)

                ## Generate residual object
                aposwath_sub = aposwath.sub_obj(aposwath_pre)

                ## Read geometiries
                szageo = SzaGeo(hdul, iswath_number)
                ltgeo = LocalTimeGeo(hdul, iswath_number)
                latlongeo = LatLonGeo(hdul, iswath_number)

                # Plot images
                mesh00 = aposwath.plot(ax[0][0], cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=30))
                mesh10 = aposwath_pre.plot(ax[1][0], cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=30))
                mesh20 = aposwath_sub.plot(ax[2][0], cmap='coolwarm', vmin=-10, vmax=10)
                mesh30 = aposwath_sub.plot(ax[3][0], cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=10))

                # Plot geometries
                mesh01 = szageo.plot(ax[0][1], cmap=plt.get_cmap('magma_r', 18))
                mesh11 = ltgeo.plot(ax[1][1], cmap=plt.get_cmap('twilight_shifted', 24))
                mesh21 = latlongeo.plot_lat(ax[2][1], cmap=plt.get_cmap('coolwarm', 18))
                mesh31 = latlongeo.plot_lon(ax[3][1], cmap=plt.get_cmap('twilight', 36))

            ## All setttings
            [[jax.set_xlabel('Slit bins') for jax in iax] for iax in ax]
            [[jax.set_ylabel('Integration bins') for jax in iax] for iax in ax]

            ## Individual setttings
            ax[0][0].set_title('Orbit ' + str(orbit_number) + ' Apoapse ' + str(wv0) + ' nm')
            cb00 = plt.colorbar(mesh00, ax=ax[0][0])
            cb00.set_label('Brightness [kR]')

            ax[1][0].set_title('Previous orbit Apoapse ' + str(wv0) + ' nm')
            cb10 = plt.colorbar(mesh10, ax=ax[1][0])
            cb10.set_label('Brightness [kR]')

            ax[2][0].set_title('Risidual  Apoapse ' + str(wv0) + ' nm')
            cb20 = plt.colorbar(mesh20, ax=ax[2][0])
            cb20.set_label('Brightness [kR]')

            ax[3][0].set_title('Risidual Apoapse ' + str(wv0) + ' nm')
            cb30 = plt.colorbar(mesh30, ax=ax[3][0])
            cb30.set_label('Brightness [kR]')

            ax[0][1].set_title('Orbit ' + str(orbit_number) + ' SZA')
            cb01 = plt.colorbar(mesh01, ax=ax[0][1])
            cb01.set_label('SZA [deg]')

            ax[1][1].set_title('Orbit ' + str(orbit_number) + ' Local Time')
            cb11 = plt.colorbar(mesh11, ax=ax[1][1])
            cb11.set_label('Local Time [hour]')

            ax[2][1].set_title('Orbit ' + str(orbit_number) + ' Latitude')
            cb22 = plt.colorbar(mesh21, ax=ax[2][1])
            cb22.set_label('Latitude [deg]')

            ax[3][1].set_title('Orbit ' + str(orbit_number) + ' Latitude')
            cb33 = plt.colorbar(mesh31, ax=ax[3][1])
            cb33.set_label('Longitude [deg]')

            plt.tight_layout()
            savepath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/diff/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            fname_save = 'orbit_' + '{:05d}'.format(orbit_number) + '.png'
            if savefig:
                plt.savefig(savepath + fname_save)
        else:
            print('----- Number of swaths do not match. Skips this orbit. -----')
    else:
        print('----- File(s) do not exist. Skips this orbit. -----')

if __name__ == '__main__':
    start_orbit = int(sys.argv[1])
    n_orbit = int(sys.argv[2])
    orbit_arr = np.arange(n_orbit) + start_orbit
    for iorbit_number in np.arange(n_orbit) + start_orbit:
        quicklook_apoapse_diff(iorbit_number)
        plt.close()
