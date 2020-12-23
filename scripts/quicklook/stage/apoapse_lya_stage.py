import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
from iuvdata_l1b import get_apoapseinfo, ApoapseSwath, SzaGeo, LocalTimeGeo, LatLonGeo
from maven_iuvs.graphics import H_colormap
from variables import saveloc
from iuvtools.time import get_timeDt, Dt2str

def quicklook_apoapse(orbit_number, wv0=121.6, wv_width=2.5, savefig=True):

    apoinfo = get_apoapseinfo(orbit_number, product_type='stage')
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(12, 16))

    if len(apoinfo.files)>0:
        for ith_file, iswath_number in enumerate(apoinfo.swath_number):
            hdul = apoinfo.get_hdul(ith_file)
            if ith_file==0:
                time0 = Dt2str(get_timeDt(hdul)[0])
            aposwath = ApoapseSwath(hdul, iswath_number, wv0, wv_width)
            szageo = SzaGeo(hdul, iswath_number)
            ltgeo = LocalTimeGeo(hdul, iswath_number)
            latlongeo = LatLonGeo(hdul, iswath_number)
            #mesh0 = aposwath.plot(ax0, cmap=H_colormap(), norm=mpl.colors.LogNorm(vmin=1e-1, vmax=50))
            mesh0 = aposwath.plot(ax0, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=5))
            mesh1 = szageo.plot(ax1, cmap=plt.get_cmap('magma_r', 18))
            mesh2 = ltgeo.plot(ax2, cmap=plt.get_cmap('twilight_shifted', 24))
            mesh3 = latlongeo.plot_lat(ax3, cmap=plt.get_cmap('coolwarm', 18))
            #mesh3 = latlongeo.plot_lon(ax3, cmap=plt.get_cmap('twilight_shifted', 36))

        #fig.suptitle(time0 + ', Orbit ' + '{:05d}'.format(orbit_number), fontsize=12)
        ax0.set_title(time0 + ' Orbit ' + '{:05d}'.format(orbit_number) + ' Apoapse ' + str(wv0) + ' nm')
        ax0.set_xlabel('Spatial bins')
        ax0.set_ylabel('Integrations')
        cb0 = plt.colorbar(mesh0, ax=ax0)
        cb0.set_label('Brightness [kR]')

        ax1.set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' SZA')
        ax1.set_xlabel('Spatial bins')
        ax1.set_ylabel('Integrations')
        cb1 = plt.colorbar(mesh1, ax=ax1)
        cb1.set_label('SZA [deg]')

        ax2.set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' Local Time')
        ax2.set_xlabel('Spatial bins')
        ax2.set_ylabel('Integrations')
        cb2 = plt.colorbar(mesh2, ax=ax2)
        cb2.set_label('Local Time [hour]')

        ax3.set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' Latitude')
        ax3.set_xlabel('Spatial bins')
        ax3.set_ylabel('Integrations')
        cb3 = plt.colorbar(mesh3, ax=ax3)
        cb3.set_label('Latitude [deg]')

        plt.tight_layout()
        savepath = saveloc + 'quicklook/stage/apoapse_l1b/Lyman-alpha/orbits/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        fname_save = 'orbit_' + '{:05d}'.format(orbit_number) + '.png'
        if savefig:
            plt.savefig(savepath + fname_save)

if __name__ == '__main__':
    start_orbit = int(sys.argv[1])
    n_orbit = int(sys.argv[2])
    orbit_arr = np.arange(n_orbit) + start_orbit
    for iorbit_number in np.arange(n_orbit) + start_orbit:
        quicklook_apoapse(iorbit_number)
        plt.close()
