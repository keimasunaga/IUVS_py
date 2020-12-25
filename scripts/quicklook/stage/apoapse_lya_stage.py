import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
from iuvdata_l1b import get_apoapseinfo, ApoapseSwath, SzaGeo, LocalTimeGeo, LatLonGeo, AltGeo
from maven_iuvs.graphics import H_colormap
from variables import saveloc
from iuvtools.time import get_timeDt, Dt2str
from iuvtools.info import get_solar_lon

def quicklook_apoapse(orbit_number, wv0=121.6, wv_width=2.5, savefig=True):

    apoinfo = get_apoapseinfo(orbit_number, product_type='stage')
    fig, ax = plt.subplots(3, 2, figsize=(24, 16))

    if len(apoinfo.files)>0:
        for ith_file, iswath_number in enumerate(apoinfo.swath_number):
            hdul = apoinfo.get_hdul(ith_file)
            if ith_file==0:
                time0 = Dt2str(get_timeDt(hdul)[0])
                Ls0 = get_solar_lon(hdul)

            aposwath = ApoapseSwath(hdul, iswath_number, wv0, wv_width)
            szageo = SzaGeo(hdul, iswath_number)
            ltgeo = LocalTimeGeo(hdul, iswath_number)
            latlongeo = LatLonGeo(hdul, iswath_number)
            altgeo = AltGeo(hdul, iswath_number)
            #mesh0 = aposwath.plot(ax0, cmap=H_colormap(), norm=mpl.colors.LogNorm(vmin=1e-1, vmax=50))
            mesh0 = aposwath.plot(ax[0,0], cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=5))
            mesh1 = szageo.plot(ax[1,0], cmap=plt.get_cmap('magma_r', 18))
            mesh2 = ltgeo.plot(ax[2,0], cmap=plt.get_cmap('twilight_shifted', 24))
            mesh3 = altgeo.plot(ax[0,1], cmap=plt.get_cmap('bone'))
            mesh4 = latlongeo.plot_lon(ax[1,1], cmap=plt.get_cmap('twilight_shifted', 36))
            mesh5 = latlongeo.plot_lat(ax[2,1], cmap=plt.get_cmap('coolwarm', 18))

        Dt_apo = Dt2str(get_Dt_apo(orbit_number))
        fig.suptitle('Orbit ' + '{:05d}'.format(orbit_number)+ '\n'+Dt_apo  + '\n Ls=' + '{:.1f}'.format(Ls0), y=0.95, fontsize=16)

        #fig.suptitle(time0 + ', Orbit ' + '{:05d}'.format(orbit_number), fontsize=12)
        ax[0,0].set_title(' Orbit ' + '{:05d}'.format(orbit_number) + ' Apoapse ' + str(wv0) + ' nm')
        ax[0,0].set_xlabel('Spatial bins')
        ax[0,0].set_ylabel('Integrations')
        cb0 = plt.colorbar(mesh0, ax=ax[0,0])
        cb0.set_label('Brightness [kR]')

        ax[1,0].set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' SZA')
        ax[1,0].set_xlabel('Spatial bins')
        ax[1,0].set_ylabel('Integrations')
        cb1 = plt.colorbar(mesh1, ax=ax[1,0])
        cb1.set_label('SZA [deg]')

        ax[2,0].set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' Local Time')
        ax[2,0].set_xlabel('Spatial bins')
        ax[2,0].set_ylabel('Integrations')
        cb2 = plt.colorbar(mesh2, ax=ax[2,0])
        cb2.set_label('Local Time [hour]')


        ax[0,1].set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' Altitude')
        ax[0,1].set_xlabel('Spatial bins')
        ax[0,1].set_ylabel('Integrations')
        cb3 = plt.colorbar(mesh3, ax=ax[0,1])
        cb3.set_label('Altitude [km]')

        ax[1,1].set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' Longitude')
        ax[1,1].set_xlabel('Spatial bins')
        ax[1,1].set_ylabel('Integrations')
        cb4 = plt.colorbar(mesh4, ax=ax[1,1])
        cb4.set_label('Latitude [deg]')

        ax[2,1].set_title('Orbit ' + '{:05d}'.format(orbit_number) + ' Latitude')
        ax[2,1].set_xlabel('Spatial bins')
        ax[2,1].set_ylabel('Integrations')
        cb5 = plt.colorbar(mesh5, ax=ax[2,1])
        cb5.set_label('Latitude [deg]')

        fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        #plt.tight_layout()
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
