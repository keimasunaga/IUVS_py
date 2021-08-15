def get_mean_br(data, altbin_edges, sza_xy_edges):
    alt_bin_edges[0], alt_bin_edges[1]

import numpy as np
import os, sys
import matplotlib.pyplot as plt

from maven_iuvs.constants import R_Mars_km as Rm
from variables import saveloc


def plot_mult_orbits(orbit_number, orbit_half_width=5, savefig=True):
    orbit_arr = np.array([iorb for iorb in range(orbit_number-orbit_half_width, orbit_number+orbit_half_width+1)])
    norbit = orbit_half_width * 2 + 1
    #orbits_found = []
    fig, ax = plt.subplots(6,2, figsize=(18, 20))
    fig.suptitle('Orbit ' + '{:05d}'.format(orbit_number)+' and surroundings' , y=0.95)

    dicpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data/all/npy/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/'
    dname_save_center = 'orbit_' + '{:05d}'.format(orbit_number) + '.npy'
    if os.path.isfile(dicpath+dname_save_center):

        for iorbit_number in orbit_arr:
            dicpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data/all/npy/orbit_' + '{:05d}'.format(iorbit_number//100 * 100) + '/'
            dname_save = 'orbit_' + '{:05d}'.format(iorbit_number) + '.npy'
            if os.path.isfile(dicpath+dname_save):
                #orbits_found.append(iorbit_number)
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
                xmesh, ymesh = np.meshgrid(x, y)
                rmesh = np.sqrt(xmesh**2 + ymesh**2)

                # set bool for limb regions
                nedge = 71
                delta_r = 80
                rbin_edges = np.arange(nedge) * delta_r
                rbin = (rbin_edges[0:nedge-1] + rbin_edges[1:nedge])/2
                bool_rbin = [(rmesh>rbin_edges[i])&(rmesh<=rbin_edges[i+1]) for i in range(nedge-1)]

                #bool_szaxy = [np.abs(sza_xy)>150, (sza_xy>=-150)&(sza_xy<-90), (sza_xy>=-90)&(sza_xy<-30),
                #             (sza_xy>=-30)&(sza_xy<30), (sza_xy>=30)&(sza_xy<90), (sza_xy>=90)&(sza_xy<150)]
                bool_szaxy = [(sza_xy>=-180)&(sza_xy<-150), (sza_xy>=-150)&(sza_xy<-120), (sza_xy>=-120)&(sza_xy<-90), (sza_xy>=-90)&(sza_xy<-60), (sza_xy>=-60)&(sza_xy<-30),(sza_xy>=-30)&(sza_xy<0),
                              (sza_xy>=0)&(sza_xy<30), (sza_xy>=30)&(sza_xy<60), (sza_xy>=60)&(sza_xy<90), (sza_xy>=90)&(sza_xy<120), (sza_xy>=120)&(sza_xy<150), (sza_xy>=150)&(sza_xy<180)]
                br = [[np.nanmedian(data[ibool_r & ibool_xy]) for ibool_r in bool_rbin] for ibool_xy in bool_szaxy]


                for ith, ibr in enumerate(br):
                    irow, icol = ith%6, int(ith/6)
                    if icol==0:
                        ax[irow,icol].set_title('region '+str(ith))
                        ax[irow,icol].set_xlim(-500, 5000)
                        ax[irow,icol].set_ylim(0, 15)
                        ax[irow,icol].set_xlabel('distance [km]')
                        ax[irow,icol].set_ylabel('brightness [R]')
                        #ic for ic in plt.get_cmap()
                        ax[irow,icol].plot(rbin,ibr, label=iorbit_number)
                        ax[irow,icol].axvline(Rm, color='grey', linewidth=1, linestyle='-')
                        #ax.legend()
                    elif icol==1:
                        ax[5-irow,icol].set_title('region '+str(ith))
                        ax[5-irow,icol].set_xlim(-500, 5000)
                        ax[5-irow,icol].set_ylim(0, 15)
                        ax[5-irow,icol].set_xlabel('distance [km]')
                        ax[5-irow,icol].set_ylabel('brightness [R]')
                        #ic for ic in plt.get_cmap()
                        ax[5-irow,icol].plot(rbin,ibr, label=iorbit_number)
                        ax[5-irow,icol].axvline(Rm, color='grey', linewidth=1, linestyle='-')
                        #ax.legend()


        colormap = plt.cm.coolwarm#RdYlBu_r#gist_ncar #nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 1,len(ax[irow,icol].lines))]
        [[j.set_color(colors[i]) for i,j in enumerate(ax[k%6,int(k/6)].lines) if i<norbit] for k in range(12)]
        [[jax.legend(loc=2) for jax in iax] for iax in ax]

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if savefig:
            savepath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data/all/lines_disk_limb/median/orbit_'+ '{:05d}'.format(orbit_number//100 * 100) + '/'
            os.makedirs(savepath, exist_ok=True)
            savename = 'mult_orbit_' + '{:05d}'.format(orbit_number)
            plt.savefig(savepath+savename)
    else:
        print('No data found at orbit ' + '{:05d}'.format(orbit_number))

if __name__ == '__main__':
    orbit_half_width = 5
    sorbit = int(sys.argv[1])
    norbit = int(sys.argv[2])
    eorbit = sorbit + norbit
    orbit_arr = range(sorbit, eorbit)#[849]
    error_orbit = []
    for iorbit_number in orbit_arr:
        print('{:05d}'.format(iorbit_number))
        plot_mult_orbits(iorbit_number, orbit_half_width=orbit_half_width, savefig=True)
        plt.close()
