import numpy as np
import os, sys
import matplotlib.pyplot as plt

from maven_iuvs.constants import R_Mars_km as Rm
from variables import saveloc

def plot(orbit_number, savefig=True):

    dicpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data/all/npy/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/'
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
        xmesh, ymesh = np.meshgrid(x, y)
        rmesh = np.sqrt(xmesh**2 + ymesh**2)

        # set bool for limb regions
        nedge = 71
        binsize = 80 #km
        rbin_edges = np.arange(nedge) * binsize
        rbin = (rbin_edges[0:nedge-1] + rbin_edges[1:nedge])/2
        bool_rbin = [(rmesh>rbin_edges[i])&(rmesh<=rbin_edges[i+1]) for i in range(nedge-1)]
        bool_szaxy = [(sza_xy>=-180)&(sza_xy<-150), (sza_xy>=-150)&(sza_xy<-120), (sza_xy>=-120)&(sza_xy<-90), (sza_xy>=-90)&(sza_xy<-60), (sza_xy>=-60)&(sza_xy<-30),(sza_xy>=-30)&(sza_xy<0),
                      (sza_xy>=0)&(sza_xy<30), (sza_xy>=30)&(sza_xy<60), (sza_xy>=60)&(sza_xy<90), (sza_xy>=90)&(sza_xy<120), (sza_xy>=120)&(sza_xy<150), (sza_xy>=150)&(sza_xy<180)]
        br = [[np.nanmedian(data[ibool_r & ibool_xy]) for ibool_r in bool_rbin] for ibool_xy in bool_szaxy]
        fig, ax = plt.subplots(1,1, figsize=(8, 6))
        ax.set_ylim(0,10)
        ax.set_xlim(0, 5000)
        ax.set_ylabel('Brightness [kR]')
        ax.set_xlabel('Distance from center [km]')
        ax.set_title('Orbit_' + '{:05d}'.format(orbit_number))
        [ax.plot(rbin,ibr, label=ith) for ith, ibr in enumerate(br)]

        colormap = plt.cm.seismic#RdYlBu_r#gist_ncar #nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
        for i,j in enumerate(ax.lines):
            j.set_color(colors[i])

        ax.axvline(Rm, color='grey', linewidth=1, linestyle='-')
        ax.legend()

        if savefig:
            savepath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data/all/regions_disk_limb/orbit_'+ '{:05d}'.format(orbit_number//100 * 100) + '/'
            os.makedirs(savepath, exist_ok=True)
            savename = 'orbit_' + '{:05d}'.format(orbit_number)
            plt.savefig(savepath+savename)

if __name__ == '__main__':
    sorbit = int(sys.argv[1])
    norbit = int(sys.argv[2])
    eorbit = sorbit + norbit
    orbit_arr = range(sorbit, eorbit)#[849]
    for iorbit_number in orbit_arr:
        print('Orbit_'+'{:05d}'.format(iorbit_number))
        plot(iorbit_number)
        plt.close()
