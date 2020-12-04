import numpy as np
import matplotlib.pyplot as plt
import os, sys
import matplotlib as mpl

from variables import saveloc
from PyUVS.graphics import H_colormap

def quicklook_apoapse_globe_median(orbit_number, orbit_half_width=5):
    orbit_arr = np.array([iorb for iorb in range(orbit_number-orbit_half_width, orbit_number+orbit_half_width+1)])
    img_arr = []
    n_img = 0
    for iorbit in orbit_arr:
        path = saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/orbit_{:05d}'.format(orbit_number//100 * 100)+'/npy/'
        fname = 'orbit_{:05d}'.format(orbit_number) + '.npy'
        print(os.path.isfile(path+fname))
        if os.path.isfile(path+fname):
            dicAll = np.load(path + fname, allow_pickle=True).item()
            dic_iuvs = dicAll['dic_iuvs']
            img_arr.append(dic_iuvs['data'])
            if iorbit == orbit_number:
                x = dic_iuvs['x']
                y = dic_iuvs['y']
            n_img += 1

    if n_img >= orbit_half_width + orbit_half_width/2:
        img_arr = np.array(img_arr)
        img_mean = np.nanmean(img_arr, axis=0)
        img_med = np.nanmedian(img_arr, axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(x, y, img_med, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2))
        cb = plt.colorbar(mesh)
        ax.set_xlabel('[km]')
        ax.set_ylabel('[km]')
        dic_save = {'orbit_number':orbit_number, 'orbit_half_width':orbit_half_width, 'orbit_arr':orbit_arr,
                    'img_mean':img_mean, 'img_median':img_med, 'n_img':n_img, 'x':x, 'y':y}

        savepath = saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/orbit_{:05d}'.format(orbit_number//100 * 100)+'/median_img/npy/'
        savename = 'orbit_{:05d}'.format(orbit_number) + '.npy'
        os.makedirs(savepath, exist_ok=True)
        np.save(savepath+savename, dic_save)

        figpath = saveloc+'quicklook/apoapse_l1b/Lyman-alpha/globe_data_sza90/orbit_{:05d}'.format(orbit_number//100 * 100)+'/median_img/fig/'
        figname = 'orbit_{:05d}'.format(orbit_number) + '.png'
        os.makedirs(figpath, exist_ok=True)
        plt.savefig(figpath + figname)


if __name__ == '__main__':
    start_orbit = int(sys.argv[1])
    n_orbit = int(sys.argv[2])
    orbit_arr = np.arange(n_orbit) + start_orbit
    orbit_half_width = 5
    for iorbit_number in np.arange(n_orbit) + start_orbit:
        quicklook_apoapse_globe_median(iorbit_number, orbit_half_width)
        plt.close()
