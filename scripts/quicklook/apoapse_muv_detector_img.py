import sys, os
import numpy as np
import matplotlib.pyplot as plt

from iuvtools.info import get_mcp_volt
from iuvdata_l1b import get_apoapseinfo, ApoapseSwath
from iuvtools.data import get_wv, get_counts_detector_img, primary_is_nan, echelle_place_ok
from variables import saveloc

def quicklook_detector_img(orbit_number):
    apoinfo = get_apoapseinfo(orbit_number)
    for ith_file, iswath_number in enumerate(apoinfo.swath_number):
        hdul = apoinfo.get_hdul(ith_file)
        if primary_is_nan(hdul): # skip if all data is nan
            continue

        # get data from hdul
        dimg = get_counts_detector_img(hdul)
        wv = get_wv(hdul)
        dimg_mean = np.nanmean(dimg, axis=0)
        wv_mean = np.nanmean(wv,axis=0)

        # judge if echelle place is ok
        echelle_ok = echelle_place_ok(hdul)

        # plot
        plt.close()
        fig = plt.figure(figsize=(4, 8))
        ax = fig.add_subplot(211)
        ax.imshow(dimg, aspect='auto')
        ax.set_xlabel('spectral pixel')
        ax.set_ylabel('spatial pixel')
        ax2 = fig.add_subplot(212)
        ax2.plot(wv_mean, dimg_mean)
        ax2.set_xlim(min(wv_mean), max(wv_mean))
        ax2.set_ylim(-1e4, 1e5)
        ax2.set_xlabel('wavelength [nm]')
        ax2.set_ylabel('counts')
        xmax2 = max(ax2.get_xlim())
        ymax2 = max(ax2.get_ylim())
        ax2.text(xmax2*0.8, ymax2*0.9, 'echell_ok: '+str(echelle_ok))

        # save figures
        fname = (apoinfo.files[ith_file]).split('/')[-1]
        fname2 = fname.split('.')[-3]
        figpath = saveloc + 'quicklook/apoapse_l1b/detector_img/orbit_' + '{:05}'.format(apoinfo.orbit_number//100*100)+'/orbit_' + '{:05}'.format(apoinfo.orbit_number) + '/'
        if not os.path.exists(figpath):
            os.makedirs(figpath, exist_ok=True)
        plt.savefig(figpath + fname2 + '.png')

if __name__ == '__main__':
    start_orbit = int(sys.argv[1])
    n_orbit = int(sys.argv[2])
    orbit_arr = np.arange(n_orbit) + start_orbit
    for iorbit_number in np.arange(n_orbit) + start_orbit:
        quicklook_detector_img(iorbit_number)
        plt.close()
