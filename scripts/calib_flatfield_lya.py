import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from flatfields import FlatFieldLya
from variables import saveloc


def plot_spa_dist(fflya, subplots=False, **kwarg):
    fig, ax = plt.subplots(1,1, figsize=(10, 6))
    ax.set_title('Flatfield data')
    ax.set_xlabel('spa_bin')
    fflya.plot_spa_dist(ax)
    ax.set_xlim(0,1024)
    ax.set_ylim(0.7, 1.3)
    path = saveloc + 'calib/flatfield/lya/'
    save_name = path + 'fflya_spa_dist_v0'
    os.makedirs(path, exist_ok=True)
    plt.savefig(save_name)

def save_flatfield_polyfit(fflya, fitdeg=6, focused_lim=[80, 920]):
    # Delete nan values and sort data
    norm_ravel = fflya.norm.ravel()
    print(norm_ravel, norm_ravel.shape)
    spa = np.array([fflya.spa_cent for i in range(fflya.norm.shape[1])]).T.ravel()
    spa_nonnan = spa[~np.isnan(norm_ravel)]
    norm_nonnan = norm_ravel[~np.isnan(norm_ravel)]
    idx_sort = np.argsort(spa_nonnan)
    spa_sort = spa_nonnan[idx_sort]
    norm_sort = norm_nonnan[idx_sort]
    print(spa_sort, norm_sort)

    # fit data, get a polynomial function, and save the flatfield
    p = np.polyfit(spa_sort, norm_sort , fitdeg)
    func_poly = np.poly1d(p)
    x = np.arange(1024)
    poly_curv = func_poly(x)
    ff = poly_curv/np.mean(poly_curv[focused_lim[0]:focused_lim[1]])
    ff[:focused_lim[0]] = np.nan
    ff[focused_lim[1]:] = np.nan
    #spamin = np.nanmin(spa_sort)
    #spamax = np.nanmax(spa_sort)
    #ff[(ff<spamin)|(ff>spamax)] = np.nan
    path = saveloc + 'calib/flatfield/lya/'
    save_name = path + 'fflya_v1_nan'
    np.save(save_name, ff)

    # plot
    fig, ax = plt.subplots(1,1, figsize=(10, 6))
    ax.set_title('polynomial fit result: deg='+str(fitdeg))
    ax.set_xlabel('spa_bin')
    ax.plot(spa, norm_ravel, 'o')
    ax.plot(x, ff)
    ax.set_xlim(0,1024)
    ax.set_ylim(0.7, 1.3)
    figname = path + 'poly_fflya_v1_nan'
    plt.savefig(figname)

def save_flatfield_rolling_avg(fflya, spa_window=20):
    norm_mean = np.zeros(1024)
    norm_mean[0:spa_window] = np.nan
    norm_mean[1024-spa_window:1024] = np.nan
    for ispa in range(spa_window, 1024-spa_window):
        spalim = [ispa-spa_window, ispa+spa_window]
        idxspa = np.where((fflya.spa_cent>=spalim[0])&(fflya.spa_cent<spalim[1]))
        if np.size(idxspa) == 0:
            norm_mean[ispa] = np.nan
            continue
        norm_selec = fflya.norm[idxspa]
        med = np.nanmedian(norm_selec)
        MAD = np.nanmedian(np.abs(norm_selec - med))
        MADlim = [med-5*MAD, med+5*MAD]
        idx_inMAD = np.where((norm_selec>MADlim[0])&(norm_selec<MADlim[1]))
        norm_mean[ispa] = np.nanmean(norm_selec[idx_inMAD])

    idx_minmax = [int(np.min(fflya.spa_cent)), int(np.max(fflya.spa_cent))]
    norm_mean[:idx_minmax[0]] = np.nan
    norm_mean[idx_minmax[1]:1024] = np.nan
    path = saveloc + 'calib/flatfield/lya/'
    save_name = path + 'fflya_rollavg_v0'
    np.save(save_name, norm_mean)

    fig, ax = plt.subplots(1,1, figsize=(10, 6))
    fflya.plot_spa_dist(ax)
    ax.plot(np.arange(1024), norm_mean, '-Dg', zorder=np.size(fflya.wv))
    ax.set_xlim(0,1024)
    ax.set_ylim(0.7, 1.3)


if __name__ == '__main__':

    fname_list = ['bet_cma_tot_ff_fuv_orbit05170-5290_11bin.sav',
                  'tau_sco_tot_ff_fuv_orbit05306-5396_11bin_200kmfilter.sav',
                  'alp_cru_ff_fuv_2018_orbit06394-6524_11bin.sav',
                  'tau_sco_tot_ff_fuv_orbit06716-6802.sav']
                  #'alpcma_tot_ff_fuv_orbit08040-08174_v2.sav']

    orb_mids = [5230, 5306, 6394, 6716] #, 8040]

    for i, ifname in enumerate(fname_list):
        if i==0:
            fflya = FlatFieldLya(ifname, orb_mids[i])
            fflya.fill_nan_near_wv0()
        if i>0:
            fflya_tmp = FlatFieldLya(ifname, orb_mids[i])
            fflya_tmp.fill_nan_near_wv0()
            fflya.append_data(fflya_tmp)

    plot_spa_dist(fflya)
    save_flatfield_polyfit(fflya)
    #save_flatfield_rolling_avg(fflya)
