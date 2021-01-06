import numpy as np
import sys, os

from variables import saveloc

def save_efield_angle_origin(orbit_number):
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

        idx_xorigin = (np.where(np.abs(x) == np.abs(x).min()))[0]
        idx_yorigin = (np.where(np.abs(y) == np.abs(y).min()))[0]
        efield_angle_origin = efield_angle[idx_yorigin, idx_xorigin]

        savefile =  saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe_data/all/anc/efield_angle_origin.npy'#orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/'
        if os.path.isfile(savefile):
            svdic = np.load(savefile, allow_pickle=True).item()
            print(type(svdic))
            svdic['orbit_' + '{:05d}'.format(orbit_number)] = efield_angle_origin[0]
            np.save(savefile, svdic)
        else:
            svdic = {}
            svdic['orbit_' + '{:05d}'.format(orbit_number)] = efield_angle_origin[0]
            np.save(savefile, svdic)

if __name__ == '__main__':
    sorbit = int(sys.argv[1])
    norbit = int(sys.argv[2])
    eorbit = sorbit + norbit
    orbit_arr = range(sorbit, eorbit)#[849]
    for iorbit_number in orbit_arr:
        print('{:05d}'.format(iorbit_number))
        save_efield_angle_origin(iorbit_number)
