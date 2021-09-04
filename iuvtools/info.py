# obsinfo
def get_orbit_number(hdul):
    orbit_number = hdul['Observation'].data['Orbit_Number']
    return orbit_number[0]

def get_segment(hdul):
    segment = hdul['Observation'].data['Orbit_Segment']
    return segment[0]

def get_channel(hdul):
    channel = hdul['Observation'].data['Channel']
    return channel

def get_solar_lon(hdul):
    Ls = hdul['Observation'].data['Solar_Longitude']
    return Ls[0]

def get_mcp_volt(hdul):
    mcp_volt = hdul['Observation'].data['MCP_Volt']
    return mcp_volt[0]

def dayside(hdul): ## Check if the voltage value is correct later
    mcp_volt = get_mcp_volt(hdul)
    if mcp_volt < 700:
        return True
    else:
        return False





import numpy as np
import os, glob
from datetime import timezone

from PyUVS.time import find_segment_et, et2datetime
from iuvtools.geometry import get_sc_sza
from iuvtools.time import get_timeDt, get_et
from iuvdata_l1b import get_apoapseinfo
from common.tools import nnDt, Dt2unix, unix2Dt
from orbit_info import get_Dt_apo, get_df_orbfiles
from variables import saveloc

def get_sza_apo(orbit_number, get_lim=False):
    Dt_apo = get_Dt_apo(orbit_number)
    apoinfo = get_apoapseinfo(orbit_number)

    Dt_arr = np.array([])
    et_arr = np.array([])
    sza_arr = np.array([])
    if apoinfo.n_files>0:
        print('Processing orbit #' + str(orbit_number))
        for ith_file, iswath_number in enumerate(apoinfo.swath_number):
            hdul = apoinfo.get_hdul(ith_file)
            et = get_et(hdul)
            timeDt = get_timeDt(hdul)
            sza = get_sc_sza(hdul)
            Dt_arr = np.append(Dt_arr, timeDt)
            et_arr = np.append(et_arr, et)
            sza_arr = np.append(sza_arr, sza)
        idx = nnDt(Dt_arr, Dt_apo)
        sza_apo = sza_arr[idx]
        sza_lim = [np.min(sza_arr), np.max(sza_arr)]
        hdul.close()
        if get_lim:
            return sza_apo, sza_lim
        else:
            return sza_apo
    else:
        #print('No files found at orbit #' + str(orbit_number))
        if get_lim:
            return None, None
        else:
            return None

def save_sza_apo_data(sorbit, eorbit):
    from common.tools import RunTime
    RT = RunTime()
    RT.start()
    sblock = int(sorbit/1000)*1000
    eblock = int(eorbit/1000)*1000
    sblocks = range(sblock, eblock, 1000)

    for iblk in sblocks:
        isorbit = iblk
        ieorbit = iblk + 1000
        orbit_block = str(iblk).zfill(5)
        orbit_list = []
        sza_apo_list = []
        sza_lim_list = []
        for iorbit in range(isorbit, ieorbit):
            sza_apo, sza_lim = get_sza_apo(iorbit, get_lim=True)
            if sza_apo is not None:
                orbit_list.append(iorbit)
                sza_apo_list.append(sza_apo)
                sza_lim_list.append(sza_lim)

        dic = {'orbit_number':np.array(orbit_list), 'sza_apo':np.array(sza_apo_list), 'sza_lim':np.array(sza_lim_list)}
        savepath = os.path.join(saveloc, 'misc_items/sza_apo_list/')
        os.makedirs(savepath, exist_ok=True)
        savename = 'sza_apo_orbit_'+orbit_block+'.npy'
        np.save(savepath+savename, dic)
    RT.stop()

def get_sza_apo_data(all=True, orbit_number=None):
    savepath = os.path.join(saveloc, 'misc_items/sza_apo_list/')
    if all:
        fnames = sorted(glob.glob(savepath+'sza_apo_orbit_*'))
        dic_all = {'orbit_number':np.array([]), 'sza_apo':np.array([]), 'sza_lim':np.array([])}
        for fname in fnames:
            dic = np.load(fname, allow_pickle=True).item()
            for ikey in dic.keys():
                dic_all[ikey] = np.append(dic_all[ikey], np.load(fname, allow_pickle=True).item()[ikey])
                #[[dic_all[ikey].append(np.load(fname, allow_pickle=True).item()[ikey]) for ikey in np.load(fname, allow_pickle=True).item().keys()] for fname in fnames]
        return dic_all
    else:
        orbit_block = str(int(orbit_number/1000)*1000).zfill(5)
        fname = glob.glob(savepath+'sza_apo_orbit_'+orbit_block+'.npy')
        dic = np.load(fname[0], allow_pickle=True).item()
        return dic

def save_orbit_sza_apo_90():
    dic = get_sza_apo_data()
    orbit_number = dic['orbit_number']
    sza_apo = dic['sza_apo']
    orbit_sza90 = orbit_number[np.where((85<sza_apo) & (sza_apo<95))]
    savepath = saveloc + '/misc_items/'
    np.save(savepath + 'orbit_sza90', orbit_sza90)

def get_orbit_sza_apo_90():
    savepath = saveloc + '/misc_items/'
    orbit_sza_apo_90 = np.load(savepath + 'orbit_sza90.npy', allow_pickle=True)
    return orbit_sza_apo_90
