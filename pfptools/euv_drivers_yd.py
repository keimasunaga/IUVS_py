import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from datetime import datetime, timezone

from common.tools import nnDt, unix2Dt
from orbit_info import get_Dt_apo

def get_euv_drivers_sav():
    '''
    Reads solar wind drivers provided by Jasper Halekas.
    To update the data, you need to ask JH for the updated .tplot file and then run an idl procedure
    /Users/masunaga/work/idl_git/masu-lib_maven/save_data/save_sw_drivers.pro
    returns: sw_drivers (dict)
    '''
    fname = '/Users/masunaga/work/save_data/maven/sav/euv/euv_drivers_YD/euv.sav'
    sav = readsav(fname)
    dic = sav['euvdata']
    utime = dic.item()[0]
    timeDt = np.array(unix2Dt(utime))

    euv = dic.item()[1][0]
    euv0 = dic.item()[1][0]
    euv1 = dic.item()[1][1]
    euv2 = dic.item()[1][2]
    euv = np.array([euv0, euv1, euv2])
    euv_drivers = {'utime':utime, 'timeDt':timeDt, 'euv':euv}
    return euv_drivers

def get_euv_driver_apo(orbit_number):
    '''
    Get solar wind drivers for given orbit number.
    arg: orbit_number (int)
    returns: sw_driver (dict)
    '''
    Dt_apo = get_Dt_apo(orbit_number)
    Dt_apo_prev = get_Dt_apo(orbit_number-1)
    Dt_apo_aftr = get_Dt_apo(orbit_number+1)
    dic_euv = get_euv_drivers_sav()
    timeDt_euv = dic_euv['timeDt']
    if Dt_apo is None:
        print('---- Dt_apo is None, returning None ----')
        return None
    idxDt = nnDt(timeDt_euv, Dt_apo)
    euv_driver = {ikey:dic_euv[ikey].T[idxDt] for ikey in dic_euv.keys()}
    if timeDt_euv[idxDt] < Dt_apo_prev or timeDt_euv[idxDt] > Dt_apo_aftr:
        print('---- No SW drivers found within the selected orbit ----')
        return None
    else:
        return euv_driver
