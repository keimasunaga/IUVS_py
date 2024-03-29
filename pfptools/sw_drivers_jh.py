import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav
from datetime import datetime, timezone

from common.tools import nnDt, unix2Dt
from orbit_info import get_Dt_apo

def get_sw_drivers_sav():
    '''
    Reads solar wind drivers provided by Jasper Halekas.
    To update the data, you need to ask JH for the updated .tplot file and then run an idl procedure
    /Users/masunaga/work/idl_git/masu-lib_maven/save_data/save_sw_drivers.pro
    returns: sw_drivers (dict)
    '''
    fname = '/Users/masunaga/work/save_data/maven/sav/sw_drivers_JH/drivers_merge_l2.sav'
    sav = readsav(fname)
    dic = sav['dic']
    utime = dic.item()[0][0][0]
    timeDt = np.array(unix2Dt(utime))
    bsw = dic.item()[0][0][2]
    npsw = dic.item()[1][0][2]
    nasw = dic.item()[2][0][2]
    vpsw = dic.item()[3][0][2]
    tp = dic.item()[4][0][2]
    vvec = dic.item()[5][0][2]
    sw_drivers = {'utime':utime, 'timeDt':timeDt, 'bsw':bsw, 'npsw':npsw, 'nasw':nasw, 'vpsw':vpsw, 'tp':tp, 'vvec':vvec}
    return sw_drivers

def get_sw_driver_apo(orbit_number):
    '''
    Get solar wind drivers for given orbit number.
    arg: orbit_number (int)
    returns: sw_driver (dict)
    '''
    Dt_apo = get_Dt_apo(orbit_number)
    Dt_apo_prev = get_Dt_apo(orbit_number-1)
    Dt_apo_aftr = get_Dt_apo(orbit_number+1)
    dic_sw = get_sw_drivers_sav()
    timeDt_sw = dic_sw['timeDt']
    if Dt_apo is None:
        print('---- Dt_apo is None, returning None ----')
        return None
    idxDt = nnDt(timeDt_sw, Dt_apo)
    sw_driver = {ikey:dic_sw[ikey].T[idxDt] for ikey in dic_sw.keys()}
    if timeDt_sw[idxDt] < Dt_apo_prev or timeDt_sw[idxDt] > Dt_apo_aftr:
        print('---- No SW drivers found within the selected orbit ----')
        return None
    else:
        return sw_driver

def test_plot():
    dic = get_sw_drivers_sav()
    utime = dic['utime']
    timeDt = dic['timeDt']
    bsw = dic['bsw']
    npsw = dic['npsw']
    nasw = dic['nasw']
    vpsw = dic['vpsw']
    tp = dic['tp']
    vvec = dic['vvec']

    plt.close()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(611)
    ax1.plot(timeDt, bsw.T[:,0], color='b', zorder=2)
    ax1.plot(timeDt, bsw.T[:,1], color='g', zorder=1)
    ax1.plot(timeDt, bsw.T[:,2], color='r', zorder=0)
    ax1.plot(timeDt, bsw.T[:,3], color='k', zorder=3)
    ax1.set_ylabel('bsw')

    ax2 = fig.add_subplot(612)
    ax2.plot(timeDt, npsw, color='k')
    ax2.set_ylabel('npsw')

    ax3 = fig.add_subplot(613)
    ax3.plot(timeDt, nasw, color='k')
    ax3.set_ylabel('nasw')

    ax4 = fig.add_subplot(614)
    ax4.plot(timeDt, vpsw, color='k')
    ax4.set_ylabel('vpsw')

    ax5 = fig.add_subplot(615)
    ax5.plot(timeDt, vvec.T[:,0], color='b', zorder=2)
    ax5.plot(timeDt, vvec.T[:,1], color='g', zorder=1)
    ax5.plot(timeDt, vvec.T[:,2], color='r', zorder=0)
    ax5.set_ylabel('vvec')

    ax6 = fig.add_subplot(616)
    ax6.plot(timeDt, tp, color='k')
    ax6.set_ylabel('tp')


    plt.show()
