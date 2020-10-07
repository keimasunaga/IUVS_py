import numpy as np
from datetime import datetime, timezone

'''
Time handling tools
'''
def get_et(hdul):
    et = hdul['Integration'].data['et']
    return et

def get_utc(hdul):
    utc = hdul['Integration'].data['utc']
    return utc

def utc2Dt(utc):
    Dt = datetime.strptime(utc, '%Y/%j %b %d %H:%M:%S.%fUTC').replace(tzinfo=timezone.utc)
    return Dt

def get_timeDt(hdul):
    utc = hdul['Integration'].data['utc']
    timeDt = np.array([utc2Dt(iutc) for iutc in utc])
    return timeDt
