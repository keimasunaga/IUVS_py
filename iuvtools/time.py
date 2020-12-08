import numpy as np
from datetime import datetime

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
    Dt = datetime.strptime(utc, '%Y/%j %b %d %H:%M:%S.%fUTC')
    return Dt

def get_timeDt(hdul):
    utc = hdul['Integration'].data['utc']
    timeDt = np.array([utc2Dt(iutc) for iutc in utc])
    return timeDt

def Dt2str(timeDt):
    '''
    Converts datetime to string format
    argment:
        timeDt: a datetime or a list of those
    returns:
        str_time: a string time in string format or list or array of them
    '''
    if np.size(timeDt) == 1:
        if type(timeDt) is datetime:
            return timeDt.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            return timeDt[0].strftime('%Y-%m-%dT%H:%M:%S')
    else:
        return [iDt.strftime('%Y-%m-%dT%H:%M:%S') for iDt in timeDt]
