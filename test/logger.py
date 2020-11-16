import numpy as np
from datetime import datetime
import os
from IPython import get_ipython
ipy = get_ipython()

def start_log():
    now = datetime.now()
    today = '{:04d}{:02d}{:02d}'.format(now.year, now.month, now.day)
    current_dir = ipy.magic('pwd')
    log_loc = current_dir + '/log/'
    log_name = 'log_test' #+ today
    fname = log_loc + log_name

    if not os.path.isfile(fname): ## when the log file does not exist create a new log file
        os.makedirs(log_loc, exist_ok=True)
        ipy.magic('logstart -r -o -t' + ' ' + fname)
    else:
        try:
            ipy.magic('logstart -r -o -t' + ' ' + fname + ' append')
        except:
            ipy.magic('logon -r -o -t' + ' ' + fname + ' append')

def stop_log():
    ipy.magic('logoff')


"""import numpy as np
from datetime import datetime
from glob import glob
import os
from IPython import get_ipython
ipy = get_ipython()

def start_log():
    current_dir = ipy.magic('pwd')
    now = datetime.now()
    log_name = 'log_test'
    fname = log_loc + log_name
    f = glob(fname)

    if np.size(f) == 0: #when file does not exist create a new log file
        os.makedirs(log_loc, exist_ok=True)
        ipy.magic('logstart -r -o ' + fname)
    else:
        ipy.magic('logon -r -o ' + fname)

def end_log():
    ipy.magic('logoff')
"""
