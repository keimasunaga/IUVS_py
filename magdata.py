'''
Functions for reading MAVEN level 2 IDL SAV files for MAG data, and rotating into MSO.

Works in Python 3.5

requires the following pip packages:
numpy
scipy
matplotlib
spiceypy
'''

import os
import datetime as dt
import calendar

import numpy as np
from scipy.io import readsav
import spiceypy as spice
from matplotlib import pyplot as plt


def numpy_UNX_to_UTC(POSIX_time_array):
    '''Convert a numpy array of POSIX times into a numpy array of datetime objects.'''

    return np.array([dt.datetime.utcfromtimestamp(i) for i in POSIX_time_array])


def numpy_UTC_to_UNX(UTC_time_array):
    '''Convert a numpy array of UTC times into a numpy array of POSIX times.'''

    return np.array([calendar.timegm(i.timetuple()) for i in UTC_time_array])


def instrument_data_file(root, date, level='2', mag_product=None):
    '''Return the path to a MAVEN instrument datafile.

    root: string, path where MAVEN data is saved.
    date: datetime object, time of requested dataset
    level: string, data product level, e.g. '2' (currently l1 and l0 unavailable)
    mag_product: string, optional, name of requested MAG data'''

    # Get the folder containing the requested data.
    mag_data_folder = os.path.join(
        root, 'maven', 'data', 'sci', 'mag', 'l' + level, 'sav', mag_product,
        date.strftime('%Y'), date.strftime('%m'))
    level_name = ''.join(('l', level))

    data_file_id = '_'.join(
        ('mvn', 'mag', level_name, 'pl', mag_product, date.strftime('%Y%m%d')))

    # Retrieve all files matching the name.
    matching_files = [file for file in os.listdir(mag_data_folder) if file.startswith(data_file_id)]

    # If there are none, there was either no file downloaded or the MAVEN data directory is
    # incorrect.
    if len(matching_files) == 0:
        raise IOError('No existing file for that time/dataset/instrument, exiting...')

    # If there are multiple, select the last one as the latest version pushed.
    if len(matching_files) != 1:
        matching_files = [matching_files[-1]]

    return os.path.join(mag_data_folder, matching_files[0])


def read_mag_data(data_directory, start_date, mag_data_product, level):

    dataset_filename = instrument_data_file(
        data_directory, start_date, level=level, mag_product=mag_data_product)
    mag_dataset = readsav(dataset_filename)
    mag_posix_time = mag_dataset['data']['time']
    mag_time_utc = numpy_UNX_to_UTC(mag_posix_time)
    print(mag_time_utc)
    if level == '1':
        mag_b_vec = mag_dataset['data']['vec']
    elif level == '2':
        mag_b_x = mag_dataset['data']['ob_bpl_x']
        mag_b_y = mag_dataset['data']['ob_bpl_y']
        mag_b_z = mag_dataset['data']['ob_bpl_z']
        mag_b_vec = np.column_stack((mag_b_x, mag_b_y, mag_b_z))

    return mag_time_utc, mag_b_vec


def load_mag_data(data_directory, target_date, n_days, mag_data_product, level):

    time, b_vec = read_mag_data(data_directory, target_date, mag_data_product, level)
    for i in range(1, n_days):
        read_date = target_date + dt.timedelta(days=i)
        day_time, day_b_vec = read_mag_data(data_directory, read_date, mag_data_product, level)
        time = np.append(time, day_time)
        if level == '1':
            b_vec = np.append(b_vec, day_b_vec)
        elif level == '2':
            b_vec = np.append(b_vec, day_b_vec, axis=0)

    return time, b_vec


def quaternion_rotation(q, v):

    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    t2 = a*b
    t3 = a*c
    t4 = a*d
    t5 = -b*b
    t6 = b*c
    t7 = b*d
    t8 = -c*c
    t9 = c*d
    t10 = -d*d

    v1new = 2*((t8 + t10)*v1 + (t6 - t4)*v2 + (t3 + t7)*v3) + v1
    v2new = 2*((t4 + t6)*v1 + (t5 + t10)*v2 + (t9 - t2)*v3) + v2
    v3new = 2*((t7 - t3)*v1 + (t2 + t9)*v2 + (t5 + t8)*v3) + v3

    return v1new, v2new, v3new


def load_kernels(sc_data_folder, mvn_software_folder):

    kernel_folder = os.path.join(sc_data_folder, 'misc', 'spice', 'naif')

    generic_kernel_folder = os.path.join(kernel_folder, 'generic_kernels')
    general_kernels =\
        [os.path.join(generic_kernel_folder, 'pck', 'pck00010.tpc'),
         os.path.join(generic_kernel_folder, 'spk', 'satellites', 'mar097.bsp'),
         os.path.join(generic_kernel_folder, 'spk', 'planets', 'de430.bsp'),
         os.path.join(generic_kernel_folder, 'lsk', 'naif0012.tls'),
         os.path.join(generic_kernel_folder, 'lsk', 'naif0011.tls')]

    maven_kernel_folder = os.path.join(kernel_folder, 'MAVEN', 'kernels')

    maven_ck = os.path.join(maven_kernel_folder, 'ck')
    maven_sclk = os.path.join(maven_kernel_folder, 'sclk')
    maven_spk = os.path.join(maven_kernel_folder, 'spk')
    maven_fk = os.path.join(
        mvn_software_folder, 'projects', 'maven', 'general', 'spice', 'kernels', 'fk')

    for g in general_kernels:
        spice.furnsh(g)
    sclk_files = [i for i in os.listdir(maven_sclk) if i.endswith('tsc')]
    for sclk in sclk_files:
        spice.furnsh(os.path.join(maven_sclk, sclk))
    spk_files = [i for i in os.listdir(maven_spk) if i.endswith('bsp')]
    for spk in spk_files:
        spice.furnsh(os.path.join(maven_spk, spk))
    ck_files = [i for i in os.listdir(maven_ck) if i.endswith('bc')]
    for ck in ck_files:
        spice.furnsh(os.path.join(maven_ck, ck))

    fk_files = [i for i in os.listdir(maven_fk) if i.endswith('tf')]
    for fk in fk_files:
        spice.furnsh(os.path.join(maven_fk, fk))




import numpy as np
from scipy.io import readsav
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import spiceypy as spice

from variables import pfpdataloc
import common.tools as ctools
from common import circular

class MagFile:
    def __init__(self, year, month, day, level='l2', dtype='1sec'):
        self.year = year
        self.month = month
        self.day = day
        self.level = level
        self.dtype = dtype

    def get_file(self):
        filepath = os.path.join(pfpdataloc, 'mag', self.level, 'sav', self.dtype, str(self.year), '{:02d}'.format(self.month))
        filename = 'mvn_mag_' + self.level + '_pl_' + self.dtype + '_' + str(self.year) + '{:02d}'.format(self.month) + '{:02d}'.format(self.day) + '.sav'
        fnames = glob.glob(os.path.join(filepath, filename))
        try:
            return fnames[0]
        except:
            print('----- '+ filename + ' not found. -----')

def get_sav(date=None, Dt=None, dtype='1sec'):

    if date is not None:
        year = int(date[0:4])
        month = int(date[4:6])
        day = int(date[6:8])

    elif Dt is not None:
        year = Dt.year
        month = Dt.month
        day = Dt.day

    magfile = MagFile(year, month, day, dtype=dtype)
    fname = magfile.get_file()
    sav = readsav(fname)
    return sav


class MagField:
    def __init__(self, sav, frame='MAVEN_MSO', load_spice=False):
        if load_spice:
            sc_data_folder = '/Volumes/Gravity/work/data' #'/Users/rejo9726/data'
            mvn_software_folder = '/Users/masunaga/work/idl/maven_sw'
            load_kernels(sc_data_folder, mvn_software_folder)
        data = sav['data']
        self.ut = data['time']
        self.timeDt = np.array([datetime.utcfromtimestamp(iut) for iut in self.ut])
        self.timeDt2 = np.array([datetime.fromtimestamp(iut) for iut in self.ut])
        self.et = spice.str2et([i.strftime('%b %d, %Y %H:%M:%S') for i in self.timeDt])
        self.x = data['ob_bpl_x']
        self.y = data['ob_bpl_y']
        self.z = data['ob_bpl_z']
        self.from_frame = 'MAVEN_SPACECRAFT'
        self.to_frame = frame
        if self.to_frame == 'MAVEN_MSO':
            self._to_mso_all()
        self.t = np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def _to_mso_all(self):
        b_mso = [[], [], []]
        for i, iet in enumerate(self.et):
            bx, by, bz = self._to_mso(iet, np.array([self.x[i], self.y[i], self.z[i]]))
            b_mso[0].append(bx)
            b_mso[1].append(by)
            b_mso[2].append(bz)
        self.x = np.array(b_mso[0])
        self.y = np.array(b_mso[1])
        self.z = np.array(b_mso[2])
        self.cone = np.rad2deg(np.arccos(self.x/np.sqrt(self.x**2 + self.y**2 + self.z**2)))
        clock_tmp = np.rad2deg(np.arccos(self.y/np.sqrt(self.y**2 + self.z**2)))
        self.clock = np.where(self.z>=0, clock_tmp, -clock_tmp)

    def _to_mso(self, et, b):
        mtrx = spice.pxform(self.from_frame, self.to_frame, et)
        q = spice.m2q(mtrx)
        x, y, z = quaternion_rotation(q, b)
        return x, y, z

    def plot_x(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.x, **kwargs)
        else:
            ax.plot(self.timeDt, self.x, **kwargs)

    def plot_y(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.y, **kwargs)
        else:
            ax.plot(self.timeDt, self.y, **kwargs)

    def plot_z(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.z, **kwargs)
        else:
            ax.plot(self.timeDt, self.z, **kwargs)

    def plot_t(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.t, **kwargs)
        else:
            ax.plot(self.timeDt, self.t, **kwargs)

    def plot_cone(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.cone, **kwargs)
        else:
            ax.plot(self.timeDt, self.cone, **kwargs)

    def plot_clock(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.clock, **kwargs)
        else:
            ax.plot(self.timeDt, self.clock, **kwargs)

    def append(self, other):
        self.ut = np.append(self.ut, other.ut)
        self.et = np.append(self.et, other.et)
        self.timeDt = np.append(self.timeDt, other.timeDt)
        self.x = np.append(self.x, other.x)
        self.y = np.append(self.y, other.y)
        self.z = np.append(self.z, other.z)
        self.t = np.append(self.t, other.t)

    def get_mean(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        mean_x = np.mean(self.x[idx[0]:idx[1]])
        mean_y = np.mean(self.y[idx[0]:idx[1]])
        mean_z = np.mean(self.z[idx[0]:idx[1]])
        return mean_x, mean_y, mean_z

    def get_std(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        std_x = np.std(self.x[idx[0]:idx[1]])
        std_y = np.std(self.y[idx[0]:idx[1]])
        std_z = np.std(self.z[idx[0]:idx[1]])
        return std_x, std_y, std_z

    def get_cone_mean(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        cone_mean = np.mean(self.cone[idx[0]:idx[1]])
        return cone_mean

    def get_cone_std(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        cone_std = np.std(self.cone[idx[0]:idx[1]])
        return cone_std

    def get_clock_mean(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        clock_mean = circular.mean(self.clock[idx[0]:idx[1]])
        if clock_mean > 180:
            clock_mean = clock_mean - 360
        return clock_mean

    def get_clock_std(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        clock_std = circular.std(self.clock[idx[0]:idx[1]])
        return clock_std

def get_magfield(sDt, n_days, frame='MAVEN_MSO', load_spice=False):
    for i in range(n_days):
        Dt = sDt + timedelta(days=i)
        sav = get_sav(Dt=Dt)
        if i == 0:
            mag = MagField(sav, load_spice=load_spice)
        else:
            other = MagField(sav)
            mag.append(other)
    return mag


def test():
    #sav = read_sav(date='20150126')
    #mag = MagField(sav)
    sc_data_folder = '/Volumes/Fenix/work/data' #'/Users/rejo9726/data'
    mvn_software_folder = '/Users/masunaga/work/idl/maven_sw'
    print('load SPICE kernels...')
    load_kernels(sc_data_folder, mvn_software_folder)
    print('Done')
    #load_kernels(sc_data_folder, mvn_software_folder)
    #load_kernels()
    sDt = datetime(2015, 1, 25, 6, 0)
    #sav = get_sav(Dt=sDt)
    #mag = MagField(sav)
    mag = get_magfield(sDt, 2)
    print(mag.ut)
    print(mag.timeDt)
    print(mag.timeDt2)
    mag.plot_x(color='b', label='Bx', zorder=3)
    mag.plot_y(color='g', label='By', zorder=2)
    mag.plot_z(color='r', label='Bz', zorder=1)
    #sDt = datetime(2015, 1, 26, 6, 0)
    #eDt = datetime(2015, 1, 26, 7, 30)
    #plt.xlim(sDt, eDt)
    plt.legend()
    plt.show()

#spice.kclear()
#test()
