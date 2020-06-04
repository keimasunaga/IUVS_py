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
from datetime import datetime
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


if __name__ == '__main__':

    # Adjust these for your set-up. The first should be the folder containing
    # all MAVEN/SPICE data, the second should be the MAVEN tplot package software folder.
    # (That contains the FK spice kernels.)
    sc_data_folder = '/Volumes/Gravity/work/data' #'/Users/rejo9726/data'
    mvn_software_folder = '/Users/masunaga/work/idl/maven_sw'#'/Users/rejo9726/Software/maven_sw'
    s = '2015-01-26'
    # s = '2017-09-12'
    # s = '2016-05-16'
    # s = '2017-07-27'
    # s = '2015-03-24'
    start_date = dt.datetime.strptime(s, '%Y-%m-%d')
    n_days = 1
    mag_level = '2'
    mag_time_cadence = '1sec'

    print('load MAG data...')
    mag_time_utc, b_pl = load_mag_data(
         sc_data_folder, start_date, n_days, mag_time_cadence, mag_level)
    print('Done')
    print(mag_time_utc)
    print('load SPICE kernels...')
    load_kernels(sc_data_folder, mvn_software_folder)
    print('Done')

    # Rotate MAG into MSO coordinates...
    # (this isn't optimized in the slightest, just FYI)
    bx = []
    by = []
    bz = []
    mag_ephemeris_time = spice.str2et([i.strftime('%b %d, %Y %H:%M:%S') for i in mag_time_utc])
    for et_b_index, et in enumerate(mag_ephemeris_time):
        b = b_pl[et_b_index, :]
        m_sc_to_MSO = spice.pxform(
            'MAVEN_SPACECRAFT', 'MAVEN_MSO', mag_ephemeris_time[et_b_index])
        q_sc_to_MSO = spice.m2q(m_sc_to_MSO)
        bx_mso, by_mso, bz_mso = quaternion_rotation(q_sc_to_MSO, b)
        bx.append(bx_mso)
        by.append(by_mso)
        bz.append(bz_mso)

    # Plot B in payload and MSO coordinates
    btime = mag_time_utc
    f0, ax = plt.subplots(2, sharex=True, figsize=(11, 6))
    ax[0].plot(btime, b_pl[:, 0], color='b')
    ax[0].plot(btime, b_pl[:, 1], color='g')
    ax[0].plot(btime, b_pl[:, 2], color='r')
    ax[0].set_ylabel('B, nT\nPayload')
    ax[1].plot(btime, bx, color='b')
    ax[1].plot(btime, by, color='g')
    ax[1].plot(btime, bz, color='r')
    ax[1].set_ylabel('B, nT\nMSO')

    plt.subplots_adjust(right=0.9)
    plt.text(0.905, 0.8, 'Bx', fontsize=14, transform=plt.gcf().transFigure, color='b')
    plt.text(0.905, 0.7, 'By', fontsize=14, transform=plt.gcf().transFigure, color='g')
    plt.text(0.905, 0.6, 'Bz', fontsize=14, transform=plt.gcf().transFigure, color='r')

    plt.text(0.905, 0.4, 'Bx', fontsize=14, transform=plt.gcf().transFigure, color='b')
    plt.text(0.905, 0.3, 'By', fontsize=14, transform=plt.gcf().transFigure, color='g')
    plt.text(0.905, 0.2, 'Bz', fontsize=14, transform=plt.gcf().transFigure, color='r')

    sDt = datetime(2015, 1, 26, 6, 0)
    eDt = datetime(2015, 1, 26, 7, 30)
    ax[0].set_xlim(sDt, eDt)
    ax[1].set_xlim(sDt, eDt)
    plt.show()
