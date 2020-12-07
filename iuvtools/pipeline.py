import fnmatch as fnm
import glob
import os
import time

import numpy as np
import paramiko
from astropy.io import fits
from scipy.io import readsav
import sysrsync as rsync

from PyUVS.miscellaneous import clear_line
from PyUVS.variables import vm_username, vm_password, data_directory, pyuvs_directory, spice_directory, slit_width_mm, \
    pixel_size_mm, focal_length_mm, muv_dispersion, slit_pix_min, slit_pix_max

from iuvtools.time import get_timeDt


class IuvPipeLine:
    def __init__(self):
        # define VM-related variables
        self.vm = 'maven-iuvs-itf'
        self.login = vm_username + '@' + self.vm + ':'
        self.production_root_path = '/maven_iuvs/production/products/'
        self.stage_root_path = '/maven_iuvs/stage/products/'
        #self.vm_production_path = login + production_path
        #self.vm_stage_path = login + stage_path
        self.vm_spice_path = self.login + '/maven_iuvs/stage/anc/spice/'

    def download_production(self, level='l1b', orbit_blocks=None, cruise=False):
        """
        Synchronize new data from the VM and remove any old files that have been replaced by newer versions.

        Parameters
        ----------
        level : string
            Data level. Defaults to l1b.
        orbit_blocks : list
            Specify orbit blocks to download with list (i.e., [12000, 12100]). Defaults to None.
        cruise: bool
            If True cruse data is included to downloaded
        Returns
        -------
        None.
        """

        # get starting time
        t0 = time.time()

        self.production_data_path = self.production_root_path + 'leve'+level+'/'
        self.vm_production_data_path = self.login + self.production_data_path

        # try to sync the files, if it fails, user probably isn't on the VPN
        try:
            # sync level 1B data
            if level=='l1b':
                # get the orbit blocks in the production and stage environments
                clear_line()

                print('Fetching level 1B production and stage folders from the VM...', end='\r')
                production_folders = get_vm_folders(self.vm, vm_username, vm_password, self.production_data_path)

                if orbit_blocks is not None:
                    production_folders = [ifolder for ifolder in production_folders[1:] if int(ifolder[5:]) in orbit_blocks]
                    if cruise:
                        production_folders.append('cruise')

                # sync and clean the level 1B production and stage environments
                clear_line()
                print('Syncing level 1B data...')
                sync_production(self.vm_production_data_path, production_folders,
                                os.path.join(data_directory, 'level1b'))

        except:
            raise Exception('Error encountered. Are you on the VPN?')

        # get ending time
        t1 = time.time()
        seconds = t1 - t0
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        # tell us how long it took
        clear_line()
        print('Data syncing and cleanup took %.2d:%.2d:%.2d.' % (h, m, s))


    def download_stage(self, level='l1b', orbit_blocks=None, cruise=False):
        """
        Synchronize new data from the VM and remove any old files that have been replaced by newer versions.

        Parameters
        ----------
        spice : bool
            Whether or not to sync SPICE kernels. Defaults to True.
        l1b : bool
            Whether or not to sync level 1B data. Defaults to True.

        Returns
        -------
        None.
        """

        # get starting time
        t0 = time.time()
        stage_data_path = self.stage_root_path + 'leve'+level+'/'
        vm_stage_data_path = self.login + stage_data_path

        # try to sync the files, if it fails, user probably isn't on the VPN
        try:
            # sync level 1B data
            if level=='l1b':
                # get the orbit blocks in the stage environment
                clear_line()

                #print('Fetching level 1B stage folders from the VM...', end='\r')
                print(stage_data_path)
                stage_folders = get_vm_folders(self.vm, vm_username, vm_password, stage_data_path)

                if orbit_blocks is not None:
                    stage_folders = [ifolder for ifolder in stage_folders if int(ifolder[5:]) in orbit_blocks]
                    if cruise:
                        stage_folders.append('cruise')

                # sync and clean the stage environment
                clear_line()
                print('Syncing level 1B data...')
                sync_production(vm_stage_data_path, stage_folders,
                                os.path.join(data_directory, 'stage/level1b'))

        except:
            raise Exception('Error encountered. Are you on the VPN?')

        # get ending time
        t1 = time.time()
        seconds = t1 - t0
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        # tell us how long it took
        clear_line()
        print('Data syncing and cleanup took %.2d:%.2d:%.2d.' % (h, m, s))

    def update_spice(self):
        update_spice_kernels(self.vm_spice_path, spice_directory)



def ignore_hidden_folder(folder_list):
    """
    Takes a list of folder names and ignores hidden folders.

    Parameters
    ----------
    folder_list : array-like
        A list of orbit block folders, e.g., orbit03400, orbit03500, etc.

    Returns
    -------
    non_hidden_folders : list
        A list of non-hidden folders.
    """

    # get the non-hidden folders
    non_hidden_folders = [f for f in folder_list if '.' not in f]

    # return the list of non-hidden folders
    return non_hidden_folders


def get_vm_folders(vm_name, username, password, folder_path):
    """
    Updates the spice kernels by syncing the currently-used spice kernels with the VM.

    Parameters
    ----------
    vm_name : str
        The VM name, e.g., 'maven-iuvs-itf'.
    username : str
        Your username for accessing the VM, i.e. 'username@vm_name'.
    password : str
        Your password for accessing the VM.
    folder_path : str
        The path to the location where the user wants the folders.

    Returns
    -------
    non_hidden_folders : list
        A sorted list of folder strings.
    """

    # access the VM and get a list of the available orbit blocks (folders)
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(vm_name, username=username, password=password)
    ftp = ssh.open_sftp()
    ftp.chdir(folder_path)

    sorted_folders = sorted(ftp.listdir())

    # remove hidden folders from the list
    non_hidden_folders = ignore_hidden_folder(sorted_folders)

    # return the list of folders
    return non_hidden_folders


def update_spice_kernels(remote_spice_path, local_spice_path):
    """
    Updates the spice kernels by syncing the currently-used spice kernels with the VM.

    Parameters
    ----------
    remote_spice_path : str
        Path to the SPICE kernels on the VM.

    local_spice_path : str
        Path to the location of SPICE kernels on your system.

    Returns
    -------
    None.
    """

    # synchronize SPICE kernels
    clear_line()
    print('Updating SPICE kernels...')
    rsync.run(source=remote_spice_path, destination=local_spice_path, options=['-trzL', '--out-format=%n'])


def make_block_directory(location):
    """
    Make a directory for a given orbit block.

    Parameters
    ----------
    location : str
        A string of the location to make, or see if it exists.

    Returns
    -------
    None.
    """

    # if there isn't a folder...
    if not os.path.exists(location):

        # put in the damn try statement to avoid race conditions and locking
        try:
            os.makedirs(location)
        except OSError:
            raise Exception('There was an OSError when trying to make ' + location)

def make_blank_file(file_name, file_path):
    """
    Make a blank file if it doesn't already exist.

    Parameters
    ----------
    file_name : str
        The file name.
    file_path : str
        Where to save the file.

    Returns
    -------
    None.
    """

    # if there isn't a file...
    if not os.path.exists(file_path + file_name):

        # put in the damn try statement to avoid race conditions and locking
        try:
            open(file_path + file_name, 'a').close()
        except OSError:
            raise Exception('There was an OSError when trying to make ' + location)


def clean_directory(data_path, folder_block):
    """
    Cleans the block of directories and records how much time it took to clean the directories.

    Parameters
    ----------
    data_path : str
        The path to the data block folders.
    folder_block : str
        The folder block, e.g., 'orbit09500'.

    Returns
    -------
    None.
    """

    data_location = os.path.join(data_path, folder_block + '/')
    delete_data(data_location)


def delete_data(location):
    """
    Delete all the old data.

    Parameters
    ----------
    location : str
        The absolute path to a folder (trailing slash not needed).

    Returns
    -------
    None.
    """

    # delete xml files
    delete_xml(location)

    # remove any non-MUV files
    #delete_nonmuv(location)

    # now get the list of actual data files
    crappy_files = []
    all_files = find_all('*', location)

    # the data are sorted in an ass-backward lettering convention so fix that shit and sort it
    all_files = sorted([f.replace('s0', 'a0') for f in all_files])

    # delete the old files
    last_time_stamp = ''
    last_channel = ''
    last_file = ''  # initialize variable as empty string
    for file in all_files:
        current_time_stamp = file[-31:-16]  # e.g., 20190428T115842
        current_channel = file[-35:-32]  # e.g., muv
        if current_time_stamp == last_time_stamp and current_channel == last_channel:

            # remove s0 files
            if 'a0' in last_file:
                crappy_files.append(last_file.replace('a0', 's0'))
                os.remove(last_file.replace('a0', 's0'))

            # remove r0 files
            else:
                crappy_files.append(last_file)
                os.remove(last_file)

        # update my variables
        last_time_stamp = current_time_stamp
        last_channel = current_channel
        last_file = file

    # remember which old files were deleted
    exclude_old_files(location, get_file_names(crappy_files))


def delete_xml(location):
    """
    Delete the XML files for an orbit.

    Parameters
    ----------
    location : str
        The file path to the XML files.

    Returns
    -------
    None.
    """

    # delete xml files and record their file names
    xml_files = find_all('*.xml', location + '/')

    # empty lists are False; non-empty lists are True
    if xml_files:
        exclude_old_files(location, get_file_names(xml_files))
        for i in xml_files:
            os.remove(i)

def find_all(pattern, path):
    """
    Find all files with a specified pattern.

    Parameters
    ----------
    pattern : str
        A Unix-style string to search for, e.g., '*.pdf'.
    path : str
        A Unix-style string of the path to search for the name, e.g., '/Users/kyco2464/'.

    Returns
    -------
    result : list
        The complete paths containing the pattern.
    """

    return sorted(glob.glob(os.path.join(path, pattern)))


def exclude_old_files(path, files):
    """
    Add all files to a list where we can tell rsync to ignore these files in the future.

    Parameters
    ----------
    path : str
        The folder containing bad files.
    files : list
        A list of strings of bad files.

    Returns
    -------
    None.
    """

    # if the file doesn't exist, create it
    if not os.path.exists(os.path.join(path, 'excluded_files.txt')):
        try:
            with open(path + 'excluded_files.txt', 'w'):
                pass
        except OSError:
            raise Exception('There was an OSError when trying to make ' + path + 'orbit' + block)

    # add old data to a list; note the 'a' means append, so we guarantee not to overwrite anything
    with open(path + 'excluded_files.txt', 'a') as f:
        for i in files:
            f.write(i + '\n')


def get_file_names(file_list):
    """
    Get the absolute path of the file names.

    Parameters
    ----------
    file_list: list
        Absolute paths to files, e.g., ['/Users/username/file1.txt', '/Users/username/file2.txt'].

    Returns
    -------
    relative_files: list
        A list of just the file names, e.g., ['file1.txt', 'file2.txt'].
    """

    relative_files = [fls.split('/')[-1] for fls in file_list]
    return relative_files


def sync_data(spice=True, l1b=True, orbit_lim=None):
    """
    Synchronize new data from the VM and remove any old files that have been replaced by newer versions.

    Parameters
    ----------
    spice : bool
        Whether or not to sync SPICE kernels. Defaults to True.
    l1b : bool
        Whether or not to sync level 1B data. Defaults to True.

    Returns
    -------
    None.
    """

    # get starting time
    t0 = time.time()

    # define VM-related variables
    vm = 'maven-iuvs-itf'
    login = vm_username + '@' + vm + ':'
    production_l1b = '/maven_iuvs/production/products/level1b/'
    stage_l1b = '/maven_iuvs/stage/products/level1b/'
    vm_production_l1b = login + production_l1b
    vm_stage_l1b = login + stage_l1b
    vm_spice = login + '/maven_iuvs/stage/anc/spice/'

    # try to sync the files, if it fails, user probably isn't on the VPN
    try:

        # sync SPICE kernels
        if spice is True:
            update_spice_kernels(vm_spice, spice_directory)

        # sync level 1B data
        if l1b is True:
            # get the orbit blocks in the production and stage environments
            clear_line()
            print('Fetching level 1B production and stage folders from the VM...', end='\r')
            l1b_production_folders = get_vm_folders(vm, vm_username, vm_password, production_l1b)
            l1b_stage_folders = get_vm_folders(vm, vm_username, vm_password, stage_l1b)
            l1b_production_folders = [l1b_production_folders[-2]]
            l1b_stage_folders = [l1b_stage_folders[-9]]
            import pdb; pdb.set_trace()
            # sync and clean the level 1B production and stage environments
            clear_line()
            print('Syncing level 1B data...')
            sync_stage(vm_stage_l1b, l1b_stage_folders, os.path.join(data_directory, 'level1b'))
            sync_production(vm_production_l1b, l1b_production_folders,
                            os.path.join(data_directory, 'level1b'))

    except:
        raise Exception('Error encountered. Are you on the VPN?')

    # get ending time
    t1 = time.time()
    seconds = t1 - t0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    # tell us how long it took
    clear_line()
    print('Data syncing and cleanup took %.2d:%.2d:%.2d.' % (h, m, s))


def sync_stage(stage, stage_folders, destination):
    """
    Sync the stage environment with the VM.

    Parameters
    ----------
    stage : str
        The stage environment, e.g., 'username@vm:path_to_stage'
    stage_folders : list
        Strings of the stage folders.
    destination : str
        Path to where to save the stage data.

    Returns
    -------
    None.
    """

    # sync the stage environment
    for folder in stage_folders:
        make_block_directory(os.path.join(destination, folder))
        make_blank_file('excluded_files.txt', os.path.join(destination, folder + '/'))

        # synchronize stage files
        clear_line()
        print('Syncing stage folder ' + folder + '...', end='\r')
        rsync.run(source=os.path.join(stage, folder), destination=os.path.join(destination, folder),
                  options=['-trzL', '--out-format=%n',
                           '--exclude-from=' + os.path.join(destination, folder, 'excluded_files.txt')])

        # remove old files from directory
        clear_line()
        print('Removing files from stage folder ' + folder + '...', end='\r')
        clean_directory(destination, folder)


def sync_production(production, production_folders, destination):
    """
    Sync the production environment with the VM.

    Parameters
    ----------
    production : str
        The production environment, e.g., 'username@vm:path_to_production'.
    production_folders : str
        A list of strings of the production folders.
    destination : str
        Path to where to save the production data.

    Returns
    -------
    None.
    """

    # sync the production environment
    for folder in production_folders:

        # skip the cruise folder in level 1B
        if folder == 'cruise':
            clear_line()
            print('Skipping the cruise folder...', end='\r')
            continue

        # skip the anc_ps folder in level 1C
        if folder == 'anc_ps':
            clear_line()
            print('Skipping the anc_ps folder...', end='\r')
            continue

        # ensure the folder and excluded_files.txt exist
        make_block_directory(os.path.join(destination, folder))
        make_blank_file('excluded_files.txt', os.path.join(destination, folder + '/'))

        # synchronize production files
        clear_line()
        print('Syncing production folder ' + folder + '...', end='\r'),
        rsync.run(source=os.path.join(production, folder), destination=os.path.join(destination, folder),
                  options=['-trzL', '--out-format=%n',
                           '--exclude-from=' + os.path.join(destination, folder, 'excluded_files.txt')])

        # remove old files from directory
        clear_line()
        print('Removing files from production folder ' + folder + '...', end='\r')
        clean_directory(destination, folder)
