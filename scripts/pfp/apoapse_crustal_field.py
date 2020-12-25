import glob
import os
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.io import fits
from skimage.transform import resize

# import PyUVS
import sys
#sys.path.append('/Users/milby/Documents/Work/MAVEN/Python')
from PyUVS.spice import load_iuvs_spice

# set data directory
#data_directory = '/Volumes/MAVEN Data/iuvs_data/'
from variables import iuvdataloc as data_directory
from variables import saveloc

def get_files(orbit_number, directory=data_directory, segment='apoapse', channel='muv', count=False):
    """
    Return file paths to FITS files for a given orbit number.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    directory : str
        Absolute system path to the location containing orbit block folders ("orbit01300", orbit01400", etc.)
    segment : str
        The orbit segment for which you want data files. Defaults to 'apoapse'.
    channel : str
        The instrument channel. Defaults to 'muv'.
    count : bool
        Whether or not to return the number of files.

    Returns
    -------
    files : array
        A sorted list of the file paths to the FITS files.
    n_files : int
        The number of files.
    """

    # determine orbit block (directories which group data by 100s)
    orbit_block = int(orbit_number / 100) * 100

    # location of FITS files (this will change depending on the user)
    filepath = os.path.join(directory, 'level1b/orbit%.5d/' % orbit_block)

    # format of FITS file names
    filename_str = '*%s-orbit%.5d-%s*.fits.gz' % (segment, orbit_number, channel)

    # get list of files
    files_list = sorted(glob.glob(os.path.join(filepath, filename_str)))

    # get number of files
    files_count = int(len(files_list))

    # return the list of files with the count if requested
    if not count:
        return files_list
    else:
        return files_list, files_count


def beta_flip(hdulist):
    """
    Determine the spacecraft orientation and see if the APP is "beta-flipped," meaning rotated 180 degrees.
    This compares the instrument x-axis direction to the spacecraft velocity direction in an inertial reference frame,
    which are either (nearly) parallel or anti-parallel.

    Parameters
    ----------
    hdulist : hdulistist
        Opened FITS file.

    Returns
    -------
    beta_flipped : bool, str
        Returns bool True of False if orientation can be determined, otherwise returns the string "unknown".

    """

    # get the instrument's x-direction vector which is parallel to the spacecraft's motion
    vi = hdulist['spacecraftgeometry'].data['vx_instrument_inertial'][-1]

    # get the spacecraft's velocity vector
    vs = hdulist['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][-1]

    # determine orientation between vectors (if they are aligned or anti-aligned)
    app_sign = np.sign(np.dot(vi, vs))

    # if negative then no beta flipping, if positive then yes beta flipping, otherwise state is unknown
    if app_sign == -1:
        beta_flipped = False
    elif app_sign == 1:
        beta_flipped = True
    else:
        beta_flipped = 'unknown'

    # return the result
    return beta_flipped


def get_apoapse_files(orbit_number, channel='muv', directory=data_directory):
    """
    Convenience function for apoapse data. In addition to returning file paths to the data, it determines how many
    swaths were taken, which swath each file belongs to since there are often 2-3 files per swath, whether the MCP
    voltage settings were for daytime or nighttime, the mirror step between integrations, and the beta-angle orientation
    of the APP.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    channel : str
        The instrument channel. Defaults to 'muv'.
    directory : str
        Absolute path to your IUVS level 1B data directory which has the orbit blocks, e.g., "orbit03400, orbit03500,"
        etc.

    Returns
    -------
    swath_info : dict
        A dictionary containing filepaths to the requested data files, the number of swaths, the swath number
        for each data file, whether or not the file is a dayside file, and whether the APP was beta-flipped
        during this orbit.

    """

    # get list of FITS files for given orbit number
    files_list, files_count = get_files(orbit_number, directory=directory, segment='apoapse', channel=channel,
                                        count=True)

    # set initial counters
    no_swaths = 0
    prev_ang = 999

    # arrays to hold final file paths, etc.
    filepaths = []
    daynight = []
    swath = []
    flipped = []

    # loop through files...
    for i in range(files_count):

        # open FITS file
        hdullist = fits.open(files_list[i])

        # skip single integrations, they are more trouble than they're worth
        if hdullist['primary'].data.ndim == 2:
            continue

        # determine if beta-flipped
        flipped.append(beta_flip(hdullist))

        # store filepath
        filepaths.append(files_list[i])

        # determine if dayside or nightside
        if hdullist['observation'].data['mcp_volt'] > 700:
            daynight.append(False)
        else:
            daynight.append(True)

        # calcualte mirror direction
        mirror_dir = np.sign(hdullist['integration'].data['mirror_deg'][-1] -
                             hdullist['integration'].data['mirror_deg'][0])
        if prev_ang == 999:
            prev_ang *= mirror_dir

        # check the angles by seeing if the mirror is still scanning in the same direction
        ang0 = hdullist['integration'].data['mirror_deg'][0]
        if ((mirror_dir == 1) & (prev_ang > ang0)) | ((mirror_dir == -1) & (prev_ang < ang0)):
            # increment the swath count
            no_swaths += 1

        # store swath number
        swath.append(no_swaths - 1)

        # change the previous angle comparison value
        prev_ang = hdullist['integration'].data['mirror_deg'][-1]

    # check beta flipping
    if len(np.unique(flipped)) != 1:
        flipped = mode(flipped)

    # make a dictionary to hold all this shit
    swath_info = {
        'files': np.array(filepaths),
        'n_swaths': no_swaths,
        'swath_number': np.array(swath),
        'dayside': np.array(daynight),
        'beta_flip': flipped,
    }

    # return the dictionary
    return swath_info


def highres_swath_geometry(hdulist, res=200):
    """
    Generates an artificial high-resolution slit, calculates viewing geometry and surface-intercept map.

    Parameters
    ----------
    hdulist : hdulistist
        Opened FITS file.
    res : int, optional
        The desired number of artificial elements along the slit. Defaults to 200.

    Returns
    -------
    latitude : array
        Array of latitudes for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't intercept
        the surface of Mars.
    longitude : array
        Array of longitudes for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't intercept
        the surface of Mars.
    x : array
        Horizontal coordinate edges in angular space. Has shape (n+1, m+1) for geometry arrays with shape (n,m).
    y : array
        Vertical coordinate edges in angular space. Has shape (n+1, m+1) for geometry arrays with shape (n,m).
    cx : array
        Horizontal coordinate centers in angular space. Same shape as geometry arrays.
    cy : array
        Vertical coordinate centers in angular space. Same shape as geometry arrays.
    context_map : array
        High-resolution image of the Mars surface as intercepted by the swath. RGB tuples with shape (n,m,3).
    """

    # calculate beta-flip state
    flipped = beta_flip(hdulist)

    # get swath vectors, ephemeris times, and mirror angles
    vec = hdulist['pixelgeometry'].data['pixel_vec']
    et = hdulist['integration'].data['et']

    # get dimensions of the input data
    n_int = hdulist['integration'].data.shape[0]
    n_spa = len(hdulist['binning'].data['spapixlo'][0])

    # set the high-resolution slit width and calculate the number of high-resolution integrations
    hifi_spa = res
    hifi_int = int(hifi_spa / n_spa * n_int)

    # make arrays of ephemeris time and array to hold the new swath vector calculations
    et_arr = np.expand_dims(et, 1) * np.ones((n_int, n_spa))
    et_arr = resize(et_arr, (hifi_int, hifi_spa), mode='edge')
    vec_arr = np.zeros((hifi_int + 1, hifi_spa + 1, 3))

    # make an artificially-divided slit and create new array of swath vectors
    if flipped:
        lower_left = vec[0, :, 0, 0]
        upper_left = vec[-1, :, 0, 1]
        lower_right = vec[0, :, -1, 2]
        upper_right = vec[-1, :, -1, 3]
    else:
        lower_left = vec[0, :, 0, 1]
        upper_left = vec[-1, :, 0, 0]
        lower_right = vec[0, :, -1, 3]
        upper_right = vec[-1, :, -1, 2]

    for e in range(3):
        a = np.linspace(lower_left[e], upper_left[e], hifi_int + 1)
        b = np.linspace(lower_right[e], upper_right[e], hifi_int + 1)
        vec_arr[:, :, e] = np.array([np.linspace(i, j, hifi_spa + 1) for i, j in zip(a, b)])

    # resize array to extract centers
    vec_arr = resize(vec_arr, (hifi_int, hifi_spa, 3), anti_aliasing=True)

    # make empty arrays to hold geometry calculations
    latitude = np.zeros((hifi_int, hifi_spa))*np.nan
    longitude = np.zeros((hifi_int, hifi_spa))*np.nan
    context_map_arr = np.zeros((hifi_int, hifi_spa, 3))*np.nan

    # load Mars magnetic field map
    mars_surface_map = plt.imread(saveloc+'/misc_items/bfield_map/br_no_labels.jpg')
    mars_surface_map = resize(mars_surface_map, [1800, 3600, 3])

    # calculate intercept latitude and longitude using SPICE, looping through each high-resolution pixel
    target = 'Mars'
    frame = 'IAU_Mars'
    abcorr = 'LT+S'
    observer = 'MAVEN'

    for i in range(hifi_int):
        for j in range(hifi_spa):
            et = et_arr[i, j]
            los_mid = vec_arr[i, j, :]

            # try to perform the SPICE calculations and record the results
            # noinspection PyBroadException
            try:

                # calculate surface intercept
                spoint, trgepc, srfvec = spice.sincpt('Ellipsoid', target, et, frame, abcorr, observer, frame, los_mid)

                # convert from rectangular to spherical coordinates
                rpoint, colatpoint, lonpoint = spice.recsph(spoint)

                # convert longitude from domain [-pi,pi) to [0,2pi)
                if lonpoint < 0.:
                    lonpoint += 2 * np.pi

                # convert spherical coordinates to latitude and longitude in degrees
                latitude[i, j] = np.degrees(np.pi / 2 - colatpoint)
                longitude[i, j] = np.degrees(lonpoint)

                # convert latitude and longitude to pixel coordinates
                map_lat = int(np.round(np.degrees(colatpoint), 1) * 10)
                map_lon = int(np.round(np.degrees(lonpoint), 1) * 10)

                # place the corresponding pixel from the high-resolution Mars map into the swath context map
                context_map_arr[i, j, :] = mars_surface_map[map_lat, map_lon, :]

            # if the SPICE calculation fails, this (probably) means it didn't intercept the planet
            except:
                pass

    # get mirror angles
    angles = hdulist['integration'].data['mirror_deg'] * 2  # convert from mirror angles to FOV angles
    dang = np.diff(angles)[0]

    # create an meshgrid of angular coordinates for the high-resolution pixel edges
    x, y = np.meshgrid(np.linspace(0, slit_width_deg, hifi_spa + 1),
                       np.linspace(angles[0] - dang / 2, angles[-1] + dang / 2, hifi_int + 1))

    # calculate the angular separation between pixels
    dslit = slit_width_deg / hifi_spa

    # create an meshgrid of angular coordinates for the high-resolution pixel centers
    cx, cy = np.meshgrid(
        np.linspace(0 + dslit, slit_width_deg - dslit, hifi_spa),
        np.linspace(angles[0], angles[-1], hifi_int))

    # beta-flip the coordinate arrays if necessary
    if flipped:
        x = np.fliplr(x)
        y = (np.fliplr(y) - 90) / (-1) + 90
        cx = np.fliplr(cx)
        cy = (np.fliplr(cy) - 90) / (-1) + 90

    # convert longitude to [-180,180)
    longitude[np.where(longitude > 180)] -= 360

    # return the geometry and coordinate arrays
    return latitude, longitude, x, y, cx, cy, context_map_arr


def latlon_grid(cx, cy, latitude, longitude, axis):
    """
    Places latitude/longitude grid lines and labels on an apoapse swath image.

    Parameters
    ----------
    cx : array
        Horizontal coordinate centers in angular space.
    cy : array
        Vertical coordinate centers in angular space.
    latitude : array
        Pixel latitude values (same shape as cx and vy).
    longitude : array
        Pixel longitude values (same shape as cx and vy).
    axis : Artist
        Axis in which you want the latitude/longitude lines drawn.
    """
    # set line and label styles
    grid_style = dict(colors='white', linestyles='-', linewidths=0.5)
    label_style = dict(fmt=r'$%i\degree$', inline=True, fontsize=8)
    dlat = 30
    dlon = 30

    # set longitude to -180 to 180
    longitude[np.where(longitude >= 180)] -= 360

    # draw latitude contours, place labels, and remove label rotation
    latc = axis.contour(cx, cy, latitude, levels=np.arange(-90, 90, dlat), **grid_style)
    latl = axis.clabel(latc, **label_style)
    [l.set_rotation(0) for l in latl]

    # longitude contours are complicated... first up setting the hard threshold at -180 to 180
    tlon = np.copy(longitude)
    tlon[np.where((tlon <= -170) | (tlon >= 170))] = np.nan
    lonc1 = axis.contour(cx, cy, tlon, levels=np.arange(-180, 180, dlon), **grid_style)
    lonl1 = axis.clabel(lonc1, **label_style)
    [l.set_rotation(0) for l in lonl1]

    # then the hard threshold at 360 to 0 using -180 as the label
    tlon = np.copy(longitude)
    tlon[np.where(tlon >= 0)] -= 360
    tlon[np.where((tlon <= -190) | (tlon >= -170))] = np.nan
    lonc2 = axis.contour(cx, cy, tlon, levels=[-180], **grid_style)
    lonl2 = axis.clabel(lonc2, **label_style)
    [l.set_rotation(0) for l in lonl2]


# filter warnings, there are a lot and they are annoying
filterwarnings('ignore')

# furnish SPICE kernels
load_iuvs_spice()

# set slit width
slit_width_deg = 10

# get orbit number
orbit = 592

# get files
apoapse_data = get_apoapse_files(orbit, channel='fuv')
files = apoapse_data['files']
n_files = len(files)
n_swaths = apoapse_data['n_swaths']
swath_number = apoapse_data['swath_number']
flip = apoapse_data['beta_flip']

# make a figure and axes
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# loop through files
for file in range(n_files):

    # open file
    hdul = fits.open(files[file])

    # determine spatial and spectral bins
    n_spatial = len(hdul['binning'].data['spapixlo'][0])
    n_spectral = len(hdul['binning'].data['spepixlo'][0])

    # try to display the geometry data (sometimes coordinates are undefined and this fails)
    try:

        # calculate high-resolution geometry information
        lat, lon, X, Y, cX, cY, context_map = highres_swath_geometry(hdul)

        # reshape map colors array
        context_map_colors = context_map.reshape(context_map.shape[0] * context_map.shape[1], context_map.shape[2])

        # reverse Y array so it goes top-to-bottom instead of bottom-to-top
        Y = (120 - Y) + 60
        cY = (120 - cY) + 60

        # offset X array by swath number
        X += slit_width_deg * swath_number[file]
        cX += slit_width_deg * swath_number[file]

        # display context map
        ax.pcolormesh(X, Y, np.ones_like(X), color=context_map_colors, linewidth=0, edgecolors='none',
                               rasterized=True).set_array(None)

        # draw latitude/longitude grid on context map
        latlon_grid(cX, cY, lat, lon, ax)

    except (ValueError, TypeError):
        pass

# figure out where to save it
savepath = saveloc + '/test/test_bfield_projection_nb.png'

# save quicklook
plt.savefig(savepath, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
plt.close('all')
