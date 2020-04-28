import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.io import fits
from skimage.transform import resize

# data directory
data_directory = '/Volumes/Gravity/work/data/maven_iuvs/'#'/Volumes/Samsung_T5/iuvs_data/'

# SPICE kernel directory and paths
spice_directory = '/Volumes/Gravity/work/data/maven_iuvs/spice/'#'/Volumes/Samsung_T5/spice/'

pyuvs_directory = '/Users/masunaga/work/save_data/iuvs/'

def beta_flip(hdul):
    """
    Determine the spacecraft orientation and see if the APP is "beta-flipped," meaning rotated 180 degrees.
    This compares the instrument x-axis direction to the spacecraft velocity direction in an inertial reference frame,
    which are either (nearly) parallel or anti-parallel.

    Parameters
    ----------
    hdul : object
        Opened FITS file.

    Returns
    -------
    beta_flipped : bool, str
        Returns bool True of False if orientation can be determined, otherwise returns the string "unknown".

    """

    # get the instrument's x-direction vector which is parallel to the spacecraft's motion
    vi = hdul['spacecraftgeometry'].data['vx_instrument_inertial'][-1]

    # get the spacecraft's velocity vector
    vs = hdul['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][-1]

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


def swath_geometry(orbit_number, data_directory=data_directory):
    """
    Determine how many swaths taken during a MAVEN/IUVS apoapse disk scan, which swath each file belongs to,
    whether the MUV settings were for daytime or nighttime, and the beta-angle orientation of the APP.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
        Absolute path to your IUVS level 1B data directory.

    Returns
    -------
    swath_info : dict
        A dictionary containing filepaths to the requested data files, the number of swaths, the swath number
        for each data file, whether or not the file is a dayside file, and whether the APP was beta-flipped
        during this orbit.

    """

    # get list of FITS files for given orbit number
    files, n_files = get_files(orbit_number, data_directory=data_directory, segment='apoapse', channel='muv',
                               count=True)

    # make sure there are files for the requested orbit.
    if n_files != 0:

        # set initial counters
        n_swaths = 0
        prev_ang = 999

        # arrays to hold final file paths, etc.
        filepaths = []
        daynight = []
        swath = []

        # loop through files...
        for i in range(len(files)):

            # open FITS file
            hdul = fits.open(files[i])

            # check for and skip single integrations
            if hdul[0].data.ndim == 2:
                continue

            # and if not...
            else:

                # determine if beta-flipped
                flipped = beta_flip(hdul)

                # store filepath
                filepaths.append(files[i])

                # determine if dayside or nightside
                if hdul['observation'].data['mcp_volt'] > 700:
                    dayside = False
                    daynight.append(0)
                else:
                    dayside = True
                    daynight.append(1)

                # extract integration extension
                integration = hdul['integration'].data

                # calcualte mirror direction
                mirror_dir = np.sign(integration['mirror_deg'][-1] - integration['mirror_deg'][0])
                if prev_ang == 999:
                    prev_ang *= mirror_dir

                # check the angles by seeing if the mirror is still scanning in the same direction
                ang0 = integration['mirror_deg'][0]
                if ((mirror_dir == 1) & (prev_ang > ang0)) | ((mirror_dir == -1) & (prev_ang < ang0)):
                    # increment the swath count
                    n_swaths += 1

                # store swath number
                swath.append(n_swaths - 1)

                # change the previous angle comparison value
                prev_ang = integration['mirror_deg'][-1]

    # if there are no files, then return empty lists
    else:
        filepaths = []
        n_swaths = 0
        swath = []
        daynight = []
        flipped = 'unknown'

    # make a dictionary to hold all this shit
    swath_info = {
        'filepaths': np.array(filepaths),
        'n_swaths': n_swaths,
        'swath_number': np.array(swath),
        'dayside': np.array(daynight),
        'beta_flip': flipped
    }

    # return the dictionary
    return swath_info


def highres_swath_geometry(hdul, res=200):
    """
    Generates an artificial high-resolution slit, calculates viewing geometry and surface-intercept map.

    Parameters
    ----------
    hdul : object
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
    sza : array
        Array of solar zenith angles for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't
        intercept the surface of Mars.
    local_time : array
        Array of local times for the centers of each high-resolution artificial pixel. NaNs if pixel doesn't intercept
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

    # get the slit width in degrees
    #from .variables import slit_width as slit_width
    slit_width = 10.64

    # calculate beta-flip state
    flipped = beta_flip(hdul)

    # get swath vectors, ephemeris times, and mirror angles
    vec = hdul['pixelgeometry'].data['pixel_vec']
    et = hdul['integration'].data['et']
    angles = hdul['integration'].data['mirror_deg'] * 2  # convert from mirror angles to FOV angles
    dang = np.mean(np.diff(angles[:-1]))  # calculate the average mirror step size

    # get dimensions of the input data
    dims = np.shape(hdul['primary'].data)
    n_int = dims[0]
    n_spa = dims[1]

    # set the high-resolution slit width and calculate the number of high-resolution integrations
    hifi_spa = res
    hifi_int = int(hifi_spa / n_spa * n_int)

    # make arrays of ephemeris time and array to hold the new swath vector calculations
    et_arr = np.expand_dims(et, 1) * np.ones((n_int, n_spa))
    et_arr = resize(et_arr, (hifi_int, hifi_spa), mode='edge')
    vec_arr = np.zeros((hifi_int + 1, hifi_spa + 1, 3))

    # make an artificially-divided slit and create new array of swath vectors
    n_int_div = int(hifi_int / n_int)
    for integ in range(n_int):
        lower_left = vec[0, :, 0, 0]
        upper_left = vec[-1, :, 0, 1]
        lower_right = vec[0, :, -1, 2]
        upper_right = vec[-1, :, -1, 3]

        for e in range(3):
            a = np.linspace(lower_left[e], upper_left[e], hifi_int + 1)
            b = np.linspace(lower_right[e], upper_right[e], hifi_int + 1)
            vec_arr[:, :, e] = np.array([np.linspace(i, j, hifi_spa + 1) for i, j in zip(a, b)])

    # resize array to extract centers
    vec_arr = resize(vec_arr, (hifi_int, hifi_spa, 3), anti_aliasing=True)

    # make empty arrays to hold geometry calculations
    latitude = np.zeros((hifi_int, hifi_spa))
    longitude = np.zeros((hifi_int, hifi_spa))
    sza = np.zeros((hifi_int, hifi_spa))
    phase_angle = np.zeros((hifi_int, hifi_spa))
    emission_angle = np.zeros((hifi_int, hifi_spa))
    local_time = np.zeros((hifi_int, hifi_spa))
    context_map = np.zeros((hifi_int, hifi_spa, 3))

    # load Mars surface map and switch longitude domain from [-180,180) to [0, 360)
    mars_surface_map = plt.imread(os.path.join(pyuvs_directory, 'ancillary/surface_map.jpg'))
    offset_map = np.zeros_like(mars_surface_map)
    offset_map[:, :1800, :] = mars_surface_map[:, 1800:, :]
    offset_map[:, 1800:, :] = mars_surface_map[:, :1800, :]
    mars_surface_map = offset_map

    # calculate intercept latitude and longitude using SPICE, looping through each high-resolution pixel
    target = 'Mars'
    frame = 'IAU_Mars'
    abcorr = 'LT+S'
    observer = 'MAVEN'
    body = 499  # Mars IAU code

    for i in range(hifi_int):
        for j in range(hifi_spa):
            et = et_arr[i, j]
            los_mid = vec_arr[i, j, :]

            # try to perform the SPICE calculations and record the results
            try:

                # calculate surface intercept
                spoint, trgepc, srfvec = spice.sincpt('Ellipsoid', target, et, frame,
                                                      abcorr, observer, frame, los_mid)

                # calculate illumination angles
                trgepc, srfvec, phase_for, solar, emissn = spice.ilumin('Ellipsoid', target, et, frame,
                                                                        abcorr, observer, spoint)

                # convert from rectangular to spherical coordinates
                rpoint, colatpoint, lonpoint = spice.recsph(spoint)

                # convert longitude from domain [-pi,pi) to [0,2pi)
                if lonpoint < 0.:
                    lonpoint += 2 * np.pi

                # convert ephemeris time to local solar time
                hr, mn, sc, time, ampm = spice.et2lst(et, body, lonpoint, 'planetocentric', timlen=256, ampmlen=256)

                # convert spherical coordinates to latitude and longitude in degrees
                latitude[i, j] = np.degrees(np.pi / 2 - colatpoint)
                longitude[i, j] = np.degrees(lonpoint)

                # convert illumination angles to degrees and record
                sza[i, j] = np.degrees(solar)
                phase_angle[i, j] = np.degrees(phase_for)
                emission_angle[i, j] = np.degrees(emissn)

                # convert local solar time to decimal hour
                local_time[i, j] = hr + mn / 60 + sc / 3600

                # convert latitude and longitude to pixel coordinates
                map_lat = int(np.round(np.degrees(colatpoint), 1) * 10)
                map_lon = int(np.round(np.degrees(lonpoint), 1) * 10)

                # instead of changing an alpha layer, I just multiply an RGB triplet by a scaling fraction in order to
                # make it darker; determine that scalar here based on solar zenith angle
                if (sza[i, j] > 90) & (sza[i, j] <= 102):
                    twilight = 0.7
                elif sza[i, j] > 102:
                    twilight = 0.4
                else:
                    twilight = 1

                # place the corresponding pixel from the high-resolution Mars map into the swath context map with the
                # twilight scaling
                context_map[i, j, :] = mars_surface_map[map_lat, map_lon, :] / 255 * twilight

            # if the SPICE calculation fails, this (probably) means it didn't intercept the planet, so record that
            # as a NaN
            except:
                latitude[i, j] = np.nan
                longitude[i, j] = np.nan
                sza[i, j] = np.nan
                phase_angle[i, j] = np.nan
                emission_angle[i, j] = np.nan
                local_time[i, j] = np.nan

    # create an meshgrid of angular coordinates for the high-resolution pixel edges
    x, y = np.meshgrid(np.linspace(0, slit_width, hifi_spa + 1),
                       np.linspace(angles[0] - dang / 2, angles[-1] + dang / 2, hifi_int + 1))

    # calculate the angular separation between pixels
    dslit = slit_width / hifi_spa

    # create an meshgrid of angular coordinates for the high-resolution pixel centers
    cx, cy = np.meshgrid(
        np.linspace(0 + dslit, slit_width - dslit, hifi_spa),
        np.linspace(angles[0], angles[-1], hifi_int))

    # beta-flip the coordinate arrays if necessary
    if flipped == True:
        x = np.fliplr(x)
        y = (np.fliplr(y) - 90) / (-1) + 90
        cx = np.fliplr(cx)
        cy = (np.fliplr(cy) - 90) / (-1) + 90

    # return the geometry and coordinate arrays
    return latitude, longitude, sza, local_time, x, y, cx, cy, context_map


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
    label_style = dict(fmt='$%i\degree$', inline=True, fontsize=8)
    dlat = 30
    dlon = 30

    # set longitude to -180 to 180
    longitude[np.where(longitude >= 180)] -= 360

    # draw latitude contours, place labels, and remove label rotation
    latc = axis.contour(cx, cy, latitude, levels=np.arange(-90, 90, dlat), **grid_style)
    try:
        latl = axis.clabel(latc, **label_style)
        [l.set_rotation(0) for l in latl]
    except:
        pass

    # longitude contours are complicated... first up setting the hard threshold at -180 to 180
    tlon = np.copy(longitude)
    tlon[np.where((tlon <= -170) | (tlon >= 170))] = np.nan
    lonc1 = axis.contour(cx, cy, tlon, levels=np.arange(-180, 180, dlon), **grid_style)
    try:
        lonl1 = axis.clabel(lonc1, **label_style)
        [l.set_rotation(0) for l in lonl1]
    except:
        pass

    # then the hard threshold at 360 to 0
    tlon = np.copy(longitude)
    tlon[np.where(tlon < 0)] += 360
    tlon[np.where((tlon <= 10) | (tlon >= 350))] = np.nan
    lonc2 = axis.contour(cx, cy, tlon, levels=[180], **grid_style)
    try:
        lonl2 = axis.clabel(lonc2, **label_style)
        [l.set_rotation(0) for l in lonl2]
    except:
        pass


def angle_meshgrid(hdul):
    """
    Returns a meshgrid of observations in angular space.

    Parameters
    ----------
    hdul : object
        Opened FITS file.
    beta_flip : bool
        Whether or not the APP beta-angle was flipped for this observation.

    Returns
    X : array
        An (n+1,m+1) array of pixel longitudes with "n" = number of slit elements and "m" = number of integrations.
    Y : array
        An (n+1,m+1) array of pixel latitudes with "n" = number of slit elements and "m" = number of integrations.
    """

    # width of the slit in degrees
    slit_width = 10.64

    # get angles of observation and convert from mirror angles to FOV angles
    angles = hdul['integration'].data['mirror_deg'] * 2

    # calculate change in angle between integrations
    dang = np.mean(np.diff(angles[:-1]))

    # get number of spatial elements and integrations
    dims = hdul['primary'].data.shape
    n_integrations = dims[0]
    n_spatial = dims[1]

    # calculate meshgrids
    X, Y = np.meshgrid(np.linspace(0, slit_width, n_spatial + 1),
                       np.linspace(angles[0] - dang / 2, angles[-1] + dang / 2, n_integrations + 1))

    # determine beta-flipping
    flipped = beta_flip(hdul)

    # rotate if beta-flipped
    if flipped == True:
        X = np.fliplr(X)
        Y = (np.fliplr(Y) - 90) / (-1) + 90

    # return meshgrids
    return X, Y


def get_files(orbit_number, data_directory=data_directory, segment='apoapse', channel='muv', count=False):
    """
    Return file paths to FITS files for a given orbit number.

    Parameters
    ----------
    orbit_number : int
        The MAVEN orbit number.
    data_directory : str
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
    filepath = os.path.join(data_directory, 'level1b/orbit%.5d/' % (orbit_block))

    # format of FITS file names
    filename_str = '*%s-orbit%.5d-%s*.fits.gz' % (segment, orbit_number, channel)

    # get list of files
    files = sorted(glob.glob(os.path.join(filepath, filename_str)))

    # get number of files
    n_files = int(len(files))

    # return the list of files with the count if requested
    if count == False:
        return files
    else:
        return files, n_files
