# for math
import numpy as np

# for plotting and graphics display
import matplotlib.pyplot as plt
plt.rc('pdf', fonttype=42) #makes sure text isn't outlined when saved as PDF

# for placing an axis with a projection into another axis
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# for custom colormap creation and data scaling
import matplotlib.colors as colors

# for map projections and transforms
import cartopy.crs as ccrs

# for reading IDL SAV files
from scipy.io import readsav

# for reading FITS files
from astropy.io import fits

# for Linux-style file searching
import glob

# for SPICE nonsense
import spiceypy as spice
import glob
import os
import re

# for fitting
from sklearn import linear_model

import spicetools

def get_files(orbit_number, segment='apoapse', channel='fuv', count=False):

    """
    Return file paths to FITS files for a given orbit number.

    Parameters
    ----------
    orbit_number : int
        Enough already. You're smarter than this.
    segment : str
        The orbit segment for which you want data files. Defaults to 'apoapse'.
    channel : str
        MUV and/or FUV channel.
    count : bool
        Whether or not to return the number of files.

    Returns
    -------
    files : array
        A sorted list of the file paths to the FITS files.
    n_files : int
        The number of files.
    """

    # determine orbit block
    orbit_block = int(orbit_number/100)*100

    # location of FITS files (this may change depending on the user)
    filepath = '/Volumes/Fenix/work/data/maven_iuvs/level2b/orbit%.5d/' %(orbit_block)

    # format of FITS files
    if (channel == 'muv') | (channel == 'fuv'):
        filename_str = '*%s*orbit%.5d*%s*.fits.gz' %(segment, orbit_number, channel)
    elif channel == 'all':
        filename_str = '*%s*orbit%.5d*.fits.gz' %(segment, orbit_number)

    # get list of files
    files = sorted(glob.glob(filepath+filename_str))

    # get number of files
    n_files = int(len(files))

    # check to see if it was successful
    if n_files == 0:
        raise Exception('No files found. Check the data path and/or orbit number.')

    # return the list of files with the count if requested
    if count == False:
        return files
    else:
        return files, n_files




def MLR(hdul):

    """
    Performs a nightside spectral MLR.

    Parameters
    ----------
    hdul : open FITS file

    Returns
    -------
    mlr_array : array
        Derived brightness values.
    """

    # extract the dimensions of the primary extension
    dims = np.shape(hdul['primary']) # get the dimensions of the primary extension
    n_integrations = dims[0] # number of integrations
    n_spatial = dims[1] # number of spatial bins along slit
    n_spectral = dims[2] # number of spectral bins

    # get the spectral wavelengths
    wavelength = hdul['observation'].data[0]['wavelength'][0]

    # this is Justin's DN threshold for valid data
    dn_threshold = 3600*4*16

    # determine wavelength index corresponding to fit start and length of fitting region
    if n_spectral == 40:
        fit_start = 32
        fit_length = n_spectral
    elif n_spectral == 174:
        fit_start = 17
        fit_length = n_spectral

    # load 256-spectral-bin templates
    templates = np.genfromtxt('Data/muv_templates.dat', skip_header=True)
    template_wavelength = templates[:,0][fit_start:fit_start+fit_length]
    calibration_curve = templates[:,1][fit_start:fit_start+fit_length]
    template_solar_continuum = templates[:,2][fit_start:fit_start+fit_length]
    template_co_cameron = templates[:,3][fit_start:fit_start+fit_length]
    template_co2uvd = templates[:,4][fit_start:fit_start+fit_length]
    template_o2972 = templates[:,5][fit_start:fit_start+fit_length]
    template_no_nightglow = templates[:,6][fit_start:fit_start+fit_length]
    template_constant = np.ones_like(template_solar_continuum)
    muv_features = ['Solar Continuum', 'CO Cameron Bands', 'CO2 UV Doublet', 'O(1S) 297.2 nm',
                    'NO Nightglow', 'MUV Background']

    # determine spectral bin spacing
    dwavelength = np.diff(wavelength)[0]

    # make an array to hold regression coefficients
    mlr_array = np.zeros((dims[0],dims[1],3))*np.nan
    chisq_array = np.zeros((dims[0],dims[1]))*np.nan

    # loop through integrations
    for i in range(n_integrations):

        # loop through spatial bins
        for j in range(n_spatial):

            # extract the dark-subtracted detector image
            detector_image = hdul['detector_dark_subtracted'].data[i,j,:]

            # extract the error
            yerr = hdul['random_dn_unc'].data[i,j,:]

            # find all the data that are less than the DN threshold
            ygood = np.where(detector_image < dn_threshold)

            # make sure at least one point is good in order to perform the fit
            if np.size(ygood) != 0:

                X = np.array([template_solar_continuum[ygood],
                              template_no_nightglow[ygood]]).T
                Y = np.array(detector_image[ygood])
                yerr = yerr[ygood]

                lm = linear_model.LinearRegression()
                model = lm.fit(X,Y,sample_weight=(1/yerr)**2)
                coeff = model.coef_
                const = model.intercept_

                # integrate each component
                mlr_array[i,j,0] = np.trapz(coeff[0]*template_solar_continuum*calibration_curve, dx=dwavelength)
                mlr_array[i,j,1] = np.trapz(coeff[1]*template_no_nightglow*calibration_curve, dx=dwavelength)
                mlr_array[i,j,2] = np.trapz(const*template_constant*calibration_curve, dx=dwavelength)

    return mlr_array


def NO_cmap(nan_color='none'):

    """
    Makes custom MAVEN/IUVS nitric oxide (NO) nightglow colormaps based on IDL color table #8.

    Parameters
    ----------
    nan_color : str
        The choice nan color. Current options available are none, black, and grey.

    Returns
    -------
    cmap : I have no idea
        Colormap with chosen nan color.
    """

    # color sequence from black -> green -> yellow-green -> white
    no_colors = [(0, 0, 0), (0, 0.5, 0), (0.61, 0.8, 0.2), (1, 1, 1)]

    if nan_color == 'none':

        # set colormap name
        cmap_name = 'NO'

        # make a colormap using the color sequence and chosen name
        cmap = colors.LinearSegmentedColormap.from_list(cmap_name, no_colors, N=2**8)

    if nan_color == 'black':

        # set colormap name
        cmap_name = 'NOBlack'

        # make a colormap using the color sequence and chosen name
        cmap = colors.LinearSegmentedColormap.from_list(cmap_name, no_colors, N=2**8)

        # set the nan color
        cmap.set_bad((0,0,0))

    if nan_color == 'grey':

        # set colormap name
        cmap_name = 'NOGrey'

        # make a colormap using the color sequence and chosen name
        cmap = colors.LinearSegmentedColormap.from_list(cmap_name, no_colors, N=2**8)

        # set the nan color
        cmap.set_bad((0.5,0.5,0.5))

    return cmap


def radiance_multiplier(radiance, spec_bin):

    """
    Scale 40-spectral-bin NO brightnesses.

    Parameters
    ----------
    radiance : float, array
        ,NO radiances in kilorayleigh (kR).
    spec_bin : int
        The number of spectral bins.

    Returns
    -------
    radiance : float, array
        NO radiances appropriately scaled. Same type/dimensions/physical units as radiance.
    """

    # check number of spectral bins and scale if 40
    if spec_bin == 40:
        radiance *= 2.20369

    # return the scaled (or unscaled) radiance
    return radiance


def beta_flip(vi, vs):

    """
    Determine the spacecraft orientation and see if it underwent a beta flip. This compares the instrument
    x-axis direction to the spacecraft velocity direction, which are either (nearly) parallel or anti-parallel.

    Parameters
    ----------
    vi : array, tuple
        Last vector [-1] of 'vx_instrument_inertial' from `spacecraftgeometry' in IUVS FITS file.
    vs : array, tuple
        Last vector [-1] of 'v_spacecraft_rate_inertial' from `spacecraftgeometry' in IUVS FITS file.

    Returns
    -------
    beta : bool
        Beta flipped? Yes or no, true or false...

    """

    # determine orientation between vectors
    app_sig = np.sign(np.dot(vi, vs))

    # if negative, then no beta flipping; if positive, then yes beta flipping
    if app_sig == -1:
        beta = False
    elif app_sig == 1:
        beta = True

    # return the bool
    return beta



def pixel_globe(orbit_number, valid=False):

    """
    Make a pixel grid of IUVS nightside swaths, approximating the view from MAVEN's apoapsis.

    Parameters
    ----------
    orbit_number : int
        I bet you can figure this one out...
    valid : bool
        If true (default), then it applies the pixel validation criteria developed by Nick, Sonal, and Zac
        in January 2019. If false, then it displays the data as observed, without any filtering of
        cosmic rays, stray sunlight, SZA, etc.

    Returns
    -------
    x : array
        Horizontal pixel edges in kilometers from the center of Mars.
    y : array
        Vertical pixel edges in kilometers from the center of Mars.
    z : array
        Grid of projected pixel brightnesses.
    """

    # dimensions of pixel grid and width of a pixel in kilometers
    pixsize = 100 #20 #[km/pixel]
    xsize = int(8000/pixsize)
    ysize = int(8000/pixsize)

    # grid to hold sum of values falling in each pixel and a count of number of values in each pixel
    total = np.zeros((ysize,xsize))
    count = np.zeros((ysize,xsize))

    # list of FITS files for given orbit number with error handling
    files, n_files = get_files(orbit_number, count=True)
    if n_files == 0:
        raise ValueError('Invalid orbit number.')

    # variable to hold beta flip value
    flip = -1

    # loop through FITS files
    for f in range(len(files)):

        # open FITS file
        hdul = fits.open(files[f])

        # determine dimensions, and if it's a single integration, skip it
        dims = hdul['primary'].shape
        primary_array = hdul['primary'].data
        if len(dims) != 3:
            continue #skip single integrations
        n_int = dims[0]
        n_spa = dims[1]

        # also skip dayside
        if hdul['observation'].data['mcp_volt'] < 790:
            continue

        # get number of spectral bins
        spec_bin = np.shape(hdul['observation'].data['wavelength'][0][1])[0]

        # get vectors for determining beta flip
        if flip == -1:
            vi = hdul['spacecraftgeometry'].data['vx_instrument_inertial'][-1]
            vs = hdul['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][-1]
            flip = beta_flip(vi,vs)

        # fit spectra
        #primary_array = MLR(hdul)

        # this is copied directly from Sonal; someday I'll figure it out and comment...
        # essentially it finds the place where the pixel position vector intersects the 400x400 grid
        # and places the pixel value in that location
        for i in range(n_int):
            vspc = hdul['spacecraftgeometry'].data[i]['v_spacecraft']
            vspcnorm = vspc/np.linalg.norm(vspc)
            vy = hdul['spacecraftgeometry'].data[i]['vy_instrument']
            vx = np.cross(vy, vspcnorm)

            ## primary_array[0]: Ly-alpha, [1]: 1304, [2]: 1356
            for j in range(n_spa):
                primary = primary_array[i,j,1] # This data will be used to plot
                solar = primary_array[i,j,0]
                const = primary_array[i,j,2]
                if spec_bin == 40:
                    solar_max = 1.15
                    const_max = 2.
                elif spec_bin == 174:
                    solar_max = 5.
                    const_max = 9.2

                # if valid is true, then eliminate pixels which fail validation criteria
                if valid == True:
                    if (solar > solar_max) | (const > const_max):
                        continue
                    else:
                        #primary = radiance_multiplier(primary, spec_bin)
                        for m in range(4):
                            try:
                                vpix = hdul['pixelgeometry'].data[i]['pixel_vec']
                                vpixcorner = (np.squeeze(vpix[:,j,m]) + np.squeeze(vpix[:,j,4]))/2
                                vdiff = vspc - (np.dot(vspc,vpixcorner)*vpixcorner)
                                x = int(np.dot(vdiff,vx)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /pixsize+xsize/2)
                                y = int(np.dot(vdiff,vy)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /pixsize+ysize/2)
                                if (x >= 0) & (y >= 0):
                                    total[y,x] += primary
                                    count[y,x] += 1
                            except:
                                continue

                # if validation turned off, then don't filter any pixels
                elif valid == False:
                    #primary = radiance_multiplier(primary, spec_bin)
                    for m in range(4):
                        try:
                            vpix = hdul['pixelgeometry'].data[i]['pixel_vec']
                            vpixcorner = (np.squeeze(vpix[:,j,m]) + np.squeeze(vpix[:,j,4]))/2
                            vdiff = vspc - (np.dot(vspc,vpixcorner)*vpixcorner)
                            x = int(np.dot(vdiff,vx)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /pixsize+xsize/2)
                            y = int(np.dot(vdiff,vy)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /pixsize+ysize/2)
                            if (x >= 0) & (y >= 0):
                                    total[y,x] += primary
                                    count[y,x] += 1
                        except:
                            continue


    #calculate the average
    #total[np.where(count == 0)] = np.nan
    z = total/count

    # beta-flip if necessary
    if flip == True:
        z = np.fliplr(z)

    #make coordinate grids for each pixel in kilometers
    x, y = np.meshgrid(np.linspace(-xsize/2*pixsize, xsize/2*pixsize, xsize), np.linspace(-ysize/2*pixsize, ysize/2*pixsize, ysize))

    #return the coordinate grids and the spherically-projected data pixels
    return x, y, z



def mandp(ax, transform, projection):

    # this function overcomes a known text placement bug in cartopy
    def double_transform(x, y, src, target, tol=2):
        rx, ry = target.transform_point(x, y, src)
        px, py = src.transform_point(rx, ry, target)
        if abs(x - px) < tol and abs(y - py) < tol:
            return rx, ry
        else:
            return None

    # make arrays of longitude/latitude values for meridian/parallel lines and labels
    dlon = 30 # spacing between longitudes
    longitudes = np.arange(-180,180+dlon,dlon)
    dlat = 30 # spacing between latitudes
    latitudes = np.arange(-90,90+dlat,dlat)

    # longitude lines and labels
    for i in longitudes:

        # plot longitude line
        line, = ax.plot(np.ones(1800)*i, np.linspace(-90,90,1800), color='white', linewidth=0.4,
                transform=transform)

        # label longitude lines inbetween each parallel
        for j in latitudes[1:-1]:

            # check to see if label should be visible
            if double_transform(i, j+dlat/2, transform, projection):

                    # place label at latitude + dlat/2
                    text = ax.text(i, j+dlat/2, r'$%i$' %i, color='white', transform=transform,
                                       ha='center', va='center', bbox=dict(alpha=0))

    # latitude lines and labels
    for i in latitudes:

        # plot latitude line
        line, = ax.plot(np.linspace(-180,180,3600), np.ones(3600)*i, color='white', linewidth=0.4,
                transform=transform)

        # label latitude lines inbetween each meridian
        for j in longitudes:#[1:-1]:

            # check to see if label should be visible
            if double_transform(j+dlon/2, i, transform, projection):

                    # place the label at longitude + dlon/2
                    text = ax.text(j+dlon/2, i, r'$%i$' %i, color='white', transform=transform,
                                       ha='center', va='center', bbox=dict(alpha=0))



def rotation_matrix(axis, theta):

    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
    To transform a vector, calculate its dot-product with the rotation matrix.

    Parameters
    ----------
    axis : 3-element list, array, or tuple
        The rotation axis in Cartesian coordinates. Does not have to be a unit vector.
    theta : float
        The angle (in radians) to rotate about the rotation axis. Positive angles rotate counter-clockwise.

    Returns
    -------
    rotation_matrix : array with dimensions (3,3)
        The 3D rotation matrix.
    """

    # convert the axis to a numpy array and normalize it
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # calculate components of the rotation matrix elements
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    # build the rotation matrix
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # return the rotation matrix
    return rotation_matrix



# draws a grid of meridians and parallels from the perspective of MAVEN/IUVS at apoapsis
def draw_grid(ax, orbit_number):

    """
    Draw and label a grid of latitude and longitude lines as viewed by MAVEN/IUVS at apoapsis.

    Parameters
    ----------
    ax : matplotlib Artist
        The axis in which to draw the perspective grid of latitude and longitude lines.
    orbit_numer : int
        The orbit number for which to calculate the grid.

    Returns
    -------
    None.
    """

    # furnish SPICE kernels
    spicetools.load_iuvs_spice()

    # calculate various parameters using SPICE
    files = get_files(orbit_number)
    hdul = fits.open(files[0])
    TIMFMT = 'YYYY-MON-DD HR:MN:SC.## (UTC) ::UTC ::RND'
    TIMLEN =  len(TIMFMT)
    target = 'Mars'
    frame = 'MAVEN_MME_2000'
    abcorr   = 'LT+S'
    observer = 'MAVEN'
    et_apr = [hdul['integration'].data['et'][0], hdul['integration'].data['et'][0]+4800.]
    cnfine = spice.utils.support_types.SPICEDOUBLE_CELL(2)
    spice.wninsd(et_apr[0], et_apr[1], cnfine)
    result = spice.utils.support_types.SPICEDOUBLE_CELL(100)
    spice.gfdist('Mars', 'none', observer, 'LOCMAX', 3396. + 6200., 0., 60., 100, cnfine, result=result)
    lr = spice.wnfetd(result, 0)
    left = lr[0]
    right = lr[1]
    strapotim = spice.timout(left, TIMFMT, TIMLEN)
    et_apoapse = spice.str2et(strapotim)
    state, ltime = spice.spkezr(target, et_apoapse, frame, abcorr, observer)
    spoint, trgepc, srfvec = spice.subpnt('Intercept: ellipsoid', target, et_apoapse, 'IAU_MARS', abcorr, observer)
    rpoint, colatpoint, lonpoint = spice.recsph(spoint)
    if lonpoint < 0.:
        lonpoint += 2*np.pi
    r_mars = 3396e3
    G = 6.673e-11*6.4273e23
    r = 1e3*state[0:3]
    v = 1e3*state[3:6]
    h = np.cross(r,v)
    n = h/np.linalg.norm(h)
    ev = np.cross(v,h)/G - r/np.linalg.norm(r)
    evn = ev/np.linalg.norm(ev)
    b = np.cross(evn,n)
    vb = np.dot(v,b)
    scx = np.cross(evn,n)

    # get the sub-spacecraft latitude and longitude, and altitude (converted to meters)
    sublat = 90 - np.degrees(colatpoint)
    sublon = np.degrees(lonpoint)
    if sublon > 180:
        sublon -= 360
    alt = np.sqrt(np.sum(srfvec**2))*1e3

    # north pole unit vector in the IAU Mars basis
    polar_vector = [0,0,1]

    # when hovering over the sub-spacecraft point unrotated (the meridian of the point is a straight vertical line,
    # this is the exact view when using cartopy's NearsidePerspective or Orthographic with central_longitude and
    # central latitude set to the sub-spacecraft point), calculate the angle by which the planet must be rotated
    # about the sub-spacecraft point
    angle = np.arctan2(np.dot(polar_vector,-b), np.dot(polar_vector,n))

    # first, rotate the pole to a different latitude given the subspacecraft latitude
    # cartopy's RotatedPole uses the location of the dateline (-180) as the lon_0 coordinate of the north pole
    pole_lat = 90+sublat
    pole_lon = -180

    # convert pole latitude to colatitude (for spherical coordinates)
    # also convert to radians for use with numpy trigonometric functions
    phi = pole_lon*np.pi/180
    theta = (90-pole_lat)*np.pi/180

    # calculate the Cartesian vector pointing to the pole
    polar_vector = [np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)]

    # by rotating the pole, the observer's sub-point in cartopy's un-transformed coordinates is (0,0)
    # the rotation axis is therefore the x-axis
    rotation_axis = [1,0,0]

    # rotate the polar vector by the calculated angle
    rotated_polar_vector = np.dot(rotation_matrix(rotation_axis, -angle), polar_vector)

    # get the new polar latitude and longitude after the rotation, with longitude offset to dateline
    rotated_polar_lon = np.arctan(rotated_polar_vector[1]/rotated_polar_vector[0])*180/np.pi - 180
    rotated_polar_lat = 90 - np.arccos(rotated_polar_vector[2]/np.linalg.norm(rotated_polar_vector))*180/np.pi

    # calculate a RotatedPole transform for the rotated pole position
    transform = ccrs.RotatedPole(pole_latitude=rotated_polar_lat, pole_longitude=rotated_polar_lon, central_rotated_longitude=0)

    # transform the viewer (0,0) point
    tcoords = transform.transform_point(0, 0, ccrs.PlateCarree())

    # find the angle by which the planet needs to be rotated about it's rotated polar axis and calculate a new
    # RotatedPole transform including this angle rotation
    rot_ang = sublon-tcoords[0]
    transform = ccrs.RotatedPole(pole_latitude=rotated_polar_lat, pole_longitude=rotated_polar_lon, central_rotated_longitude=rot_ang)

    # make a cartopy globe with the radius of Mars and a NearsidePerspective projection, centered above (0,0), with
    # a viewer altitude at the spacecraft's altitude
    R_Mars = 3.3895e6 #[m]
    globe = ccrs.Globe(semimajor_axis=R_Mars, semiminor_axis=R_Mars)
    projection = ccrs.NearsidePerspective(central_latitude=0, central_longitude=0, satellite_height=alt, globe=globe)

    # make sure the original axis is equal-aspect
    ax.set_aspect('equal')

    # make a new axis on top of the one with the green data grid
    ax1 = plt.axes([0,0,1,1], projection=projection)
    corner_pos = (1-R_Mars/4e6)/2
    bbox = [corner_pos, corner_pos, 1-2*corner_pos, 1-2*corner_pos]
    ax1.set_axes_locator(InsetPosition(ax, bbox))

    # turn off the circular outline of the projection and the opaque background
    ax1.patch.set_visible(False)
    ax1.outline_patch.set_visible(False)
    ax1.background_patch.set_visible(False)

    # draw and label meridians and parallels
    mandp(ax1, transform, projection)


def apoapse_globe_quicklook(orbit_number):

    # make a figure
    fig = plt.figure(figsize=(5,5))

    # make axes to hold the pixel data grid and turn off the box, axis labels, etc.
    ax = plt.axes([0,0,1,1])
    ax.axis('off')

    # calculate the pixel data grid
    x, y, z = pixel_globe(orbit_number, valid=False)

    # display the pixel data grid using the colormap
    img = ax.pcolormesh(x, y, z, cmap=NO_cmap('black'), norm=colors.PowerNorm(gamma=1/2, vmin=0, vmax=1.4)) ## gamma=1: Linear scale, gamma=1/2: squrt(I)
    #cb = ax.colorbar()

    # draw and the meridian/parallel grid
    draw_grid(ax, orbit_number)

    # show the figure

    #plt.savefig('apoapse_globe_orbit%.5d.png' %(orbit_number), dpi=300)
    plt.show()

apoapse_globe_quicklook(3800)
