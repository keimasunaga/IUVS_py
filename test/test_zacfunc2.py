import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import math

import iuvtools  ## functions_for_kei.py was renamed as iuvtools.py
import spicetools

# furnish SPICE kernels
spicetools.load_iuvs_spice()
from importlib import reload
reload(iuvtools)
fnames = iuvtools.get_files(6350)

#for i in range(len(fnames)):
#print(i)
hdul = fits.open(fnames[0])
#X, Y = iuvtools.angle_meshgrid(hdul)   ##
lat, lon, sza, lt, x, y, cx, cy, context_map = iuvtools.highres_swath_geometry(hdul)
context_map_colors = context_map.reshape(context_map.shape[0] * context_map.shape[1], context_map.shape[2])
y = (120 - y) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
cy = (120 - cy) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
plt.pcolormesh(x, y, np.ones_like(x), color=context_map_colors, linewidth=0, edgecolors='none', rasterized=True).set_array(None)
iuvtools.latlon_grid(cx, cy, lat, lon, plt.gca())
#import pdb; pdb.set_trace()
plt.show()


'''
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


# initialize night mapping arrays and variables
xsize_night = 250
ysize_night = 250
pixel_size = 50
count_night = np.zeros((ysize_night, xsize_night))
total_night = np.zeros((ysize_night, xsize_night))
apoapse_altitude = 6200e3

fnames = iuvtools.get_files(6350)
n_fnames = len(fnames)


# loop through nightside files
for i in range(n_fnames):

    # open FITS file
    hdul = fits.open(fnames[i])
    beta_flipped = iuvtools.beta_flip(hdul)
    print(beta_flipped)
    # get day and night binning
    n_integrations = hdul['integration'].data.shape[0]
    n_spatial = len(hdul['binning'].data['spapixlo'][0])

    # make arrays and get version information from first file
    if i == 0:

        # calculate pixel size in km
        pixel_vec = hdul['pixelgeometry'].data[0]['pixel_vec']
        pixel_angle = np.arccos(np.dot(pixel_vec[:, 0, 0], pixel_vec[:, 0, 1]))
        if np.isnan(pixel_angle):
            pixel_angle = np.radians(slit_width_deg/n_spatial)
        pixel_size = 2 * np.tan(pixel_angle) * (apoapse_altitude / 1e3)

        # dimensions of pixel grid and width of a pixel in kilometers
        xsize_night = int(8000 / pixel_size)
        ysize_night = int(8000 / pixel_size)

        # arrays to hold projected data (total and count for averaging after every data point placed)
        total_night = np.zeros((ysize_night, xsize_night))
        count_night = np.zeros((ysize_night, xsize_night))

    # make an array to hold integrated brightnesses
    mlr_array = hdul['primary'].data#MLR(hdul)#nightside_pixels(hdul, feature='NO')

    # calculate pixel position at apoapsis projected to plane through center of Mars
    for j in range(n_integrations):

        # get vectors and calculate some stuff...
        vspc = hdul['spacecraftgeometry'].data[j]['v_spacecraft']
        vspcnorm = vspc / np.linalg.norm(vspc)
        vy = hdul['spacecraftgeometry'].data[j]['vy_instrument']
        vx = np.cross(vy, vspcnorm)

        # loop through spatial elements
        for k in range(n_spatial):

            # calculate horizontal and vertical positions
            for m in range(5):

                try:
                    vpixcorner = hdul['pixelgeometry'].data[j]['pixel_vec'][:, k, m]
                    vdiff = vspc - (np.dot(vspc, vpixcorner) * vpixcorner)
                    x = int(np.dot(vdiff, vx) * np.linalg.norm(vdiff) /
                            np.linalg.norm(
                                [np.dot(vdiff, vx), np.dot(vdiff, vy)]) / pixel_size + xsize_night / 2)
                    y = int(np.dot(vdiff, vy) * np.linalg.norm(vdiff) /
                            np.linalg.norm(
                                [np.dot(vdiff, vx), np.dot(vdiff, vy)]) / pixel_size + ysize_night / 2)

                    # make sure they fall within the grid...
                    if (x >= 0) and (x < 8000) and (y >= 0) and (y < 8000):
                        # put the value in the grid
                        total_night[y, x] += mlr_array[j, k]
                        count_night[y, x] += 1
                except (IndexError, ValueError):
                    pass

# calculate the average of the nightside grid
total_night[np.where(count_night == 0)] = np.nan
night_grid = total_night / count_night
#import pdb; pdb.set_trace()
# rotate if not beta-flipped
if beta_flipped:
    night_grid = np.rot90(night_grid, k=2, axes=(0, 1))
#import pdb; pdb.set_trace()
# meshgrids for data dislpay
x_night, y_night = np.meshgrid(np.linspace(-4000, 4000, xsize_night), np.linspace(-4000, 4000, ysize_night))

# place the nightside grid
plt.pcolormesh(x_night, y_night, night_grid)
#projection_ax.pcolormesh(x_night, y_night, night_grid, cmap=NO_colormap(),
#                         norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=10))
'''
