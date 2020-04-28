import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import math

import iuvtools

def beta_flip(vi, vs):

   """
   Determine the spacecraft orientation and see if it underwent a beta flip. This compares the instrument
   x-axis direction to the spacecraft velocity direction, which are either (nearly) parallel or anti-parallel.

   Parameters
   ----------
   hdul : object
       Opened FITS file.

   Returns
   -------
   beta : bool
       Beta flipped? Yes or no, true or false...

   """

   #vi = hdul['spacecraftgeometry'].data['vx_instrument_inertial'][-1]
   #vs = hdul['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][-1]

   # determine orientation between vectors
   app_sig = np.sign(np.dot(vi, vs))

   # if negative, then no beta flipping; if positive, then yes beta flipping
   if app_sig == -1:
       beta = False
   elif app_sig == 1:
       beta = True

   # return the bool
   return beta

def swath_geometry(files):

   """
   Determine how many swaths taken during a MAVEN/IUVS apoapse disk scan.

   Parameters
   ----------
   orbit_number : int
       The orbit number you want to query.

   Returns
   -------
   n_swaths : int
       The number of swaths for orbit_number.
   dayside_swath_dimensions : array
       The width (spatial dimension) and height (integration dimension) of the dayside swath images in pixels.
   nightside_swath_dimensions : array
       Same as above but for nightside data.

   """

   # get list of FITS files for given orbit number
   ##files, n_files = get_files(orbit_number)

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
       import pdb; pdb.set_trace()
       # check for and skip single integrations
       if hdul[0].data.ndim == 2:
           continue

       # and if not...
       else:

           # determine beta-flip
           vi = hdul['spacecraftgeometry'].data['vx_instrument_inertial'][-1]
           vs = hdul['spacecraftgeometry'].data['v_spacecraft_rate_inertial'][-1]
           flipped = beta_flip(vi,vs)

           # store filepath
           filepaths.append(files[i])

           # determine if dayside or nightside
           if hdul['observation'].data['mcp_volt'] > 790:
               dayside = False
               daynight.append(0)
           else:
               dayside = True
               daynight.append(1)

           # extract integration extension
           integration = hdul['integration'].data

           # calcualte mirror direction
           mirror_dir = np.sign(integration['mirror_deg'][-1]-integration['mirror_deg'][0])

           if prev_ang == 999:
               prev_ang *= mirror_dir

           # check the angles by seeing if the mirror is still scanning in the same direction
           ang0 = integration['mirror_deg'][0]
           if ((mirror_dir == 1) & (prev_ang > ang0)) | ((mirror_dir == -1) & (prev_ang < ang0)):

               # increment the swath count
               n_swaths += 1

           # store swath number
           swath.append(n_swaths-1)

           # change the previous angle comparison value
           prev_ang = integration['mirror_deg'][-1]

   # make a dictionary to hold all this shit
   swath_info = {
       'filepaths':np.array(filepaths),
       'n_swaths':n_swaths,
       'swath_number':np.array(swath),
       'dayside':np.array(daynight),
       'beta_flip':flipped
   }

   return swath_info






class IuvFile():

    def __init__(self):
        lv = None
        mode = None
        orbit = None
        uvch = None
        date = None
        vr = None
        rl = None

    def get_path(self):
        orbflr = math.floor(self.orbit/100)*100
        path = '/Volumes/Gravity/work/data/maven_iuvs/' + \
               'level' + self.lv[1:] + '/orbit' + str(orbflr).zfill(5) + '/'
        pattern = 'mvn_iuv_' + self.lv + '_' + self.mode + '-' + \
                  'orbit' + str(self.orbit).zfill(5) + '-' + self.uvch + '_' + \
                  '*.fits'
        filepaths = glob.glob(path + pattern)
        return filepaths

class IuvData():

    def __init__(self, filepath):
        self.fpath = filepath ## Should be a single file path
        self.fname = self.fpath.split('/')[-1]
        fnrpl = self.fname.replace('-', '_')
        fnspl = fnrpl.split('_')
        self.lv = fnrpl[2]
        self.mode = fnrpl[3]
        self.orbit = int(fnspl[4][5:])
        self.uvch = fnspl[5]
        self.date = fnspl[6][0:8]
        self.hh = fnspl[6][9:11]
        self.mm = fnspl[6][11:13]
        self.ss = fnspl[6][13:15]

    def open(self):
        self.hdul = fits.open(self.fpath)

    def get_nextend(self):
        return len(self.hdul)

    def close(self):
        self.hdul.close()


"""
def get_filename(orbit, mode='apoapse', uvch='fuv', lv='l2b', date='*', vr='*', rl='*'):
    path = '/Volumes/Gravity/work/data/maven_iuvs/' + \
           'level' + lv[1:] + '/orbit' + str(orbit).zfill(5)
    pattern = 'mvn_iuv_' + lv + '_' + mode + '-' + \
              'orbit' + str(orbit).zfill(5) + '-' + uvch + '_' + \
              date + 'T' + '*.fits'
    filepath = glob.glob(path + '/' + pattern)
    if np.size(filepath) == 1:
        fname = filepath[0].split('/')[-1]
    else:
        fname = [ifname.split('/')[-1] for ifname in filepath]
    return fname
"""
#fname = 'mvn_iuv_l2b_apoapse-orbit03799-fuv_20160910T151814_v13_r02.fits'

fnames = iuvtools.get_files(7008)
hdul = fits.open(fnames[0])
X, Y = iuvtools.angle_meshgrid(hdul)
_, _, _, _, X, Y, cX, cY, context_map = iuvtools.highres_swath_geometry(hdul)
context_map_colors = context_map.reshape(context_map.shape[0] * context_map.shape[1], context_map.shape[2])
Y = (120 - Y) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
#cY = (120 - cY) + 60  # reverse Y array so scan goes top-to-bottom instead of bottom-to-top
plt.pcolormesh(X, Y, np.ones_like(X), color=context_map_colors, linewidth=0, edgecolors='none', rasterized=True).set_array(None)
plt.show()
import pdb; pdb.set_trace()
## Define file information and get file paths
iuvfile = IuvFile()
iuvfile.lv = 'l1b'
iuvfile.mode = 'apoapse'
iuvfile.orbit = 7008#3780#6321#3799
iuvfile.uvch = 'fuv'
filepaths = iuvfile.get_path()
ndat = len(filepaths)
ncol  = 2
nrow = int(ndat/ncol)
print(filepaths)
iuvdat = IuvData(filepaths)
iuvdat.open()
x, y, z = iuvtools.angle_meshgrid(iuvdat.hdul)

import pdb; pdb.set_trace()
#iuvdat = IuvData(filepaths[0])
#iuvdat.open()
#hdul = iuvdat.hdul
#import pdb; pdb.set_trace()
#iuvdat.close()
#geom = iuvdat.hdul['PixelGeometry']
## Create figure and axis objects
fig, ax = plt.subplots(1,1, figsize=[15,8])
cmap = plt.get_cmap('Blues_r')
slit_width = 10.64 #[degree]
img_lst = []
img1304_arr = np.zeros([1, 10])
ltdat_arr = np.zeros([1, 10])

swath_dic = swath_geometry(filepaths)
swath_number = swath_dic['swath_number']
beta_flip = swath_dic['beta_flip']
import pdb; pdb.set_trace()

for i, ipath in enumerate(filepaths):#[::-1]
    iuvdat = IuvData(ipath)
    iuvdat.open()
    #beta = beta_flip(iuvdat.hdul) ## Judge if flippped or not
    imgdat = iuvdat.hdul['primary'].data
    ltdat = iuvdat.hdul['PixelGeometry'].data['pixel_local_time']
    img1304 = imgdat[:,:,1]
    img1304_arr = np.append(img1304_arr, img1304, axis=0)
    ltdat_arr = np.append(ltdat_arr, ltdat, axis=0)
    #img_lst.append(img1304)

    # get angles of observation
    angles = iuvdat.hdul['integration'].data['mirror_deg']*2 #convert from mirror angles to FOV angles
    dang = np.mean(np.diff(angles[:-1]))

    # number of integrations
    dims = iuvdat.hdul['primary'].data.shape
    n_integrations = dims[0]
    n_spatial = dims[1]

    iuvdat.close()

    # make meshgrid of angles for plotting
    if beta_flip:
        X, Y = np.meshgrid(np.flip(np.linspace(slit_width*swath_number[i],slit_width*(swath_number[i]+1),n_spatial+1)),np.linspace(angles[0]-dang/2, angles[-1]+dang/2, n_integrations+1))
    else:
        X, Y = np.meshgrid(np.linspace(slit_width*swath_number[i],slit_width*(swath_number[i]+1),n_spatial+1)),np.linspace(angles[0]-dang/2, angles[-1]+dang/2, n_integrations+1)

    # display image
    try:
        img = ax.pcolormesh(X, Y, img1304, vmin=0, vmax=1.5, cmap=cmap)
    except:
        continue

print(np.max(ltdat_arr))
#print(img1304_arr.shape)
idx = np.where((ltdat_arr >= 11.5) & (ltdat_arr <= 12.5))
print(img1304_arr)
print(np.nanmean(img1304_arr[idx[0]]))
#import pdb; pdb.set_trace()
cbar = plt.colorbar(img)
cbar.set_label('Brightness [kR]', rotation=270, labelpad=15)
ax.set_aspect('equal')
ax.set_title('Brightness of OI 130.4 nm observed on ' + iuvdat.date + ' (orbit #' + str(iuvfile.orbit) + ')')
plt.tight_layout()
plt.show()
