import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

from variables import saveloc
from PyUVS.geometry import beta_flip
from PyUVS.graphics import H_colormap
from iuvdata_l1b import ApoapseInfo, ApoapseSwath
from common.tools import RunTime

class GlobeData:
    def __init__(self, xlength=8000, ylength=8000, pixres=100):
        # Bin setting
        self.xlength = xlength
        self.ylength = ylength
        self.pixres = pixres #20 #[km/pixel]
        self.xsize = int(self.xlength/self.pixres)
        self.ysize = int(self.ylength/self.pixres)
        # Empty bins to be filled with data
        self.databin = np.zeros((self.ysize, self.xsize))
        self.ndat = np.zeros((self.ysize, self.xsize))
        self.flip = None

    def get_primary_dims(self, hdul):
        # determine dimensions, and if it's a single integration, skip it
        dims = hdul['primary'].shape
        if len(dims) == 3:
            n_int = dims[0]
            n_spa = dims[1]
            return n_int, n_spa
        else:
            print('No data or single integration, load skipped')
            return None, None

    #def dayside(self, hdul):
    #    if hdul['observation'].data['mcp_volt'] < 700:
    #        return True
    #    else:
    #        return False

    def mesh_data(self, hdul):

        if self.flip is None:
            self.flip = beta_flip(hdul)

        n_int, n_spa = self.get_primary_dims(hdul)

        if n_int is not None:

            #if self.dayside(hdul):

            self.flip = beta_flip(hdul)
            aposwath = ApoapseSwath(hdul)
            primary_arr = aposwath.get_img()

            # this is copied directly from Sonal; someday I'll figure it out and comment...
            # essentially it finds the place where the pixel position vector intersects the 400x400 grid
            # and places the pixel value in that location
            for i in range(n_int):
                vspc = hdul['spacecraftgeometry'].data[i]['v_spacecraft']
                vspcnorm = vspc/np.linalg.norm(vspc)
                vy = hdul['spacecraftgeometry'].data[i]['vy_instrument']
                vx = np.cross(vy, vspcnorm)

                for j in range(n_spa):
                    primary = primary_arr[i,j]

                    for m in range(4):
                        try:
                            vpix = hdul['pixelgeometry'].data[i]['pixel_vec']
                            vpixcorner = (np.squeeze(vpix[:,j,m]) + np.squeeze(vpix[:,j,4]))/2
                            vdiff = vspc - (np.dot(vspc,vpixcorner)*vpixcorner)
                            x = int(np.dot(vdiff,vx)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /self.pixres+self.xsize/2)
                            y = int(np.dot(vdiff,vy)*np.linalg.norm(vdiff) / np.linalg.norm([np.dot(vdiff,vx),np.dot(vdiff,vy)]) /self.pixres+self.ysize/2)
                            if (x >= 0) & (y >= 0):
                                    self.databin[y,x] += primary
                                    self.ndat[y,x] += 1
                        except:
                            continue

    def get_xygrids(self):
        #make coordinate grids for each pixel in kilometers
        x, y = np.meshgrid(np.linspace(-self.xsize/2*self.pixres, self.xsize/2*self.pixres, self.xsize), np.linspace(-self.ysize/2*self.pixres, self.ysize/2*self.pixres, self.ysize))

        #return the coordinate grids and the spherically-projected data pixels
        return x, y

    def get_data(self):
        #calculate the average
        z = self.databin/self.ndat
        # beta-flip if necessary
        if self.flip == True:
            z = np.flip(z)
        return z

    def plot(self, ax=None, **kwargs):
        x, y = self.get_xygrids()
        z = self.get_data()
        if ax is None:
            mesh = plt.pcolormesh(x, y, z, **kwargs)
        else:
            mesh = ax.pcolormesh(x, y, z, **kwargs)
        return mesh

    def save_data(self, savepath):
        x, y = self.get_xygrids()
        z = self.get_data()
        dic = {'x':x, 'y':y, 'z':z}
        np.save(savepath, dic)

def quicklook_apoapse_globe(orbit_number):
    apoinfo = ApoapseInfo(orbit_number)
    glb = GlobeData()
    for ith_file, iswath_number in enumerate(apoinfo.swath_number):
        hdul = apoinfo.get_hdul(ith_file)
        glb.mesh_data(hdul) ## add hdul in a mesh

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    mesh = glb.plot(ax=ax, cmap=H_colormap(), norm=mpl.colors.PowerNorm(gamma=1/2, vmin=0, vmax=30))
    ax.set_xlabel('[km]')
    ax.set_ylabel('[km]')
    ax.set_aspect(1)
    ax.set_title('Orbit ' + str(orbit_number))
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(mesh, cax=cax)
    cb.set_label('Brightness [kR]')

    # save figure
    pngpath = saveloc + 'quicklook/apoapse_l1b/Lyman-alpha/globe/orbit_' + '{:05d}'.format(orbit_number//100 * 100) + '/'
    fname_save = 'orbit_' + '{:05d}'.format(orbit_number)
    if not os.path.exists(pngpath):
        os.makedirs(pngpath)
    plt.savefig(pngpath + fname_save)

    # save data
    savepath = pngpath + 'npy/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    glb.save_data(savepath + fname_save)

if __name__ == '__main__':
    rt = RunTime()
    rt.start()
    start_orbit = int(sys.argv[1])
    n_orbit = int(sys.argv[2])
    orbit_arr = np.arange(n_orbit) + start_orbit
    for iorbit_number in np.arange(n_orbit) + start_orbit:
        quicklook_apoapse_globe(iorbit_number)
        plt.close()
    rt.stop()
