from scipy.io import readsav
import matplotlib.pyplot as plt
import numpy as np
import os

from common.tools import shift_grids
from variables import saveloc

def get_bfield_map_sav(comp='br', altbin=0):
    fname = '/Users/masunaga/work/save_data/maven/sav/bfield_map_Tristan/' + comp + '.sav'
    sav = readsav(fname)
    barr = sav[comp][altbin]
    return barr

def plot_bfield_map(comp='br', altbin=0):
    alt = 150 * (altbin+1)
    bmap = get_bfield_map_sav(comp, altbin)
    lat = np.arange(180) - 90
    lon = np.arange(360)
    lat_shifted, lon_shifted = shift_grids(lat, lon)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(lon_shifted, lat_shifted, bmap, cmap='coolwarm', vmin=-100, vmax=100)
    cb = plt.colorbar(mesh, ax=ax)
    cb.set_label(comp + ' [nT]')
    ax.set_title('Alt = ' + str(alt) + ' km')
    ax.set_xlabel('East longitude')
    ax.set_ylabel('Latitude')
    path = saveloc + 'misc_items/bfield_map/'
    os.makedirs(path, exist_ok=True)
    fname = comp + '_with_labels.jpg'
    plt.savefig(path + fname)

def plot_bfield_map_no_labels(comp='br', altbin=0):
    alt = 150 * (altbin+1)
    bmap = get_bfield_map_sav(comp, altbin)
    lat = np.arange(180) - 90
    lon = np.arange(360)
    lat_shifted, lon_shifted = shift_grids(lat, lon)
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(lon_shifted, lat_shifted, bmap, cmap='coolwarm', vmin=-100, vmax=100)
    plt.xticks([])
    plt.yticks([])
    ax.axis('off')
    path = saveloc + 'misc_items/bfield_map/'
    os.makedirs(path, exist_ok=True)
    fname = comp + '_no_labels.jpg'
    plt.savefig(path + fname)

def test_plot():
    plot_bfield_map_no_labels('br',0)
#test_plot()
