import numpy as np
# data
def get_primary(hdul):
    primary = hdul['primary'].data
    return primary

def get_counts(hdul):
    counts = hdul['Detector_Dark_Subtracted'].data
    return counts

def get_wv(hdul):
    'return wavelength in nm, dim = (n_space, n_spectral)'
    wv = hdul['Observation'].data['Wavelength'][0]
    return wv


def get_primary_detector_img(hdul, mean=True, i_integration=None):
    if mean:
        img = np.nanmean(hdul['primary'].data, axis=0)
    else:
        img = hdul['primary'].data[i_integration]
    return img

def get_counts_detector_img(hdul, mean=True, i_integration=None):
    if mean:
        img = np.nanmean(hdul['detector_dark_subtracted'].data, axis=0)
    else:
        img = hdul['detector_dark_subtracted'].data[i_integration]
    return img
