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

def primary_is_nan(hdul):
    prim = get_primary(hdul)
    boolens = np.isnan(prim)
    judge = boolens.all()
    return judge

def echelle_place_ok(hdul):
    pos_ok = 'unknown'
    if not primary_is_nan(hdul):
        dimg = get_counts_detector_img(hdul)
        dimg_mean = np.nanmean(dimg, axis=0)
        wv = get_wv(hdul)
        wv_mean = np.nanmean(wv,axis=0)
        idx_lya = np.where((wv_mean>=120)&(wv_mean<=123))
        lya = np.nansum(dimg_mean[idx_lya])
        idx_128 = np.where((wv_mean>=127)&(wv_mean<=129))
        bg_128 = np.nansum(dimg_mean[idx_128])
        r_lya_128 = lya/bg_128

        if lya > 1e3 and r_lya_128 > 10:
            pos_ok = True
        else:
            pos_ok = False
        return pos_ok
