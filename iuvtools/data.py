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
