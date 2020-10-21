import numpy as np

from variables import saveloc

def get_pa_peri_list():
    fname = saveloc + 'misc_items/proton_aurora_list/PA_data_npz.npz'
    npz = np.load(fname)
    dic = dict([(iname, npz[iname]) for iname in npz.files])
    return dic

def detect_pa_neighbor(orbit_number_apo):
    dic = get_pa_peri_list()
    orbit_pa_peri = dic['pa_orbitarray']
    result = [iorbit in orbit_pa_peri for iorbit in (orbit_number, orbit_number + 1)]
    return result


class PAListPeri:
    def __init__(self):
        self.dic = get_pa_peri_list()

    def neighbor_detected(self, orbit_number_apo):
        orbit_pa_peri = self.dic['pa_orbitarray']
        result = [iorbit in orbit_pa_peri for iorbit in (orbit_number_apo, orbit_number_apo + 1)]
        return result
