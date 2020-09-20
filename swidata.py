import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import cdflib
import common.tools as ctools

from PyUVS.time import et2datetime
from variables import pfpdataloc
from PyUVS.time import et2datetime


class SwiSpec:
    def __init__(self, cdf):
        self.et = cdf.varget('time_met')
        self.timeDt = np.array([et2datetime(iet) for iet in self.et])
        self._remove_tzinfo()
        self.data = cdf.varget('spectra_diff_en_fluxes')
        self.v = cdf.varget('energy_spectra')

    def _remove_tzinfo(self):
        self.timeDt = np.array([iDt.replace(tzinfo=None) for iDt in self.timeDt])

    def get_energy_time_grids(self):
        mt = np.array([mdates.date2num(iDt) for iDt in self.timeDt])
        shifted_mt = np.r_[mt[0], mt[1:] - np.diff(mt)*0.5, mt[-1]]
        shifted_v = np.r_[self.v[0], self.v[1:] - np.diff(self.v)*0.5, self.v[-1]]
        return shifted_mt, shifted_v

    def plot(self, ax=None, xylabels=True, **kwargs):
        mt, energy = self.get_energy_time_grids()
        mt = np.array([mdates.date2num(iDt) for iDt in self.timeDt])

        if ax is None:
            plt.pcolormesh(mt, energy, self.data.T, **kwargs)
        else:
            ax.pcolormesh(mt, energy, self.data.T, **kwargs)

        plt.gca().set_yscale('log')
        date_format = mdates.DateFormatter('%H:%M:%S')
        plt.gca().xaxis.set_major_formatter(date_format)
        if xylabels:
            plt.gca().set_xlabel('Time')
            plt.gca().set_ylabel('Energy [eV]')
            plt.gca().set_ylim(25, 2.5e4)

    def append(self, other):
        self.timeDt = np.append(self.timeDt, other.timeDt)
        self.data = np.append(self.data, other.data, axis=0)


class SwimDens:
    def __init__(self, cdf):
        self.et = cdf.varget('time_met')
        self.timeDt = np.array([et2datetime(iet) for iet in self.et])
        self._remove_tzinfo()
        self.data = cdf.varget('density')

    def _remove_tzinfo(self):
        self.timeDt = np.array([iDt.replace(tzinfo=None) for iDt in self.timeDt])

    def plot(self, ax=None, xylabels=True, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.data, **kwargs)
        else:
            ax.plot(self.timeDt, self.data, **kwargs)
        if xylabels:
            plt.gca().set_xlabel('Time')
            plt.gca().set_ylabel('Density [cm-3]')

    def append(self, other):
        self.timeDt = np.append(self.timeDt, other.timeDt)
        self.data = np.append(self.data, other.data)

    def get_mean(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        return np.mean(self.data[idx[0]:idx[1]])

    def get_std(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        return np.std(self.data[idx[0]:idx[1]])


class SwimVel:
    def __init__(self, cdf):
        self.et = cdf.varget('time_met')
        self.timeDt = np.array([et2datetime(iet) for iet in self.et])
        self._remove_tzinfo()
        self.data = cdf.varget('velocity_mso')
        self.data_t = np.linalg.norm(self.data, axis=1)

    def _remove_tzinfo(self):
        self.timeDt = np.array([iDt.replace(tzinfo=None) for iDt in self.timeDt])

    def plot_x(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.data[:,0], **kwargs)
        else:
            ax.plot(self.timeDt, self.data[:,0], **kwargs)

    def plot_y(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.data[:,1], **kwargs)
        else:
            ax.plot(self.timeDt, self.data[:,1], **kwargs)

    def plot_z(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.data[:,2], **kwargs)
        else:
            ax.plot(self.timeDt, self.data[:,2], **kwargs)

    def plot_t(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.timeDt, self.data_t, **kwargs)
        else:
            ax.plot(self.timeDt, self.data_t, **kwargs)

    def append(self, other):
        self.timeDt = np.append(self.timeDt, other.timeDt)
        self.data = np.append(self.data, other.data, axis=0)
        self.data_t = np.append(self.data_t, other.data_t)

    def get_mean(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        return np.mean(self.data[idx[0]:idx[1],:], axis=0)

    def get_std(self, Dtrange=None):
        idx = ctools.nnDt(self.timeDt, Dtrange)
        return np.std(self.data[idx[0]:idx[1],:], axis=0)


class SwiInfo:
    def __init__(self, year, month, day, level='l2', dtype='swim'):
        self.year = year
        self.month = month
        self.day = day
        self.level = level
        self.dtype = dtype
        self.dname = self._get_dname()
        #self.version = version
        #self.revision = revision

    def _get_dname(self):
        if self.dtype == 'swim':
            return 'onboardsvymom'

        elif self.dtype == 'swis':
            return 'onboardsvyspec'

        elif self.dtype == 'swica':
            return 'coarsearc3d'

        elif self.dtype == 'swics':
            return 'coarsesvy3d'

        elif self.dtype == 'swifa':
            return 'finearc3d'

        elif self.dtype == 'swifs':
            return 'finesvy3d'

    def get_file(self):
        filepath = os.path.join(pfpdataloc, 'swi', self.level, str(self.year), '{:02d}'.format(self.month))
        filename = 'mvn_swi_' + self.level + '_' + self.dname + '_' + str(self.year) + '{:02d}'.format(self.month) + '{:02d}'.format(self.day) + '_v??_r??.cdf'
        fnames = glob.glob(os.path.join(filepath, filename))
        try:
            return self._find_newest(fnames)
        except:
            print('----- '+ filename + ' not found. -----')

    def _find_newest(self, fnames):
        if len(fnames) > 1:
            ftemp = np.array([ifname.split('/')[-1] for ifname in fnames])
            vtemp = np.array([int((iftemp.split('_')[5])[1:]) for iftemp in ftemp])
            if len(list(set(vtemp))) == 1:
                rtemp = np.array([int((iftemp.split('_')[6])[1:3]) for iftemp in ftemp])
                idx = np.where(rtemp == max(rtemp))[0][0]
                fname = fnames[idx]
            else:
                idx = np.where(vtemp == max(vtemp))[0][0]
                fname = fnames[idx]
        elif len(fnames) == 1:
            fname = fnames[0]
        return fname


def get_swi_obj(sDt, eDt):
    date_st = sDt.strftime('%Y%m%d')
    date_et = eDt.strftime('%Y%m%d')
    if date_st == date_et:
        swiminfo = SwiInfo(sDt.year, sDt.month, sDt.day, level='l2', dtype='swim')
        fswim = swiminfo.get_file()
        swisinfo = SwiInfo(sDt.year, sDt.month, sDt.day, level='l2', dtype='swis')
        fswis = swisinfo.get_file()
        cdf_swim = cdflib.CDF(fswim)
        cdf_swis = cdflib.CDF(fswis)
        swispec = SwiSpec(cdf_swis)
        swidens = SwimDens(cdf_swim)
        swivel = SwimVel(cdf_swim)
    else:
        d = eDt.day - sDt.day
        data_ok = False
        for i in range(d+1):
            iDt = sDt + timedelta(days=i)
            swiminfo = SwiInfo(iDt.year, iDt.month, iDt.day, level='l2', dtype='swim')
            swisinfo = SwiInfo(iDt.year, iDt.month, iDt.day, level='l2', dtype='swis')
            fswim = swiminfo.get_file()
            fswis = swisinfo.get_file()
            cdf_swim = cdflib.CDF(fswim)
            cdf_swis = cdflib.CDF(fswis)
            if data_ok:
                swispec.append(SwiSpec(cdf_swis))
                swidens.append(SwimDens(cdf_swim))
                swivel.append(SwimVel(cdf_swim))
            else:
                swispec = SwiSpec(cdf_swis)
                swidens = SwimDens(cdf_swim)
                swivel = SwimVel(cdf_swim)
                data_ok = True

    return swispec, swidens, swivel


def get_swi_stat(sDt, eDt):
    _, swidens, swivel = get_swi_obj(sDt, eDt)
    dens_mean = swidens.get_mean([sDt, eDt])
    dens_std = swidens.get_std([sDt, eDt])
    vel_mean = swivel.get_mean([sDt, eDt])
    vel_std = swivel.get_std([sDt, eDt])
    dic = {'dens_mean':dens_mean, 'dens_std':dens_std, 'vel_mean':vel_mean, 'vel_std':vel_std}
    return dic


def test():

    ## Fix later: Currently spice has to be furnished before using this module, due to using et2datetime.
    ## et2datetime could be revised as unix2datetime using unixtime data and dt_object = datetime.fromtimestamp(timestamp)
    import PyUVS.spice as Pyspice ##
    Pyspice.load_iuvs_spice()

    fswi_mom = pfpdataloc + 'swi/l2/2019/07/mvn_swi_l2_onboardsvymom_20190707_v01_r01.cdf'
    fswi_mom2 = pfpdataloc + 'swi/l2/2019/07/mvn_swi_l2_onboardsvymom_20190708_v01_r01.cdf'
    fswi_spec = pfpdataloc + 'swi/l2/2019/07/mvn_swi_l2_onboardsvyspec_20190707_v01_r01.cdf'
    fswi_spec2 = pfpdataloc + 'swi/l2/2019/07/mvn_swi_l2_onboardsvyspec_20190708_v01_r01.cdf'
    cdf_spec = cdflib.CDF(fswi_spec)
    cdf_spec2 = cdflib.CDF(fswi_spec2)
    swispec = SwiSpec(cdf_spec)
    swispec2 = SwiSpec(cdf_spec2)
    swispec.append(swispec2)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    swispec.plot(ax, cmap='jet', norm=mpl.colors.LogNorm(vmin=1e4, vmax=1e8))
