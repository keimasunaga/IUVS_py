import numpy as np
from glob import glob
import pandas as pd
from datetime import datetime, timezone

def get_df_orbfiles():
    datapath = '/Volumes/Fenix/work/data/misc/spice/naif/MAVEN/kernels/spk/'
    f = glob(datapath+'maven_orb_rec_*.orb')

    keys = ['orbit_number', 'utc_peri', 'Event_SCLK_PERI',  'utc_apo', 'SolLon',  'SolLat', 'SCLon', 'SCLat',  'Alt', 'SolDist']
    farr = np.array([])
    for i, ifile in enumerate(f):
        if i==0:
            df = pd.read_fwf(ifile)
            df = df.drop(df.index[0])
        else:
            df2 = pd.read_fwf(ifile)
            df2 = df2.drop(df2.index[0])
            df = df.append(df2)

    df.columns = keys
    df = df.reset_index(drop=True)
    convert_dict = {'orbit_number': int, 'SolLon': float, 'SolLat':float, 'SCLon':float, 'SCLat':float,  'Alt':float, 'SolDist':float}
    df = df.astype(convert_dict)
    return df

def get_dic_orbfiles():
    df = get_df_orbfiles()
    dic = df.to_dict(orient='list')
    return dic

def get_Dt_peri(orbit_number):
    df = get_df_orbfiles()
    df_selec = df[df['orbit_number'] == orbit_number]
    if len(df_selec) == 0:
        print('---- Seems no orbit info in spk kernel ----')
        return None
    else:
        Dt_peri = pd.to_datetime(df_selec['utc_peri']).tolist()[0].to_pydatetime()
        return Dt_peri

def get_Dt_apo(orbit_number):
    df = get_df_orbfiles()
    df_selec = df[df['orbit_number'] == orbit_number]
    if len(df_selec) == 0:
        print('---- Seems no orbit info in spk kernel ----')
        return None
    else:
        Dt_apo = pd.to_datetime(df_selec['utc_apo']).tolist()[0].to_pydatetime()
        return Dt_apo

def get_Dtlim(orbit_number, center='periapse'):
    if center == 'periapse':
        df = get_df_orbfiles()
        df_selec = df[df['orbit_number'] == orbit_number]
        df_selec_pre = df[df['orbit_number'] == orbit_number - 1]
        eDt = pd.to_datetime(df_selec['utc_apo']).tolist()[0].to_pydatetime()
        sDt = pd.to_datetime(df_selec_pre['utc_apo']).tolist()[0].to_pydatetime()
    elif center == 'apoapse':
        df = get_df_orbfiles()
        df_selec = df[df['orbit_number'] == orbit_number]
        df_selec_lat = df[df['orbit_number'] == orbit_number + 1]
        sDt = pd.to_datetime(df_selec['utc_peri']).tolist()[0].to_pydatetime()
        eDt = pd.to_datetime(df_selec_lat['utc_peri']).tolist()[0].to_pydatetime()
    return [sDt, eDt]

def get_orbit_number(Dt):
    dic = get_dic_orbfiles()
    orbit_number_all = np.array(dic['orbit_number'])
    Dt_peri = np.array([datetime.strptime(istr, '%Y %b %d %H:%M:%S') for istr in dic['utc_peri']]) #add tzinfo here too?
    Dt_apo = np.array([datetime.strptime(istr, '%Y %b %d %H:%M:%S') for istr in dic['utc_apo']]) #add tzinfo here too?
    idx_orb = np.where((Dt >= Dt_peri[0:-1]) & (Dt < Dt_peri[1:]))[0][0]
    orbit_number = orbit_number_all[idx_orb]
    return orbit_number

def test():
    df = get_df_orbfiles()
    dic = get_dic_orbfiles()
    sDt, eDt = get_Dtlim(650, 'periapse')
    orb = get_orbit_number(eDt)
    Dt_peri = get_Dt_peri(650)
    Dt_apo = get_Dt_apo(650)
    print(sDt, eDt)
    print(Dt_peri, Dt_apo)
    print(orb)


class OrbitInfo:
    def __init__(self):
        datapath = '/Volumes/Fenix/work/data/misc/spice/naif/MAVEN/kernels/spk/'
        f = glob(datapath+'maven_orb_rec_*.orb')

        keys = ['orbit_number', 'utc_peri', 'Event_SCLK_PERI',  'utc_apo', 'SolLon',  'SolLat', 'SCLon', 'SCLat',  'Alt', 'SolDist']
        farr = np.array([])
        for i, ifile in enumerate(f):
            if i==0:
                df = pd.read_fwf(ifile)
                df = df.drop(df.index[0])
            else:
                df2 = pd.read_fwf(ifile)
                df2 = df2.drop(df2.index[0])
                df = df.append(df2)

        df.columns = keys
        df = df.reset_index(drop=True)
        convert_dict = {'orbit_number': int, 'SolLon': float, 'SolLat':float, 'SCLon':float, 'SCLat':float,  'Alt':float, 'SolDist':float}
        self.df = df.astype(convert_dict)

    def get_dic_orbfiles(self):
        dic = self.df.to_dict(orient='list')
        return dic

    def get_Dt_peri(self, orbit_number):
        return get_Dt_peri(orbit_number)

    def get_Dt_apo(self, orbit_number):
        return get_Dt_apo(orbit_number)

    def get_Dtlim(self, orbit_number, center='periapse'):
        if center == 'periapse':
            df_selec = self.df[self.df['orbit_number'] == orbit_number]
            df_selec_pre = self.df[self.df['orbit_number'] == orbit_number - 1]
            eDt = pd.to_datetime(df_selec['utc_apo']).tolist()[0].to_pydatetime()
            sDt = pd.to_datetime(df_selec_pre['utc_apo']).tolist()[0].to_pydatetime()
        elif center == 'apoapse':
            df_selec = self.df[self.df['orbit_number'] == orbit_number]
            df_selec_lat = self.df[self.df['orbit_number'] == orbit_number + 1]
            sDt = pd.to_datetime(df_selec['utc_peri']).tolist()[0].to_pydatetime()
            eDt = pd.to_datetime(df_selec_lat['utc_peri']).tolist()[0].to_pydatetime()
        return [sDt, eDt]

    def get_orbit_number(self, Dt):
        dic = self.get_dic_orbfiles()
        orbit_number_all = np.array(dic['orbit_number'])
        Dt_peri = np.array([datetime.strptime(istr, '%Y %b %d %H:%M:%S') for istr in dic['utc_peri']]) #add tzinfo here too?
        Dt_apo = np.array([datetime.strptime(istr, '%Y %b %d %H:%M:%S') for istr in dic['utc_apo']]) #add tzinfo here too?
        idx_orb = np.where((Dt >= Dt_peri[0:-1]) & (Dt < Dt_peri[1:]))[0][0]
        orbit_number = orbit_number_all[idx_orb]
        return orbit_number
