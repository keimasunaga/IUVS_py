import numpy as np

## (sc)geometry
def get_sc_lat(hdul):
    lat = hdul['SpacecraftGeometry'].data['Sub_Spacecraft_Lat']
    return lat

def get_sc_lon(hdul):
    lon = hdul['SpacecraftGeometry'].data['Sub_Spacecraft_Lon']
    return lon

def get_sc_pos(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        pos = hdul['SpacecraftGeometry'].data['V_spacecraft']
    elif frame == 'MAVEN_MSO':
        pos = hdul['SpacecraftGeometry'].data['V_spacecraft_MSO']
    elif frame == 'Inertial':
        pos = hdul['SpacecraftGeometry'].data['V_spacecraft_Inertial']
    return pos

def get_sc_vel(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        vel = hdul['SpacecraftGeometry'].data['V_spacecraft_Rate']
    elif frame == 'MAVEN_MSO':
        vel = hdul['SpacecraftGeometry'].data['V_spacecraft_Rate_MSO']
    elif frame == 'MAVEN_MSO':
        vel = hdul['SpacecraftGeometry'].data['V_spacecraft_Rate_Inertial']
    return vel

#def pos2sza(pos):
#    sza = np.cos(pos[0]/np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2))
#    return sza

def get_sc_sza(hdul, unit='degree'):
    pos = get_sc_pos(hdul)
    if unit == 'radian':
        sza = np.arccos(pos[:,0]/np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2))
    else:
        sza = np.rad2deg(np.arccos(pos[:,0]/np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)))
    return sza

def get_sc_att_x(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        att_x = hdul['SpacecraftGeometry'].data['Vx_spacecraft']
    elif frame == 'MAVEN_MSO':
        att_x = hdul['SpacecraftGeometry'].data['Vx_spacecraft_MSO']
    elif frame == 'Inertial':
        att_x = hdul['SpacecraftGeometry'].data['Vx_spacecraft_Inertial']
    return att_x

def get_sc_att_y(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        att_y = hdul['SpacecraftGeometry'].data['Vy_spacecraft']
    elif frame == 'MAVEN_MSO':
        att_y = hdul['SpacecraftGeometry'].data['Vy_spacecraft_MSO']
    elif frame == 'Inertial':
        att_y = hdul['SpacecraftGeometry'].data['Vy_spacecraft_Inertial']
    return att_y

def get_sc_att_z(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        att_z = hdul['SpacecraftGeometry'].data['Vz_spacecraft']
    elif frame == 'MAVEN_MSO':
        att_z = hdul['SpacecraftGeometry'].data['Vz_spacecraft_MSO']
    elif frame == 'Inertial':
        att_z = hdul['SpacecraftGeometry'].data['Vz_spacecraft_Inertial']
    return att_z

def get_inst_dir_x(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        dir_x = hdul['SpacecraftGeometry'].data['Vx_instrument']
    elif frame == 'MAVEN_MSO':
        dir_x = hdul['SpacecraftGeometry'].data['Vx_instrument_MSO']
    elif frame == 'Inertial':
        dir_x = hdul['SpacecraftGeometry'].data['Vx_instrument_Inertial']
    return dir_x

def get_inst_dir_y(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        dir_x = hdul['SpacecraftGeometry'].data['Vy_instrument']
    elif frame == 'MAVEN_MSO':
        dir_x = hdul['SpacecraftGeometry'].data['Vy_instrument_MSO']
    elif frame == 'Inertial':
        dir_x = hdul['SpacecraftGeometry'].data['Vy_instrument_Inertial']
    return dir_y

def get_inst_dir_z(hdul, frame='MAVEN_MSO'):
    if frame == 'IAU_MARS':
        dir_z = hdul['SpacecraftGeometry'].data['Vz_instrument']
    elif frame == 'MAVEN_MSO':
        dir_z = hdul['SpacecraftGeometry'].data['Vz_instrument_MSO']
    elif frame == 'Inertial':
        dir_z = hdul['SpacecraftGeometry'].data['Vz_instrument_Inertial']
    return dir_z

#def get_rot_mat(hdul):
#    pass

## (pixel)geometry
def get_pix_lat(hdul, ind=4):
    lat = hdul['PixelGeometry'].data['Pixel_Corner_Lat'][ind, :]
    return lat

def get_pix_lon(hdul, ind=4):
    lon = hdul['PixelGeometry'].data['Pixel_Corner_Lon'][ind, :]
    return lon

def get_pix_alt(hdul, ind=4):
    alt = hdul['PixelGeometry'].data['Pixel_Corner_Mrh_Alt'][ind, :]
    return alt

def get_los_length(hdul, ind=4):
    length = hdul['PixelGeometry'].data['Pixel_Corner_LOS'][ind, :]
    return length

def get_pix_sza(hdul):
    sza = hdul['PixelGeometry'].data['Pixel_Solar_Zenith_Angle']
    return sza

def get_pix_ems_agl(hdul):
    ems_angl = hdul['PixelGeometry'].data['Pixel_Emission_Angle']
    return ems_angl

def get_pix_zenith_agl(hdul):
    zenith_angl = hdul['PixelGeometry'].data['Pixel_Zenith_Angle']
    return zenith_angl

def get_pix_phase_agl(hdul):
    phase_angl = hdul['PixelGeometry'].data['Pixel_Phase_Angle']
    return phase_angl

def get_pix_lt(hdul):
    lt = hdul['PixelGeometry'].data['Pixel_Local_Time']
    return lt
