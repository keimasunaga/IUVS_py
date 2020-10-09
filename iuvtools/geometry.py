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

def get_pix_vec(hdul):
    pixel_uvec_from_sc = hdul['PixelGeometry'].data['PIXEL_VEC'] ## pixel unit vector from sc in IAU MARS frame
    return pixel_uvec_from_sc

def get_los_length(hdul):
    los_length = hdul['PixelGeometry'].data['PIXEL_CORNER_LOS']
    return los_length


class PixelTransCoord:
    '''
    This class transforms pixel vectors in an apoapse swath
    from default (IAU_MARS) to destination frame (MSO/Inertial).
    usage:
        pixtrans = PixelTransCoord(hdul, to_frame='MSO')
        pixel_vec_from_pla_mso = pixtrans.pixel_vec_from_pla_dest
    '''
    def __init__(self, hdul, from_frame='IAU_MARS', to_frame='MSO'):
        '''
        arg:
            hdul: Open fits file
            from_frame: An original frame
            to_frame: A destination frame
        '''
        self.from_frame = from_frame
        self.to_frame = to_frame
        self.hdul = hdul
        self.pixel_uvec_from_sc = self.hdul['PixelGeometry'].data['PIXEL_VEC'] ## pixel unit vector from sc
        self.los_length = self.hdul['PixelGeometry'].data['PIXEL_CORNER_LOS'] ## LOS length till tangent/impact point
        self.sc_pos_iau = self.hdul['SpacecraftGeometry'].data['V_SPACECRAFT'] ## SC pos vector from Mars center in km
        # Calc pixel vector from sc and planet
        self.pixel_vec_from_sc_iau = self.pixel_uvec_from_sc * self.los_length[:, None, :, :]
        self.pixel_vec_from_pla_iau = self.sc_pos_iau[:, :, None, None] + self.pixel_vec_from_sc_iau
        #self.calc_pixel_vec_dest()

    def get_trans_mat_to_sc_frame(self, frame_tbc):
        '''
        Returns matrix(ces) converting vectors to sc frame
        arg:
            frame_tbc: A frame to be converted (tbc) to sc frame
        '''
        if frame_tbc != 'IAU_MARS':
            frame_tbc = '_' + frame_tbc
        else:
            frame_tbc = ''
        return np.transpose(np.array(
                [self.hdul['SpacecraftGeometry'].data[:]['VX_SPACECRAFT'+frame_tbc],
                 self.hdul['SpacecraftGeometry'].data[:]['VY_SPACECRAFT'+frame_tbc],
                 self.hdul['SpacecraftGeometry'].data[:]['VZ_SPACECRAFT'+frame_tbc]]),axes=[1,2,0])

    def get_trans_mat(self):
        '''
        Returns matrix(ces) converting vectors from one to another frames
        What is done here is converting a frame via sc_frame:
            1. Get a trans mat A = get_trans_mat_to_sc_frame(from_frame) (from_frame -> sc_frame)
            2. Get a trans mat B = get_trans_mat_to_sc_frame(to_frame)   (to_frame -> sc_frame)
            3. Get a trans mat inverse(B) (sc_frame -> to_frame)
            4. Then create a trans mat inverse(B)A (from_frame -> sc_frame -> to_frame)
        '''
        return np.array([np.matmul(tomat,np.linalg.inv(frommat))
                         for tomat,frommat in zip(self.get_trans_mat_to_sc_frame(self.to_frame),
                                                  self.get_trans_mat_to_sc_frame(self.from_frame))])


    def calc_pixel_vec_dest(self):
        tmat = self.get_trans_mat()
        pixel_vec_from_sc_iau = np.transpose(self.pixel_vec_from_sc_iau, axes=(0,2,3,1))
        pixel_vec_from_pla_iau = np.transpose(self.pixel_vec_from_pla_iau, axes=(0,2,3,1))
        self.pixel_vec_from_sc_dest = np.transpose(np.array([[[np.matmul(itmat, kpv) for kpv in jpv] for jpv in ipv]
                                        for itmat, ipv in zip(tmat,pixel_vec_from_sc_iau)]), axes=(0,3,1,2))
        self.pixel_vec_from_pla_dest = np.transpose(np.array([[[np.matmul(itmat, kpv) for kpv in jpv] for jpv in ipv]
                                        for itmat, ipv in zip(tmat,pixel_vec_from_pla_iau)]), axes=(0,3,1,2))



    # Functions below may need to be implemented in the future
    """def xyoffset(pixelvec,mat,r,x,y):# need km, km/s
        pv=np.matmul(mat,pixelvec)
        ps=-r-np.dot(r,pv)*pv
        return [np.dot(ps,x),np.dot(ps,y)]

    def getpixeloffsets(fits):
        tomats=gettmats(fits,"","INERTIAL")#used to convert pixel_vec from IAU_MARS (default) to inertial
        scvecs=fits['SpacecraftGeometry'].data[:]['V_SPACECRAFT_INERTIAL']
        scr=[np.linalg.norm(scv) for scv in scvecs]
        scvels=fits['SpacecraftGeometry'].data[:]['V_SPACECRAFT_RATE_INERTIAL']
        normvecs=[v/np.linalg.norm(v) for v in [np.cross(r,v) for r,v in zip(scvecs,scvels)]]
        lrlvecs=[lrl_vec(r,v) for r,v in zip(scvecs,scvels)]
        alongvecs=[-np.cross(l,n) for l,n in zip(lrlvecs,normvecs)]
        pixelvecs=np.transpose(fits['PixelGeometry'].data[:]['PIXEL_VEC'],axes=[0,2,3,1])
        pvshape=pixelvecs.shape
        print(pvshape)
        pv=np.array([[[xyoffset(pv,mat,r,x,y)
                                   for pv in p] for p in s]
                                 for s,mat,r,x,y in zip(pixelvecs,tomats,scvecs,alongvecs,normvecs)])
        #pv=np.reshape(pixelvecs,[pvshape[0],-1,3])
        #pixeloffsets=np.array([map(xyoffset,
        #                           s,
        #                           itertools.repeat(mat,len(s)),
        #                           itertools.repeat(r,len(s)),
        #                           itertools.repeat(x,len(s)),
        #                           itertools.repeat(y,len(s)))
        #                         for s,mat,r,x,y in zip(pv,tomats,scvecs,alongvecs,normvecs)])
        #pv=np.reshape(pixeloffsets,[pvshape[0],pvshape[1],pvshape[2],2])
        return pv"""
