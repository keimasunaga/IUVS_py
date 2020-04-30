class IuvRc:
    def __init__(self):
        self.dataloc = '/Volumes/Gravity/work/data/maven_iuvs/'
        self.saveloc = '/Users/masunaga/work/python_git/maven/iuvs/product/'
        self.spiceloc = '/Volumes/Gravity/work/data/maven_iuvs/spice/'

# data directory
dataloc = IuvRc().dataloc

# SPICE kernel directory and paths
spiceloc = IuvRc().spiceloc

# product directory'
productloc = IuvRc().saveloc

# VM access info
vm_username = 'username'
vm_password = 'password'

# physical variables
R_Mars_km = 3.3895e3  # [km]

# instrument variables
slit_width_deg = 10  # [deg]
slit_width_mm = 0.1  # [mm]
limb_port_for = 24  # [deg]
nadir_port_for = 60  # [deg]
port_separation = 36  # [deg]
pixel_size_mm = 0.018  # [mm]
focal_length_mm = 100.  # [mm]
muv_dispersion = 0.16325  # [nm/pix]
slit_pix_min = 77  # starting pixel position of slit
slit_pix_max = 916  # ending pixel position of slit
