class IuvRc:
    def __init__(self):
        self.iuvdataloc = '/Volumes/Fenix/work/data/maven_iuvs/'
        self.pfpdataloc = '/Volumes/Fenix/work/data/maven/data/sci/'
        self.saveloc = '/Users/masunaga/work/python_git/maven/iuvs/products/'
        self.spiceloc = '/Volumes/Fenix/work/data/maven_iuvs/spice/'

# IUVS data directory
iuvdataloc = IuvRc().iuvdataloc

# Particle and Field data directory
pfpdataloc = IuvRc().pfpdataloc

# SPICE kernel directory and paths
spiceloc = IuvRc().spiceloc

# product directory'
saveloc = IuvRc().saveloc

# PyUVS directory
pyuvs_directory = '/Users/masunaga/work/python_git/maven/iuvs/lib/PyUVS'

# chaffin directory
chaffin_directory = '/Users/masunaga/work/python_git/maven/iuvs/lib/chaffin'

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
