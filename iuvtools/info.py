# obsinfo
def get_orbit_number(hdul):
    orbit_number = hdul['Observation'].data['Orbit_Number']
    return orbit_number[0]

def get_segment(hdul):
    segment = hdul['Observation'].data['Orbit_Segment']
    return segment[0]

def get_channel(hdul):
    channel = hdul['Observation'].data['Channel']
    return channel

def get_solar_lon(hdul):
    Ls = hdul['Observation'].data['Solar_Longitude']
    return Ls[0]
