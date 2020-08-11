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

def get_mcp_volt(hdul):
    mcp_volt = hdul['Observation'].data['MCP_Volt']
    return mcp_volt[0]

def dayside(hdul): ## Check if the voltage value is correct later
    mcp_volt = get_mcp_volt(hdul)
    if mcp_volt < 700:
        return True
    else:
        return False
