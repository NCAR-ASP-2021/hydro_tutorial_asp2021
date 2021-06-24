"""
Template code for reading ERA5 data in netCDF format

Author: 2020, John Methven
"""

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import datetime
from datetime import date
from datetime import timedelta

def read_ground(fstem, fname):

    '''
    Read in the ground variable data from the netCDF file.
    Input: path and name of file to read.
    Output: 
    :longitude  - degrees
    :latitude   - degrees
    :time       - month number
    :gpheight   - geopotential height of the ground (above the geoid)
    :lsmask     - land-sea mask
    :fraclake   - fraction of grid box covered by lakes
    '''
    filename = str(fstem+fname)
    data = Dataset(filename, 'r')
    print(data)
    print()
    print(data.dimensions)
    print()
    print(data.variables)
    print()
    rtime = data.variables['time'][:]
    alon = data.variables['longitude'][:]
    alat = data.variables['latitude'][:]
    gp = data.variables['z'][:,:,:]
    lsmask = data.variables['lsm'][:,:,:]
    fraclake = data.variables['cl'][:,:,:]    
    data.close()
    #
    # Convert surface geopotential to geopotential height (m)
    #
    gpheight = gp/9.80665
    #
    # Time is in hours since 00UT, 1 Jan 1900.
    # Convert to timedelta format.
    #
    ftime = float(rtime)
    dtime = timedelta(hours=ftime)
    #
    # Note that you can add times in datetime and timedelta formats
    # which allows for leap years etc in time calculations.
    #
    startcal = datetime.datetime(1900, 1, 1)
    newcal = startcal+dtime
    print(newcal)

    return alon, alat, newcal, gpheight, lsmask, fraclake


def plot_basic(alon,alat,itime,field3d,fieldname):

    '''
    Plot 2-D field.
    Input: longitude, latitude, time-index, infield, name of field
    Output: Plot of field
    '''  
    field = field3d[itime,:,:]
    fig = plt.figure()
    plt.imshow(field,interpolation='nearest')
    plt.colorbar(pad=0.04,fraction=0.046)
    plt.title(fieldname)
    plt.show()

    return
 

if __name__ == '__main__':
    
    '''
    Main program script for reading ERA5 data.
    '''

    fstem = '../data/ERA5/'
    #fname = 'europe_ERA5ground.20180601.nc'
    fname = 'japan_ERA5ground.20180601.nc'
    #Read the data
    alon, alat, time, gpheight, lsmask, fraclake = read_ground(fstem, fname)
    #
    # Plot the surface geopotential height on a map at time point itime
    #
    itime = 0
    plot_basic(alon,alat,itime,lsmask,'land-sea mask')
    plot_basic(alon,alat,itime,gpheight,'z  (m)')    
    #plot_basic(alon,alat,itime,fraclake,'lake area fraction')
    
    

