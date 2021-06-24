"""
Template code for reading S2S data in netCDF format

Author: 2020, John Methven
"""

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import datetime
from datetime import date
from datetime import timedelta

def read_s2srunoff(fstem, fname):

    '''
    Read in the runoff data from netCDF files.
    Input: path and name of file to read.
    Output: 
    :longitude  - degrees
    :latitude   - degrees
    :time       - month number
    :runoff     - daily runoff accumulation
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
    msl = data.variables['msl'][:]    
    runoff = data.variables['ro'][:]  
    if (runoff.ndim) == 4:
        member = data.variables['number'][:]
    else:
        member = np.zeros(1)
        
    data.close()
    #
    # Convert mean sea level pressure from Pa to hPa
    #
    msl = msl/100
    #
    # Time is in hours since 00UT, 1 Jan 1900.
    # Convert to timedelta format.
    #
    ftime = float(rtime[0])
    dtime = timedelta(hours=ftime)
    #
    # Note that you can add times in datetime and timedelta formats
    # which allows for leap years etc in time calculations.
    #
    startcal = datetime.datetime(1900, 1, 1)
    newcal = startcal+dtime
    print(newcal)
    leadtime = np.arange(float(len(rtime)))

    return leadtime, member, alon, alat, newcal, msl, runoff


def plot_basic(alon,alat,itime,field,fieldname):

    '''
    Plot 2-D field.
    Input: longitude, latitude, time-index, infield, name of field
    Output: Plot of field
    '''  
    fig = plt.figure()
    plt.imshow(field,interpolation='nearest')
    plt.colorbar(pad=0.04,fraction=0.046)
    plt.title(fieldname)
    plt.show()

    return
 

if __name__ == '__main__':
    
    '''
    Main program script for reading S2S runoff data.
    '''

    syy = '2018'
    smm = '06'
    sdd = '04'
    centre = 'ecmwf/'
    forc = 'PF.'
    fstem = '../data/S2S/japan/'+centre+'japan_ECMF_'+syy+'/'
    fname = 'japan_ECMF_'+forc+syy+smm+sdd+'.nc'
    #Read the data
    lead, member, alon, alat, sdate, msl, runoff = read_s2srunoff(fstem, fname)
    #
    # Plot the surface geopotential height on a map at time point itime
    #
    itime = 45
    if len(member) > 1:
        imem = 5
        field1 = runoff[itime, imem, :, :]
        field2 = msl[itime, imem, :, :]
    else:
        field1 = runoff[itime, :, :]
        field2 = msl[itime, :, :] 
        
    plot_basic(alon,alat,itime,field1,'runoff  (kg m**-2)')
    plot_basic(alon,alat,itime,field2,'MSLP  (hPa)')
    
    

