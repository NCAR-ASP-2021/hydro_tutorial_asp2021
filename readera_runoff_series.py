"""
Template code for looping over dates, reading multiple ERA5 data files in netCDF format,
sub-setting the data from a particular location and creating a time series.

Author: 2020, John Methven
"""

import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset
import datetime
from datetime import date
from datetime import timedelta

def read_era(filename, iprint):

    '''
    Read in the ground variable data from the netCDF file.
    Input: name of file to read.
    Output: 
    :longitude  - degrees
    :latitude   - degrees
    :time       - month number
    :runoff     - runoff (m)
    '''
    data = Dataset(filename, 'r')
    if iprint == 1:
        print(data)
        print()
        print(data.dimensions)
        print()
        print(data.variables)
        print()
        
    rtime = data.variables['time'][:]
    alon = data.variables['longitude'][:]
    alat = data.variables['latitude'][:]
    runoff = data.variables['ro'][:,:,:]  
    data.close()
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
    print('Read data for accumulation to ',newcal)

    return alon, alat, newcal, runoff


def subset_field(alon, alat, lonpick, latpick):

    '''
    Find the indices of the grid point centred closest to chosen location.
    Input: 
    :alon       - longitude points
    :alat       - latitude points
    :lonpick    - longitude of chosen location
    :latpick    - latitude of chosen location
    Output:
    :intlon     - index of longitude for chosen point
    :intlat     = index of latitude for chosen point
    '''
    #
    # Using the fact that longitude and latitude are regularly spaced on grid.
    # Also, points ordered from north to south in latitude.
    # The indices (intlon, intlat) correspond to the centre of the closest grid-box 
    # to the chosen location.
    #
    dlon = alon[1]-alon[0]
    dlat = alat[1]-alat[0] # note that dlat is negative due to ordering of grid
    lonwest = alon[0]-0.5*dlon
    latnorth = alat[0]-0.5*dlat
    intlon = int(round((lonpick-lonwest)/dlon))
    intlat = int(round((latpick-latnorth)/dlat))
    print()
    print('Longitude of nearest grid box = ',alon[intlon])
    print('Latitude of nearest grid box = ',alat[intlat])
    
    return intlon, intlat


def plot_basic(alon,alat,itime,field3d,fieldname):

    '''
    Plot 2-D field as a simple pixel image.
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
 
    
def plot_onproj(alon,alat,itime,field3d,fieldname):

    '''
    Plot 2-D field on map using cartopy map projection.
    Input: longitude, latitude, time-index, infield, name of field
    Output: Plot of field
    '''  
    field = field3d[itime,:,:]
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.gridlines()
    plt.title(fieldname)
    nlevs = 20
    plt.contourf(alon, alat, field, nlevs,
             transform=ccrs.PlateCarree())
    plt.show()

    return


def plot_series(timarr, y, ylabel, mytitle):
    '''
    Plot the subset time series
    Inputs:
        timarr   - time array in datetime format
        y        - data time series
        ylabel   - string name for data
        mytitle  - plot title
    '''
    fig = plt.figure()
    plt.plot(timarr,y,label=ylabel)
    plt.xlabel("Days")
    plt.ylabel(ylabel)
    plt.title(mytitle)
    plt.legend()
    plt.show()


def extract_series(fpath, fstem, lonpick, latpick, dstart, dend):
    '''
    High level function controlling extraction of runoff time series 
    for chosen location.
    Input: fpath, fstem determine the name of file to read
    :lonpick    - longitude of chosen location
    :latpick    - latitude of chosen location
    :dstart     - start date in datetime.date format
    :dend       - end date in datetime.date format
    Output: 
    :dayarr     - time in days since start
    :timarr     - time series in datetime format
    :runoffloc  - runoff (m) time series at chosen location
    '''   
    #
    # Set end date and start date of required time series
    #
    dendp = dend+timedelta(days=1)
    tinterval = dendp-dstart
    ndays = tinterval.days
    #
    # Plot the data for the first date in the interval
    #
    fdate = dstart.strftime("%Y%m%d")
    iprint = 0  # set to 1 to print variables on reading files; 0 for no print
    #Read the data
    filename = str(fpath+fstem+fdate+'.nc')
    # Note that the str() function is included to ensure that these
    # variables are interpreted as character strings.
    alon, alat, time, runoff = read_era(filename, iprint)
    #
    # Find the indices of the grid box centred closest to the chosen location
    #
    intlon, intlat = subset_field(alon, alat, lonpick, latpick)
    #
    # Plot runoff on a map at time point itime
    #
    itime = 0
    plot_basic(alon,alat,itime,runoff,'runoff  (m)')
    #plot_onproj(alon,alat,itime,runoff,'runoff  (m)')
    #
    # Setup arrays to save time series data
    #
    dayarr = np.arange(ndays)
    timarr = np.arange(np.datetime64(str(dstart)), np.datetime64(str(dendp)))
    runoffarr = np.zeros(ndays)
    #
    # Loop over dates, reading files and saving data
    #
    dcur = dstart
    for n in range(ndays):
        fdate = dcur.strftime("%Y%m%d")
        #Read the data
        filename = str(fpath+fstem+fdate+'.nc')
        # Note that the str() function is included to ensure that these
        # variables are interpreted as character strings.
        alon, alat, time, runoff = read_era(filename, iprint)
        #
        # Save the data required from this time-point
        #
        runoffarr[n] = runoff[0, intlat, intlon]
        #
        # Increment the date variable by one day
        #
        dcur=dcur+timedelta(days=1)

    #
    # Now plot the time series
    #
    varname = 'runoff  (m)'
    mytitle = 'Runoff time series for chosen location'
    plot_series(dayarr, runoffarr, varname, mytitle)
    
    return dayarr, timarr, runoffarr


if __name__ == '__main__':
    
    '''
    Main program script extracting time series from ERA5 data.
    '''
    #
    # Pick the location to extract the time series
    #
    lonpick = 136.47642
    latpick = 35.63422
    #
    # Select the start and end date required for the time series
    #
    dstart = datetime.date(2018, 6, 1)
    dend = datetime.date(2018, 7, 31)
    #
    # Set the path and filename stem for data files.   
    #
    fpath = '../data/ERA5/japan/unzipped/'
    fstem = 'japan_ERA5land.'
    #
    # Call the function to extract the run-off time series
    #
    dayarr, timarr, runoffarr = extract_series(fpath, fstem, lonpick, latpick, dstart, dend)
    
