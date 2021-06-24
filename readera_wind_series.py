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

def read_era(filename, fyear, iprint):

    '''
    Read in the ground variable data from the netCDF file.
    Input: name of file to read.
    Output: 
    :longitude  - degrees
    :latitude   - degrees
    :outdates   - calendar dates for data points
    :u          - zonal component of the wind (m/s)
    :v          - meridional component of the wind (m/s)
    :mslp       - mean sea level pressure (hPa)
    '''
    print()
    print('Reading file ',filename)
    data = Dataset(filename, 'r')
    if iprint == 1:
        print(data)
        print()
        print(data.dimensions)
        print()
        print(data.variables)
        print()
        
    ftime = data.variables['time'][:]
    alon = data.variables['lon'][:]
    alat = data.variables['lat'][:]
    u = data.variables['u10'][:,:,:]
    v = data.variables['v10'][:,:,:]
    # Note that pressure is converted from Pa to hPa
    # In some years the files contain mean sea level pressure.
    # In some years the files contain surface pressure.
    if any((fyear < 2000, fyear >= 2018)):
        mslp = 0.01*data.variables['msl'][:,:,:]
    else:
        mslp = 0.01*data.variables['sp'][:,:,:]

    data.close()
    #
    # Time is in hours since 00UT, 1 Jan 1900.
    # Convert to timedelta format and then add to startcal to make a list of dates.
    # Note that you can add times in datetime and timedelta formats
    # which allows for leap years etc in time calculations.
    #
    startcal = datetime.datetime(1900, 1, 1)
    outdates = [startcal+timedelta(hours=ftimel) for ftimel in ftime]

    return alon, alat, outdates, u, v, mslp


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
    :windarr    - wind speed (m/s) time series at chosen location
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
    fdate = dstart.strftime("%Y_%m")
    fyear = dstart.year
    iprint = 0  # set to 1 to print variables on reading files; 0 for no print
    #Read the data
    filename = str(fpath+fstem+fdate+'_DET.nc')
    # Note that the str() function is included to ensure that these
    # variables are interpreted as character strings.
    alon, alat, outdates, u, v, mslp = read_era(filename, fyear, iprint)
    #
    # Find the indices of the grid box centred closest to the chosen location
    #
    intlon, intlat = subset_field(alon, alat, lonpick, latpick)
    #
    # Plot runoff on a map at time point itime
    #
    itime = 0
    plot_basic(alon,alat,itime,mslp,'sea level pressure  (hPa)')
    #plot_onproj(alon,alat,itime,mslp,'sea level pressure  (hPa)')
    #
    # Setup arrays to save time series data
    #
    dayarr = np.arange(ndays)
    timarr = np.arange(np.datetime64(str(dstart)), np.datetime64(str(dendp)))
    windarr = np.zeros(ndays)
    #
    # Loop over months, reading files and saving daily data
    #
    icarryon = 1
    dcur = dstart
    n=0
    while icarryon == 1:
        fdate = dcur.strftime("%Y_%m")
        fyear = dcur.year
        #Read the data
        filename = str(fpath+fstem+fdate+'_DET.nc')
        # Note that the str() function is included to ensure that these
        # variables are interpreted as character strings.
        alon, alat, outdates, u, v, mslp = read_era(filename, fyear, iprint)
        npts = len(outdates)
        #
        # Check whether the last day requested is within this month
        #
        if n+npts >= ndays:
            npts = ndays-n
            icarryon = 0
        #
        # Save the daily data required from this file
        #
        for i in range(npts):
            windarr[n+i] = np.sqrt(u[i, intlat, intlon]**2 + v[i, intlat, intlon]**2)
        #
        # Increment the date variable by number of days in this month
        #
        n = n+npts
        dcur=dcur+timedelta(days=npts)
    #
    # Now plot the time series
    #
    varname = 'wind speed  (m/s)'
    mytitle = 'Wind speed time series for chosen location'
    plot_series(dayarr, windarr, varname, mytitle)
    
    return dayarr, timarr, windarr


if __name__ == '__main__':
    
    '''
    Main program script extracting time series from ERA5 data.
    '''
    #
    # Pick the location to extract the time series
    # This coordinate is Reading.
    #
    lonpick = -0.9781
    latpick = 51.4543
    #
    # Select the start and end date required for the time series
    #
    dstart = datetime.date(2018, 1, 1)
    dend = datetime.date(2018, 3, 9)
    #
    # Set the path and filename stem for data files.   
    #
    fpath = '../data/ERA5/europe/europe_1halfx1half_ERA5_winds/'
    fstem = 'eur_remap_bilinear_1halfx1half_ERA5_3hr_'
    #
    # Call the function to extract the run-off time series
    #
    dayarr, timarr, windarr = extract_series(fpath, fstem, lonpick, latpick, dstart, dend)
    
