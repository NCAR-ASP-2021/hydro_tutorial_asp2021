# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:57:30 2020

@author: John Methven
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats.stats import linregress
import datetime

def read_ERA_csv(fpath, filename):
    '''
    Function to read ERA5 data from selected file in csv format.
    Inputs:
        fpath - string for file path
        filename - string for data file name
    Outputs:
        ncols - number of columns found on first row of data file
        ndates - number of rows, each with data for a different date
        header - top row of file contains header for data
        datarr - data below the top row containing dates only in string format
        data - data below the top row (each row is a separate time-point)
    '''
    fnam = fpath+filename
    header = np.genfromtxt(fnam, delimiter=',', dtype='str', max_rows=1)
    ncols = len(header)
    datarr = np.genfromtxt(fnam, delimiter=',', skip_header=1, usecols=0, dtype='str')
    data = np.genfromtxt(fnam, delimiter=',', skip_header=1)
    ndates = len(data[:,0])
    return ncols, ndates, header, datarr, data


def read_ENTSOE_csv(fpath, filename):
    '''
    Function to read ENTSO-E daily data from selected file in csv format.
    Note that the units of national electricity load are in GW.
    Inputs:
        fpath - string for file path
        filename - string for data file name
    Outputs:
        ncols - number of columns found on first row of data file
        ndates - number of rows, each with data for a different date
        header - top row of file contains header for data
        data - data below the top row (each row is a separate time-point)
    '''
    fnam = fpath+filename
    header = np.genfromtxt(fnam, delimiter=',', dtype='str', max_rows=1)
    ncols = len(header)
    data = np.genfromtxt(fnam, delimiter=',', skip_header=1)
    ndates = len(data[:,0])
    return ncols, ndates, header, data


def read_allvariables(fpath):
    '''
    Function to read all the relevant data files and to arrange the data into
    suitable arrays.
    Inputs:
        fpath - string for file path
    Outputs:
        datarr - array containing dates in the datetime format
        t2m_uk - temperature (at 2m) data
        demand_uk - simulated demand data using ERA5
        entso - metered demand data from the ENTSOE database
    '''
    #
    # Set path for data files and file names.
    #
    erapath = fpath+'ERA5_reanalysis_models/demand_model_outputs/'
    t2m_file = 'ERA5_T2m_1979_2018.csv'
    demand_file = 'ERA5_weather_dependent_demand_1979_2018.csv'
    entpath = fpath+'ENTSOE/'
    ENTSO_file = 'GB-daily-load-values_2010-2015.csv'
    #
    # Read the ERA5 2m temperature and modelled demand data
    #
    tncols, ntdates, theader, ttim, tdata = read_ERA_csv(erapath, t2m_file)
    dncols, nddates, dheader, dtim, demand = read_ERA_csv(erapath, demand_file)
    #
    # Convert the dates read in as strings into the datetime format.
    # This can be used in calculations as a real time variable and plotting.
    #
    datlist = [datetime.datetime.strptime(dtimelement, "%Y-%m-%d") for dtimelement in dtim]
    datarr = np.asarray(datlist)
    #
    # Check for consistency between the datasets
    #
    assert nddates == ntdates, 'Error: number of time points in T and demand do not match'
    assert dncols == tncols, 'Error: number of country columns in T and demand do not match'
    #
    # Find UK data, knowing that it is in the last column of demand array
    #
    print(dheader[dncols-1])
    demand_uk = demand[:,dncols-1]
    #
    # Note that the date format for the T2m file is not the same, but the date rows correspond.
    # This means that there is an extra column in the tdata array (compared with theader).
    #
    print(theader[tncols-1])
    tukpos = len(tdata[0,:])
    t2m_uk = tdata[:,tukpos-1]
    #
    # Read the daily ENTSO-E data for the UK.
    # Note that both ENTSO-E data and modelled demand are in Giga-Watts.
    #
    encols, nedates, eheader, entso = read_ENTSOE_csv(entpath, ENTSO_file)
    print()
    print('header for ENTSO-E data')
    print(eheader)
    
    return datarr, t2m_uk, demand_uk, entso

    
if __name__ == '__main__':

    #Run the main program
    '''
    Main script for Lab 1.
    Calibrating electricity demand versus temperature.
    Using linear regression to create demand time series.
    '''
    #
    # Set your own filepath pointing to the data in your directories
    #
    fpath = '../data/'
    #
    # Read in daily data for T2m and modelled demand from ERA5 and
    # also the measured UK electricity demand from ENTSO-E.
    #
    datarr, t2m_uk, demand_uk, entso = read_allvariables(fpath)
    #
    ndates = len(datarr)
    print(datarr[0],' is the first date in the ERA files')
    print(datarr[ndates-1], ' is the last date')
    print('')
    print('Now over to you from here on!!')
    