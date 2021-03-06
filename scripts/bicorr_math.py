# A few convenient math functions for the bicorr project

import matplotlib
#matplotlib.use('agg') # for flux
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

import sys
import os
import os.path
import scipy.io as sio
from scipy.optimize import curve_fit

import time
import numpy as np
np.set_printoptions(threshold=np.nan) # print entire matrices
import pandas as pd
from tqdm import *

# Don't import any bicorr modules here
# Other modules will import bicorr_math, but not the other way around


def prop_err_division(num,num_err,denom,denom_err):
    A = num/denom
    
    A_err = A*np.sqrt((num_err/num)**2+(denom_err/denom)**2)
    return A, A_err

def calc_centers(edges):
    """
    Simple method for returning centers from an array of bin edges. Calculates center between each point as difference between containing edges. 
    Example, plt.plot(bicorr.centers(edges),counts,'.k')
    Serves as a shortcode to first producing array of bin centers.
    
    Parameters
    ----------
    edges : ndarray
        Array of bin edges
    
    Returns
    -------
    centers : ndarray
        Array of bin edges
    """
    return (edges[:-1]+edges[1:])/2

def calc_histogram_mean(bin_edges, counts, print_flag = False, bin_centers_flag = False):
    """
    Calculate mean of a count rate distribution, counts vs. x. 
    Errors are calculated under the assumption that you are working
        with counting statistics. (C_err = sqrt(C) in each bin)
    
    Parameters
    ----------
    bin_edges : ndarray
        Bin edges for x
    counts : ndarray
        Bin counts
    print_flag : bool
        Option to print intermediate values
    bin_centers_flag : bool
        Option to provide bin centers instead of bin edges (useful for 2d histograms)
    
    Returns
    -------
    x_mean : float
    x_mean_err : float
    """
    if bin_centers_flag == True:
        bin_centers = bin_edges
    else: 
        bin_centers = calc_centers(bin_edges)
    
    num = np.sum(np.multiply(bin_centers,counts))  
    num_err = np.sqrt(np.sum(np.multiply(bin_centers**2,counts)))
    denom = np.sum(counts)    
    denom_err = np.sqrt(denom)    

    if print_flag:
        print('num: ',num)
        print('num_err: ',num_err)
        print('denom: ',denom)
        print('denom_err: ',denom_err)
    
    x_mean = num/denom
    x_mean_err = x_mean * np.sqrt((num_err/num)**2+(denom_err/denom)**2)
    
    if print_flag:
        print('x_mean: ',x_mean)
        print('x_mean_err:',x_mean_err)
    
    return x_mean, x_mean_err
    
def convert_energy_to_time(energy, distance = 1.05522):
    '''
    Convert energy in MeV to time in ns for neutrons that travel 1 m. From Matthew's `reldist.m` script. 
    6/5/18 Changing default to 105.522 cm, which is mean distance.
    
    Parameters
    ----------
    energy : float
        Neutron energy in MeV
    distance : float, optional
        Neutron flight distance in meters
        
    Returns
    -------
    time : float
        Time of flight of neutron
    '''
    # Constants
    m_n = 939.565 # MeV/c2
    c = 2.99e8 # m/s
    
    # Calculations
    v = c*np.sqrt(2*energy/m_n)
    time = np.divide(distance/v,1e-9)
    
    return time
    
def convert_time_to_energy(time, distance = 1.05522):
    '''
    Convert time in ns to energy in MeV for neutrons that travel 1 m. From Matthew's `reldist.m` script.
    6/5/18 Changing default to 105.522 cm, which is mean distance.
    
    If an array of times, use energy_bin_edges =  np.asarray(np.insert([bicorr.convert_time_to_energy(t) for t in dt_bin_edges[1:]],0,10000))
    
    Parameters
    ----------
    time : float
        Time of flight of neutron in ns
    distance : float, optional
        Neutron flight distance in meters
        
    Returns
    -------
    energy : float
        Neutron energy in MeV
    '''
    
    # Constants
    m_n = 939.565 # MeV/c2
    c = 2.99e8 # m/s
    
    v = distance * 1e9 / time # ns -> s
    energy = (m_n/2)*(v/c)**2
    
    return energy
    
    
def f_line(x, m, b): 
    """
    Line fit with equation y = mx + b

    Parameters
    ----------
    x : array
        x values
    m : float
        slope
    b : float
        y-intercept
        
    Returns
    -------
    y : array
        y values
    """
    y = m*x + b
    return y

def fit_f_line(x, y, y_err=None, p0=None, bounds=(-np.inf,np.inf)):
    """
    Fit a straight line with equation y = mx + b
    
    Parameters
    ----------
    x : ndarray
    y : ndarray
    y_err : ndarray, optional
    p0 : ndarra
        Initial guess of coefficients
    bounds : ndarray
        Boundaries for searching for coefficients
    
    Returns
    -------
    m, m_err : float
    b, b_err : float    
    """
    if y_err is None:
        y_err = np.ones(x.size)
    
    # Only use dat apoints with non-zero error
    w = np.where(y_err != 0)
    
    popt, pcov = curve_fit(f_line, x[w], y[w], sigma=y_err[w], p0=p0, absolute_sigma = True, bounds = bounds)
    
    errors = np.sqrt(np.diag(pcov))
    
    [m, b] = popt
    [m_err, b_err] = errors
   
    return m, m_err, b, b_err
    
    
    
    
    
    
    