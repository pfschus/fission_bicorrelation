# A few convenient math functions for the bicorr project



import os
import os.path
import time
import numpy as np
np.set_printoptions(threshold=np.nan) # print entire matrices
import pandas as pd
import scipy.io as sio
import sys
import matplotlib
import seaborn as sns
sns.set(style='ticks')

#matplotlib.use('agg') # for flux
import matplotlib.pyplot as plt
import time
from tqdm import *

from bicorr import *
from bicorr_sums import *
from bicorr_plot import * 


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

def calc_histogram_mean(bin_edges, counts, print_flag = False):
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
    
    Returns
    -------
    x_mean : float
    x_mean_err : float
    """
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