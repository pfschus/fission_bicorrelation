'''
Functions for dealing with bhm in energy space.
'''
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
from bicorr_plot import * 
from bicorr_sums import *
from bicorr_math import *

def build_energy_bin_edges(e_min=0,e_max=15,e_step=.025,print_flag=False):
    """
    Construct energy_bin_edges for the two-dimensional bicorrelation energy histograms. Energy in MeV units.
    
    Use as: e_bin_edges, num_e_bins = bicorr.build_energy_bin_edges()
    
    Parameters
    ----------
    e_min : int, optional
        Lower energy boundary
    e_max : int, optional
        Upper energy boundary
    e_step : float, optional
        Energy bin size
    print_flag : bool, optional
        Whether to print array details
        
    Returns
    -------
    e_bin_edges : ndarray
        One-dimensional array of energy bin edges
    num_e_bins : ndarray
        Number of bins in energy dimension
    """
    e_bin_edges = np.arange(e_min,e_max+e_step,e_step)
    num_e_bins = len(e_bin_edges)-1
    
    if print_flag:
        print('Built array of energy bin edges from', e_min, 'to', e_max, 'in', num_e_bins,'steps of',e_step,'MeV.')
    
    return e_bin_edges, num_e_bins    

def build_dict_det_dist(file_path = '../meas_info/detector_distances.xlsx'):
    '''
    Load excel file with detector distances, convert to m, and send to a dictionary.
    
    Parameters
    ----------
    file_path : str, optional
        Relative location of excel file with detector distances in it
        
    Returns
    -------
    dict_det_dist : dict
        Dictionary of detector channel number : distance from fc
    '''
    det_distance_df = pd.read_excel(file_path)
    det_distance_df['Distance (cm)'] /= 100 # convert to m
    dict_det_dist = dict(zip(det_distance_df['Channel'], det_distance_df['Distance (cm)']))
    return dict_det_dist
    
def alloc_bhm_e(num_det_pairs, num_intn_types, num_e_bins):
    """
    Preallocate bhm_e
    
    Four dimensions: num_det_pairs x num_intn_types x num_e_bins x num_e_bins
    Interaction type index:  (0=nn, 1=np, 2=pn, 3=pp)
    
    Parameters
    ----------
    num_det_pairs : int
    num_intn_types : int, optional
    num_e_bins : int
    
    Returns
    -------
    bhm_e : ndarray
        Zero-filled bhm_e in energy space    
    """
    bhm_e = np.zeros((num_det_pairs,num_intn_types,num_e_bins,num_e_bins),dtype=np.uint32)

    return bhm_e    
    
def fill_bhm_e(bhm_e, bicorr_data, det_df, dict_det_dist, e_bin_edges, disable_tqdm = False):
    """
    Fill bhm_e. Structure:
        Dimension 0: detector pair, use dictionary 'dict_pair_to_index', where pair is (100*det1ch+det2ch)
        Dimension 1: interaction type, length 1. Only storing 0=nn.
        Dimension 2: e bin for detector 1
        Dimension 3: e bin for detector 2
        
    Must have allocated bhm_e and loaded bicorr_data
    
    Parameters
    ----------
    bhm_e : ndarray
        Master histogram of bicorrelation events in energy space. Dimensions listed above.
    bicorr_data : ndarray
        Each element contains the following info for one bicorrelation pair
        Columns are 0: event, np.int32
                    1: det1ch, np.int8
                    2: det1par, np.int8
                    3: det1t, np.float16
                    4: det2ch, np.int8
                    5: det2par, np.int8
                    6: det2t, np.float16
    det_df : pandas dataFrame
        dataFrame of detector pair indices and angles
    dict_det_dist : dict
        Dictionary of detector channel number : distance from fc
    e_bin_edges : ndarray
        One-dimensional array of energy bin edges
    disable_tqdm : bool, optional
        Flag to disable tqdm progress bar
        
    Returns
    -------
    bhm_e : ndarray
        Same as input, but filled with event information from bicorr_data
    """
    dict_pair_to_index, dict_index_to_pair, dict_pair_to_angle = build_dict_det_pair(det_df)
    
    e_min = np.min(e_bin_edges)
    e_max = np.max(e_bin_edges)
    e_step = e_bin_edges[1]-e_bin_edges[0]
    
    for i in tqdm(np.arange(bicorr_data.shape[0]),ascii=True,disable=False):
        event = bicorr_data[i]
        det1t = event['det1t']; det2t = event['det2t'];

        logic = np.logical_and([det1t > 0, event['det1par'] == 1], [det2t>0, event['det2par'] == 1])

        if np.logical_and(logic[0],logic[1]): # nn with both t > 0
            det1dist = dict_det_dist[event['det1ch']]
            det2dist = dict_det_dist[event['det2ch']]

            det1e = convert_time_to_energy(det1t, det1dist)
            det2e = convert_time_to_energy(det2t, det2dist)

            # Check that they are in range of the histogram        
            if np.logical_and(e_min < det1e < e_max, e_min < det2e < e_max):
                # Determine index of detector pairs
                pair_i = dict_pair_to_index[event['det1ch']*100+event['det2ch']]

                # Determine indices of energy values
                e1_i = int(np.floor((det1e-e_min)/e_step))
                e2_i = int(np.floor((det2e-e_min)/e_step))

                # Increment bhm_e
                pair_i = dict_pair_to_index[event['det1ch']*100+event['det2ch']]
                bhm_e[pair_i,0,e1_i,e2_i] += 1

    return bhm_e
    
    
    
    
def build_bhm_e(folder_start=1,folder_end=2, det_df = None, dict_det_dist = None, e_bin_edges = None, checkpoint_flag = True, save_flag = True, bhm_e_filename = 'bhm_e', root_path = None, disable_tqdm = False, print_flag = True, return_flag = True):
    """
    Load bicorr_data from folder's bicorr# file and fill energy histogram. Loop through folders specified by `folder_start` and `folder_end`. Built for e_bin_edges generated using default settings in bicorr.build_energy_bin_edges().
    
    Parameters
    ----------
    folder_start : int, optional
        First folder
    folder_end : int, optional
        Last folder + 1 (for example, folder_end = 2 will end at folder 1)
    det_df : pandas dataFrame, optional
        dataFrame of detector pair indices and angles   
        Default is to look for the file in '../meas_info/det_df_pairs_angles.csv'
    dict_det_dist : dict
        Dictionary of detector channel number : distance from fc
        Default is to look for the file in '../meas_info/detector_distances.xlsx'
    e_bin_edges : ndarray, optional
        Edges of energy bin array in MeV
        If None, use default settings from build_energy_bin_edges()
    checkpoint_flag : bool, optional
        Generate checkpoint plots?
    save_flag : bool, optional
        Save sparse matrix to disk?
    bhm_e_filename : str, optional
        Filename for bhm_e array
    root_path : int, optional
        Relative path to folder where data folders exist (1, 2, 3, etc.). default = cwd
    disable_tqdm : bool, optional
        Flag to disable tqdm progress bar
    print_flag : bool, optional
        Print status updates along the way?
    return_flag : bool, optional
        Option to return bhm_e and e_bin_edges. Otherwise, return nothing.
    
    Returns
    -------
    bhm_e : ndarray
        Master histogram of bicorrelation events in energy space. 
        Dimension 0: detector pair, use dictionary 'dict_pair_to_index', where pair is (100*det1ch+det2ch)
        Dimension 1: interaction type, length 1. Only storing 0=nn.
        Dimension 2: e bin for detector 1
        Dimension 3: e bin for detector 2
    e_bin_edges : ndarray
        One-dimensional array of energy bin edges
    """    
    # Load det_df, dict_det_dist if not provided
    if det_df is None: det_df = load_det_df()
    if dict_det_dist is None: dict_det_dist = build_dict_det_dist()
    
    # If no data path provided, look for data folders here
    if root_path is None: root_path = os.getcwd()
    
    # Folders to run
    folders = np.arange(folder_start,folder_end,1)
    if print_flag: print('Generating bicorr histogram for bicorr data in folders: ', folders)    

    # Handle dt_bin_edges
    if e_bin_edges is None: # Use default settings
        e_bin_edges, num_e_bins  = build_energy_bin_edges()
    else:
        num_e_bins = len(e_bin_edges)-1
    
    # Set up binning
    num_det_pairs = len(det_df)
    num_intn_types = 1
    
    # Create bhm_e, empty
    bhm_e = alloc_bhm_e(num_det_pairs, num_intn_types, num_e_bins)
    
    # Loop through each folder and fill the histogram
    for folder in folders:
        if print_flag: print('Loading data in folder ',folder)
        bicorr_data = load_bicorr(folder, root_path = root_path)
        if checkpoint_flag:
            fig_folder = os.path.join(root_path + '/' + str(folder) + '/fig')
            bicorr_checkpoint_plots(bicorr_data,fig_folder = fig_folder,show_flag=False)
        if print_flag: print('Building bhm in folder ',folder)
        bhm_e = fill_bhm_e(bhm_e, bicorr_data, det_df, dict_det_dist, e_bin_edges, disable_tqdm = False)
        
    if save_flag: 
        # Generate sparse matrix
        if print_flag: print('Saving bhm_e to .npz file')
        note = 'bhm_e generated from folder {} to {} in directory {}'.format(folder_start,folder_end,os.getcwd())
        save_bhm_e(bhm_e, e_bin_edges, bhm_e_filename=bhm_e_filename, note=note)
        
    if print_flag: print('Bicorr hist master bhm_e build complete')
                
    if return_flag: return bhm_e, e_bin_edges     
    
    
    
    
    
    
    
    
    
    
    
    
def save_bhm_e(bhm_e, e_bin_edges, save_folder = None, bhm_e_filename = 'bhm_e', note = 'note'):
    """
    Save bhm_e to .npz file. (Reload using load_bhm_e function)
    
    Parameters
    ----------
    bhm_e : ndarray
        Master histogram of bicorrelation events in energy space. 
        Dimension 0: detector pair, use dictionary 'dict_pair_to_index', where pair is (100*det1ch+det2ch)
        Dimension 1: interaction type, length 1. Only storing 0=nn.
        Dimension 2: e bin for detector 1
        Dimension 3: e bin for detector 2
    e_bin_edges : ndarray
        One-dimensional array of energy bin edges
    save_folder : str, optional
        Optional destination folder. If None, then save in current working directory
    bhm_e_filename : str, optional
        Filename
    note : str, optional
        Note to include
    
    Returns
    -------
    n/a
    """
    if save_folder is not None:
        # check if save_folder exists
        try:
            os.stat(save_folder)
        except:
            os.mkdir(save_folder)
        bhm_e_filename = os.path.join(save_folder, bhm_e_filename)    
    
    np.savez(bhm_e_filename, bhm_e = bhm_e, e_bin_edges=e_bin_edges, note = note)
    
def load_bhm_e(filepath = None, filename = None):
    """
    Load .npz file containing `bhm_e`, `e_bin_edges`, and `note`. This file was probably generated by the function `save_bhm_e`.
    
    Parameters
    ----------
    filepath : str, optional
        Where is the `bhm_e.npz` file? If None, look in current directory
    filename : str, optional
        What is the file called? If None, then look for `bhm_e.npz`
    
    Returns
    -------
    bhm_e : ndarray
        Master histogram of bicorrelation events in energy space. 
        Dimension 0: detector pair, use dictionary 'dict_pair_to_index', where pair is (100*det1ch+det2ch)
        Dimension 1: interaction type, length 1. Only storing 0=nn.
        Dimension 2: e bin for detector 1
        Dimension 3: e bin for detector 2
    e_bin_edges : ndarray
        One-dimensional array of energy bin edges
    note : str, optional
        Note to include
    """
    if filename is None:
        filename = 'bhm_e.npz'
        
    if filepath is None:
        npzfile = np.load(filename)
    else:
        npzfile = np.load(os.path.join(filepath,filename))
        
    bhm_e = npzfile['bhm_e']
    e_bin_edges = npzfile['e_bin_edges']
    if 'note' in npzfile:
        note = npzfile['note']    
    else:
        note = 'note'
        
    return bhm_e, e_bin_edges, note
    