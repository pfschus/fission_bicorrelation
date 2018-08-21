"""
Custom functions for bicorr simulation data
"""

import matplotlib
#matplotlib.use('agg') # for flux
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

import sys
import os
import os.path
import scipy.io as sio

import time
import numpy as np
np.set_printoptions(threshold=np.nan) # print entire matrices
import pandas as pd
from tqdm import *

# These other modules are called by bicorr.py
# bicorr.py does not call these modules at all
import bicorr as bicorr
import bicorr_e as bicorr_e
import bicorr_plot as bicorr_plot
import bicorr_math as bicorr_math


######## READ CCED -> BICORR ###########
def generate_bicorr_sim(cced_filename,bicorr_filename,Ethresh=0.1):
    """
    Parse simulation cced file and produce bicorr output file.
    Developed in fnpc\analysis\cgmf\import_data.ipynb
    
	Reads in cced file, format: event, detector, particle_type, time, FILLER, height, FILLER
	Produces bicorr file, format: event, det1ch, det1par, det1t, det2ch, det2par, det2t
    
    Parameters
    ----------
    cced_filename : str
        Filename (including path) of cced file to parse
    bicorr_filename : str
        Filename (including path) of bicorr file to generate
    Ethresh : float, optional
        Pulse height threshold, MeVee
    
    Returns
    -------
    n/a
    """
    ccedTypeSim = np.dtype([('event', np.int32), 
                        ('detector', np.int8), 
                        ('particle_type', np.int8), 
                        ('time', np.float16), 
                        ('height', np.float32)])
    data = np.genfromtxt(cced_filename,dtype=ccedTypeSim,usecols=(0,1,2,3,5))
    
    print_file = open(bicorr_filename,'w')
    print('write bicorr to:', bicorr_filename)

    # eventNum is the current event number. From lines i to j.
    eventNum = data[0]['event']; # Start with the first event in the data chunk.
                                 # If reading entire file, this is 1. If reading a chunk, this may be higher
    i        = 0;                # First line number of first event is always 0

    # Run through the data matrix (all information from cced file)        
    #     l is the line number of the current line, starting at 0.
    #     e is the event number of the current line, starting at 1
    for l, e in enumerate(tqdm(data[:]['event'],ascii=True)):
        if e == eventNum: # Still on the same event
            pass
        
        if e != eventNum: # Store info from current event, move onto the next
            j=l
            n_ints = j-i
            
            if n_ints > 1: # This differs from the original
                ccedEvent = data[i:j][:] # Data from this event
                heights = ccedEvent['heights']                
                
                # I can skip all the dets and fc logic here  
                for d1 in range(0,len(ccedEvent)-1,1):
                    for d2 in range(d1+1,len(ccedEvent),1):
                        if np.logical_and(heights[d1]>Ethresh,heights[d2]>Ethresh):
                            # Note: I am removing events where there are two interactions in a single channel. This could be a thing to revisit later if I run into bugs. 
                            if (ccedEvent[d1]['detector'] != ccedEvent[d2]['detector']):
                                print_file.write(
                                  str(ccedEvent[0]['event']) + '  ' +
                                  str(ccedEvent[d1]['detector']) + '  ' +
                                  str(ccedEvent[d1]['particle_type']) + '  ' +
                                  str(ccedEvent[d1]['time']) + '  ' +
                                  str(ccedEvent[d2]['detector']) + '  ' +
                                  str(ccedEvent[d2]['particle_type']) + '  ' +
                                  str(ccedEvent[d2]['time']) + '\n')
                
            eventNum = e
            i = l
            
    print_file.close()
    
def build_bhm_both_sim(bicorr_path, bicorr_filenames, det_df = None, dt_bin_edges = None, dict_det_dist = None, e_bin_edges = None, checkpoint_flag = True, save_flag = True, sparse_filename = 'sparse_bhm', bhm_e_filename = 'bhm_e', disable_tqdm = False, print_flag = True, return_flag = False):
    """
    Load bicorr_data specified by name. Built for e_bin_edges generated using default settings in bicorr.build_energy_bin_edges().
    
    Parameters
    ----------
    bicorr_path : str, optional
        Full path of bicorr file, NOT including filename
    bicorr_filenames : str, optional
        Filenames
        If just one, put in brackets: [filename]
    det_df : pandas dataFrame, optional
        dataFrame of detector pair indices and angles   
        Default is to look for the file in '../meas_info/det_df_pairs_angles.csv'
    dt_bin_edges : ndarray, optional
        Edges of time bin array in ns
        If None, use default settings from build_dt_bin_edges()    
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
    sparse_filename : str, optional
        Filename for sparse matrix
    bhm_e_filename : str, optional
        Filename for bhm_e array
    disable_tqdm : bool, optional
        Flag to disable tqdm progress bar
    print_flag : bool, optional
        Print status updates along the way?
    return_flag : bool, optional
        Option to return bhm_e and e_bin_edges. Otherwise, return nothing.
    
    Returns
    -------
    bhm : ndarray
        Master histogram of bicorrelation histograms across all detector pairs and interaction types.
        Dimension 0: detector pair, use dictionary `dict_pair_to_index` where pair is (100*det1ch+det2ch)  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
    dt_bin_edges : ndarray
        One-dimensional array of time bin edges
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
    if det_df is None: det_df = bicorr.load_det_df()
    if dict_det_dist is None: dict_det_dist = bicorr.build_dict_det_dist()

    # Handle dt_bin_edges
    if dt_bin_edges is None: # Use default settings
        dt_bin_edges, num_dt_bins = bicorr.build_dt_bin_edges(dt_min=0)
    else:
        num_dt_bins = len(dt_bin_edges)-1
    # Handle e_bin_edges
    if e_bin_edges is None: # Use default settings
        e_bin_edges, num_e_bins  = bicorr_e.build_energy_bin_edges()
    else:
        num_e_bins = len(e_bin_edges)-1
    
    # Set up binning
    num_det_pairs = len(det_df)
    
    # Create bhm, empty- all event types
    bhm = bicorr.alloc_bhm(num_det_pairs, 4, num_dt_bins)    
    # Create bhm_e, empty- nn only
    bhm_e = bicorr_e.alloc_bhm_e(num_det_pairs, 1, num_e_bins)

    
    for bicorr_filename in bicorr_filenames:    
        # Set up bicorr file full path
        bicorr_path_full = os.path.join(bicorr_path, bicorr_filename)
        
        # File to process (not going through folders)
        if print_flag: print('Generating bicorr histograms for bicorr data in file: ', bicorr_path_full)    

        # Load files, fill the histograms
        if print_flag: print('Loading data in file ',bicorr_path_full)
        bicorr_data = bicorr.load_bicorr(bicorr_path = bicorr_path_full)
        if checkpoint_flag:
            fig_folder = os.path.join(bicorr_path,'fig')
            bicorr_plot.bicorr_checkpoint_plots(bicorr_data,fig_folder = fig_folder,show_flag=False)
        if print_flag: print('Building bhms in ', bicorr_path)
        bhm = bicorr.fill_bhm(bhm,bicorr_data, det_df, dt_bin_edges, disable_tqdm = disable_tqdm)
        bhm_e = bicorr_e.fill_bhm_e(bhm_e, bicorr_data, det_df, dict_det_dist, e_bin_edges, disable_tqdm = False)
        
    if save_flag: 
        # bhm: Generate sparse matrix
        if print_flag: print('Generating sparse matrix')
        sparse_bhm = bicorr.generate_sparse_bhm(bhm,disable_tqdm = disable_tqdm)        
        if print_flag: print('Saving sparse matrix data to .npz file')
        bicorr.save_sparse_bhm(sparse_bhm, dt_bin_edges, save_folder = bicorr_path)
    
        # bhm_e: store npz
        if print_flag: print('Saving bhm_e to .npz file')
        note = 'bhm_e generated for file'.format(bicorr_path)
        bicorr_e.save_bhm_e(bhm_e, e_bin_edges, save_folder = bicorr_path, note='Original file: '+bicorr_filename)
        
    if print_flag: print('Bicorr hist master bhm, bhm_e build complete')
                
    if return_flag: return bhm, dt_bin_edges, bhm_e, e_bin_edges       
    
    
def build_singles_hist_sim(cced_filenames, cced_path, dict_det_dist = None, plot_flag = True, fig_folder = 'fig', show_flag = False, save_flag = False):
    """
    Parse cced file and generate histogram of singles timing information.
    
    Parameters
    ----------
    cced_filenames : str
        If only one, use list [filename]
    cced_path : str
    dict_det_dist : dict, optional
    plot_flag : bool, optional
    fig_folder : str, optional
    show_flag : bool, optional
    save_flag : bool, optional
    
    Returns
    -------
    singles_hist : ndarray
        Histogram of singles timing information
        Dimension 0: particle type, 0=n, 1=g
        Dimension 1: detector channel
        Dimension 2: dt bin
    dt_bin_edges : ndarray
        Time bin array    
    singles_hist_n_e : ndarray
        Histogram of singles timing information
        Dimension 1: detector channel
        Dimension 2: e bin
    e_bin_edges : ndarray
        Time bin array
    dict_det_to_index : dict
        Dict from detector channel number to index in singles_hist    
    """
    # Build channel lists
    chList, fcList, detList, num_dets, num_det_pairs = bicorr.build_ch_lists()
    if dict_det_dist is None: dict_det_dist = bicorr.build_dict_det_dist()
    
    # Set up histogram: time
    dt_bin_edges, num_dt_bins = bicorr.build_dt_bin_edges(0,300,0.25)
    dt_min = np.min(dt_bin_edges)
    dt_max = np.max(dt_bin_edges)
    dt_step = dt_bin_edges[1]-dt_bin_edges[0]
    singles_hist = np.zeros((2,num_dets,num_dt_bins),dtype=np.uint64)
    # Set up histogram: energy
    e_bin_edges, num_e_bins = bicorr_e.build_energy_bin_edges()
    e_min = np.min(e_bin_edges)
    e_max = np.max(e_bin_edges)
    e_step = e_bin_edges[1]-e_bin_edges[0]
    singles_hist_e_n = np.zeros((num_dets,num_e_bins),dtype=np.uint64)
    
    # Set up det -> index dictionary
    det_indices = np.arange(num_dets)
    dict_det_to_index = dict(zip(detList,det_indices))
    dict_index_to_det = dict(zip(det_indices,detList))    
    
    # Load cced file
    ccedTypeSim = np.dtype([('event', np.int32), 
                        ('detector', np.int8), 
                        ('particle_type', np.int8), 
                        ('time', np.float16), 
                        ('height', np.float32)])
    
    for cced_filename in cced_filenames:
        data = np.genfromtxt(os.path.join(cced_path,cced_filename),dtype=ccedTypeSim,usecols=(0,1,2,3,5))    
        print('Loading data from: ',os.path.join(cced_path,cced_filename))
        
        # Fill histogram- debugging option
        print_flag = False # Make input parameter?

        # l is the line number of the current line, starting at 0.
        # e is the event number of the current line, starting at 1
        # This is a clever way of keeping track what line you're on. Enumerate through the event numbers, `e`, and python also keeps track of the line number `l`.
        for l, e in enumerate(tqdm(data['event'],ascii=True)):
            if print_flag: print("Reading line: ",l,"; event: ",e)
            event = data[l]
            det = event['detector']
            dt = event['time']
            par_type = event['particle_type']
          
            # Store time to histogram        
            if (dt_min < dt < dt_max):
                t_i = int(np.floor((dt-dt_min)/dt_step))
                singles_hist[par_type-1,dict_det_to_index[det],t_i]+= 1
            if print_flag: print('t_i:',t_i)
                    
            # Store to histogram: energy
            if par_type == 1: # Neutrons only
                dist = dict_det_dist[det]
                energy = bicorr_math.convert_time_to_energy(dt,dist)
                
                if (e_min < energy < e_max):
                    e_i = int(np.floor((energy-e_min)/e_step))
                    singles_hist_e_n[dict_det_to_index[det],e_i] += 1
                        
            eventNum = e      # Move onto the next event
            i = l             # Current line is the first line for next event
        
    # Make some plots
    if plot_flag:
        dt_bin_centers = (dt_bin_edges[:-1]+dt_bin_edges[1:])/2
        plt.plot(dt_bin_centers,np.sum(singles_hist[0,:,:],axis=(0)))
        plt.plot(dt_bin_centers,np.sum(singles_hist[1,:,:],axis=(0)))
        plt.xlabel('Time (ns)')
        plt.ylabel('Number of events')
        plt.title('Singles TOF distribution, all channels')
        plt.legend(['N','G'])
        plt.yscale('log')
        bicorr_plot.save_fig_to_folder('singles_TOF_dist.png',fig_folder)
        if show_flag: plt.show()
        plt.clf()  
    
        e_bin_centers = bicorr_math.calc_centers(e_bin_edges)
        plt.plot(e_bin_centers, np.sum(singles_hist_e_n[:,:],axis=(0)))
        plt.xlabel('Energy (MeV)')  
        plt.ylabel('Number of events')
        plt.title('Singles energy distribution, all channels')
        plt.yscale('log')
        bicorr_plot.save_fig_to_folder('singles_e_dist.png',fig_folder)
        if show_flag: plt.show()
        plt.clf()  
    
    # Save to file
    if save_flag:
        # time
        np.savez(os.path.join(cced_path,'singles_hist'),singles_hist=singles_hist, dict_det_to_index=dict_det_to_index, dt_bin_edges = dt_bin_edges)
        # energy        
        np.savez(os.path.join(cced_path,'singles_hist_e_n'),
                 singles_hist_e_n=singles_hist_e_n,dict_det_to_index=dict_det_to_index,
                 e_bin_edges=e_bin_edges)
        
    return singles_hist, dt_bin_edges, singles_hist_e_n, e_bin_edges, dict_det_to_index