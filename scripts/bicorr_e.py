'''
Functions for dealing with bhm in energy space.
'''
import matplotlib
#matplotlib.use('agg') # for flux
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

import sys
import os
import os.path
import scipy.io as sio
import numpy as np

import time
import numpy as np
np.set_printoptions(threshold=np.nan) # print entire matrices
import pandas as pd
from tqdm import *

import bicorr as bicorr
import bicorr_math as bicorr_math
import bicorr_plot as bicorr_plot

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

############### SINGLES HIST ############################
def build_singles_hist_both(cced_filenames, cced_path, dict_det_dist = None, plot_flag = True, fig_folder = 'fig', show_flag = False, save_flag = False, Ethresh = 0.06046):
    """
    Parse cced file and generate histogram of singles timing information. This must be used with a script, assumed folder min and folder max structure from measurements.
    
    Parameters
    ----------
    cced_filenames : str
        List of cced filenames. If one: [filename]
    cced_path : str
    dict_det_dist : dict, optional
    plot_flag : bool, optional
    fig_folder : str, optional
    show_flag : bool, optional
    save_flag : bool, optional
    Ethresh : float, optional
        Pulse height threshold, V (Experiment only. Simulation is MeVee)
        Converting from MeVee where 0.289 Volts at 0.478 MeVee
        Ex: 0.1 MeVee threshold * 0.289 V / 0.478 MeVee = 0.06046 V
    
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
    dt_bin_edges, num_dt_bins = bicorr.build_dt_bin_edges(-300,300,0.25)    
    dt_min = np.min(dt_bin_edges)
    dt_max = np.max(dt_bin_edges)
    dt_step = dt_bin_edges[1]-dt_bin_edges[0]
    num_dt_bins = len(dt_bin_edges)-1
    singles_hist = np.zeros((2,num_dets,num_dt_bins),dtype=np.uint64)
    # Set up histogram: energy
    e_bin_edges, num_e_bins = build_energy_bin_edges()
    e_min = np.min(e_bin_edges)
    e_max = np.max(e_bin_edges)
    e_step = e_bin_edges[1]-e_bin_edges[0]
    singles_hist_e_n = np.zeros((num_dets,num_e_bins),dtype=np.uint64)
    
    # Set up det -> index dictionary
    det_indices = np.arange(num_dets)
    dict_det_to_index = dict(zip(detList,det_indices))
    dict_index_to_det = dict(zip(det_indices,detList))    
    
    # Load cced file
    ccedType = np.dtype([('event', np.int32), ('detector', np.int8), ('particle_type', np.int8), ('time', np.float16), ('integral', np.float32), ('height', np.float32)])
    
    for cced_filename in cced_filenames:
    
        # Import time offset data (assume in same location)
        timeOffsetData = np.genfromtxt(os.path.join(cced_path,str(cced_filename[0]),'timeOffset.txt'))
    
        data = np.genfromtxt(os.path.join(cced_path,cced_filename),dtype=ccedType)
        print('Loading data from: ',os.path.join(cced_path,cced_filename) )
        
        # Fill histogram
        print_flag = False # Make input parameter?

        # l is the line number of the current line, starting at 0.
        # e is the event number of the current line, starting at 1

        # eventNum is the current event number, extending from lines i to j.
        eventNum = data[0]['event']; # Start with the first event in the data chunk.
                                     # If reading entire file, this is 1. 
                                     # If reading a chunk, this may be higher.
        i = 0;                       # First line number of first event is always 0

        # This is a clever way of keeping track what line you're on. Enumerate through the event numbers, `e`, and python also keeps track of the line number `l`.
        for l, e in enumerate(tqdm(data['event'],ascii=True)):
            if print_flag: print("Reading line: ",l,"; event: ",e)
            
            if e == eventNum: # Still on the same event
                pass
            if e != eventNum: # Store info from current event, move onto next event.
                j = l         # Store line number
                n_ints = j-i  # Number interactions in current event
                if print_flag: print(n_ints)
                    
                if n_ints >= 2:# At least two channels
                    ccedEvent = data[i:j][:]   # Data in this event
                    chs_present = ccedEvent[:]['detector']   # What channels triggered?
                    chs_bool = np.in1d(chs_present,detList)  # True = detector, False = fission chamber
                    dets_present = chs_present[chs_bool]
                    fc_corr = (16*np.floor(dets_present/16)).astype(int) # Corr fc for each det ch
                    fc_bool = np.in1d(fc_corr, chs_present) # Did fc corr trigger?
                    
                    if print_flag: print(i,j,ccedEvent)
                    if print_flag: print('Chs:', chs_present,chs_bool,'Dets:',dets_present,fc_corr,fc_bool)
                    
                    if sum(fc_bool) >=1 : # At least one det-fc pair triggered
                        dets_present = dets_present[fc_bool]
                        fc_corr = fc_corr[fc_bool]
                        if print_flag: print(e-1, dets_present, fc_corr)
                    
                        # Set up vectors
                        det_indices = np.zeros(len(dets_present),dtype=np.int8) # det in chs_present
                        fc_indices  = np.zeros(len(fc_corr),dtype=np.int8) # fc in chs_present
                        time_offset = np.zeros(len(dets_present),dtype=np.float16) # time offset
                        
                        for d in range(0,len(dets_present),1):
                            det_indices[d] = np.where(chs_present == dets_present[d])[0]
                            fc_indices[d] = np.where(chs_present == fc_corr[d])[0]
                            time_offset[d] = timeOffsetData[fc_corr[d]][dets_present[d]]
                            if print_flag: print(det_indices, fc_indices, time_offset)
                        
                        # Store dt and particle type for each detector event
                        dt       = ccedEvent[det_indices]['time']-ccedEvent[fc_indices]['time']+time_offset
                        heights = ccedEvent[det_indices]['height']
                        par_type = ccedEvent[det_indices]['particle_type']
                        if print_flag: pass
                        
                        # Store to histogram here! (Filled in later section)
                        for d in np.arange(len(dets_present)): # Loop through verified singles
                            if print_flag: print(d,'of:',len(dt))
                            if print_flag: print(dt[d])
                            if print_flag: print(par_type[d])
                            
                            if heights[d] > Ethresh: # Pulse height threshold check
                                t_i = int(np.floor((dt[d]-dt_min)/dt_step))
                                t_i_check = np.logical_and(t_i>=0, t_i<num_dt_bins) # Within range?
                                if print_flag: print('t_i:',t_i)
                                
                                if t_i_check:
                                    singles_hist[par_type[d]-1,dict_det_to_index[dets_present[d]],t_i]+= 1
                                
                                # Store to energy histogram    
                                if np.logical_and(par_type[d] == 1,dt[d] > 0):
                                    dist = dict_det_dist[dets_present[d]]
                                    energy = bicorr_math.convert_time_to_energy(dt[d],dist)
                                    if (e_min < energy < e_max):
                                        e_i = int(np.floor((energy-e_min)/e_step))
                                        singles_hist_e_n[dict_det_to_index[dets_present[d]],e_i] += 1
           
                                
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

def load_singles_hist_both(filenames=['singles_hist.npz','singles_hist_e_n.npz'], filepath=None, t_flag = False, e_flag = True,plot_flag = False, fig_folder = 'fig', show_flag = False, save_flag = False):
    """
    Load existing singles histogram data. This is super hacky.
    
    Parameters
    ----------
    filenames : str, optional
        filenames of singles_hist  and singles_hist_e_n data
    filepath : str, optional
        location of singles_hist data. If None, in cwd
    t_flag : bool, optional
        Option to load time data
    e_flag : bool, optional
        Option to load energy data
    plot_flag : bool, optional
        Whether to display plots
    fig_folder : str, optional
        Location for saving figure
    show_flag : bool, optional
        Whether to display figure to screen
    save_flag : bool, optional
        Whether to save figs to file
        
    Returns
    -------
    singles_hist : ndarray
        Histogram of singles timing information
        Dimension 0: particle type, 0=n, 1=g
        Dimension 1: detector channel
        Dimension 2: dt bin
    dt_bin_edges : ndarray)
        Time bin edges array
    singles_hist_e_n : ndarray
        Histogram of singles timing information
        Dimension 1: detector channel
        Dimension 2: e bin
    e_bin_edges : ndarray
        Time bin array
    dict_det_to_index : dict
        Dict from detector channel number to index in singles_hist
    """
    if filepath is None:
        if t_flag: npzfile_t = np.load(filenames[0])
        if e_flag: npzfile_e = np.load(filenames[1])
    else:
        if t_flag: npzfile_t = np.load(os.path.join(filepath,filenames[0]))    
        if e_flag: npzfile_e = np.load(os.path.join(filepath,filenames[1]))    
    # Extract time
    if t_flag:
        singles_hist = npzfile_t['singles_hist']
        dict_det_to_index = npzfile_t['dict_det_to_index'][()]
        dict_index_to_det = {v: k for k, v in dict_det_to_index.items()}
        dt_bin_edges       = npzfile_t['dt_bin_edges']
        
    # Extract energy
    if e_flag:
        singles_hist_e_n = npzfile_e['singles_hist_e_n']
        dict_det_to_index = npzfile_e['dict_det_to_index'][()]
        dict_index_to_det = {v: k for k, v in dict_det_to_index.items()}
        e_bin_edges = npzfile_e['e_bin_edges']
    
    # Make some plots
    if plot_flag:
        if t_flag: bicorr_plot.plot_singles_hist(singles_hist,dt_bin_edges,save_flag,fig_folder,show_flag)
        
        if e_flag: bicorr_plot.plot_singles_hist_e_n(singles_hist_e_n,e_bin_edges,save_flag,fig_folder,show_flag)
  
    if (t_flag and e_flag): return singles_hist, dt_bin_edges, singles_hist_e_n, e_bin_edges, dict_det_to_index, dict_index_to_det
    elif t_flag: return singles_hist, dt_bin_edges, dict_det_to_index, dict_index_to_det
    else: return singles_hist_e_n, e_bin_edges, dict_det_to_index, dict_index_to_det
    
######################## BHM_E ##################################
    
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
    dict_pair_to_index, dict_index_to_pair, dict_pair_to_angle = bicorr.build_dict_det_pair(det_df)
    
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

            det1e = bicorr_math.convert_time_to_energy(det1t, det1dist)
            det2e = bicorr_math.convert_time_to_energy(det2t, det2dist)

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
    if det_df is None: det_df = bicorr.load_det_df()
    if dict_det_dist is None: dict_det_dist = bicorr.build_dict_det_dist()
    
    # If no data path provided, look for data folders here
    if root_path is None: root_path = os.getcwd()
    
    # Folders to run
    folders = np.arange(folder_start,folder_end,1)
    if print_flag: print('Generating bicorr histogram for bicorr data in folders: ', folders)    

    # Handle e_bin_edges
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
        bicorr_data = bicorr.load_bicorr(folder, root_path = root_path)
        if checkpoint_flag:
            fig_folder = os.path.join(root_path + '/' + str(folder) + '/fig')
            bicorr_plot.bicorr_checkpoint_plots(bicorr_data,fig_folder = fig_folder,show_flag=False)
        if print_flag: print('Building bhm in folder ',folder)
        bhm_e = fill_bhm_e(bhm_e, bicorr_data, det_df, dict_det_dist, e_bin_edges, disable_tqdm = False)
        
    if save_flag: 
        # Generate sparse matrix
        if print_flag: print('Saving bhm_e to .npz file')
        note = 'bhm_e generated from folder {} to {} in directory {}'.format(folder_start,folder_end,os.getcwd())
        save_bhm_e(bhm_e, e_bin_edges, bhm_e_filename=bhm_e_filename, note=note)
        
    if print_flag: print('Bicorr hist master bhm_e build complete')
                
    if return_flag: return bhm_e, e_bin_edges     
    
    
    
def build_bhm_both(folder_start=1,folder_end=2, det_df = None, dt_bin_edges = None, dict_det_dist = None, e_bin_edges = None, checkpoint_flag = True, save_flag = True, sparse_filename = 'sparse_bhm', bhm_e_filename = 'bhm_e', root_path = None, disable_tqdm = False, print_flag = True, return_flag = False):
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
    
    # If no data path provided, look for data folders here
    if root_path is None: root_path = os.getcwd()
    
    # Folders to run
    folders = np.arange(folder_start,folder_end,1)
    if print_flag: print('Generating bicorr histograms for bicorr data in folders: ', folders)    

    
    # Handle dt_bin_edges
    if dt_bin_edges is None: # Use default settings
        dt_bin_edges, num_dt_bins = bicorr.build_dt_bin_edges()
    else:
        num_dt_bins = len(dt_bin_edges)-1
    # Handle e_bin_edges
    if e_bin_edges is None: # Use default settings
        e_bin_edges, num_e_bins  = build_energy_bin_edges()
    else:
        num_e_bins = len(e_bin_edges)-1
    
    # Set up binning
    num_det_pairs = len(det_df)
    
    # Create bhm, empty- all event types
    bhm = bicorr.alloc_bhm(num_det_pairs, 4, num_dt_bins)    
    # Create bhm_e, empty- nn only
    bhm_e = alloc_bhm_e(num_det_pairs, 1, num_e_bins)
    
    # Loop through each folder and fill the histogram
    for folder in folders:
        if print_flag: print('Loading data in folder ',folder)
        bicorr_data = bicorr.load_bicorr(folder, root_path = root_path)
        if checkpoint_flag:
            fig_folder = os.path.join(root_path + '/' + str(folder) + '/fig')
            bicorr_plot.bicorr_checkpoint_plots(bicorr_data,fig_folder = fig_folder,show_flag=False)
        if print_flag: print('Building bhms in folder ',folder)
        bhm = bicorr.fill_bhm(bhm,bicorr_data, det_df, dt_bin_edges, disable_tqdm = disable_tqdm)
        bhm_e = fill_bhm_e(bhm_e, bicorr_data, det_df, dict_det_dist, e_bin_edges, disable_tqdm = False)
        
    if save_flag: 
        # bhm: Generate sparse matrix
        if print_flag: print('Generating sparse matrix')
        sparse_bhm = bicorr.generate_sparse_bhm(bhm,disable_tqdm = disable_tqdm)        
        if print_flag: print('Saving sparse matrix data to .npz file')
        bicorr.save_sparse_bhm(sparse_bhm, dt_bin_edges, save_folder = root_path, sparse_filename = sparse_filename)
    
        # bhm_e: store npz
        if print_flag: print('Saving bhm_e to .npz file')
        note = 'bhm_e generated from folder {} to {} in directory {}'.format(folder_start,folder_end,os.getcwd())
        save_bhm_e(bhm_e, e_bin_edges, bhm_e_filename=bhm_e_filename, note=note)
        
    if print_flag: print('Bicorr hist master bhm_e build complete')
                
    if return_flag: return bhm, dt_bin_edges, bhm_e, e_bin_edges    
    
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
    
    
#################### BHP_E ######################

def build_bhp_e(bhm_e, e_bin_edges, num_fissions = None,
              pair_is = 'all', print_flag = False):
    """
    Build the bicorr_hist_plot by selecting events from bhm and applying normalization factor. The normalization factor is only applied if norm_factor is provided. If not, norm_factor remains at default value 1 and the units are in number of counts. 
    
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
    num_fissions : float, optional
        Number of fissions for normalization. If provided, then proceed with normalization. If not provided, then no normalization performed
    pair_is : list, optional
        Indices of selected detector pairs in bhm
    print_flag : bool, optional
        If set to True, print out some information about array size and selections
    
    Returns
    -------
    bhp_e : ndarray
        Array to plot. Two-dimensional with axes sizes corresponding to dt_bin_edges x dt_bin_edges.
    norm_factor : float
        Normalization factor to translate to counts per fission-pair
    """
    if print_flag:
        print('Creating bhp_e for...')
        print('pair_is = ',pair_is)  
    
    # If plotting all indices for pair_is, generate those indices
    if pair_is is 'all':
        pair_is = np.arange(0,bhm_e.shape[0])        
        
    # If normalizing, calculate normalization factor
    if num_fissions is None:
        norm_factor = 1 # No normalization
    else:               # Normalize by number fissions, detector pairs, and time bin size
        norm_factor = num_fissions * len(pair_is) * np.power((e_bin_edges[1]-e_bin_edges[0]),2)
        
    # Produce bicorr_hist_plot
    bhp_e = np.sum(bhm_e[pair_is,:,:,:],axis=(0,1)) / norm_factor
    
    if print_flag:
        print('energy bin width (MeV) = ', (e_bin_edges[1]-e_bin_edges[0]))
        print('length of pair_is = ', len(pair_is))
        print('norm_factor = ',norm_factor)               
    
    return bhp_e, norm_factor
    
################# SLICES ######################
def slice_bhp_e(bhp_e, e_bin_edges, Ej_min, Ej_max = None, print_flag = False):
    """
    Produce counts vs. E_i at constant E_j from bhp_e

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
    Ej_min : float
        Energy at which to slice bhp_e- lower boundary
    Ej_max : float
        Energy at which to slice bhp_e- upper boundary. If not provided, only use one bin in which delta_tj_min exists
    print_flag : bool
        Option to print status updates
        
    Returns
    -------
    bhp_e_slice : ndarray
        Slice through bhp at Ej_min
    slice_e_range : list
        Two-element list
        Lower and upper bound of slice energy window
        slice_e_range[0] = lower energy bound, slice_e_range[1] = upper energy bound
    """
    i_Ej_min = np.digitize(Ej_min,e_bin_edges)-1
    
    if Ej_max is None:
        i_Ej_max = i_Ej_min
    else:
        if Ej_max < Ej_min:
            print('ERROR in slice_bhp: Ej_max < Ej_min')
        i_Ej_max = np.digitize(Ej_max,e_bin_edges)-1
        
    bhp_e_slice = (np.sum(bhp_e[i_Ej_min:i_Ej_max+1,:],axis=0) +
                 np.sum(bhp_e[:,i_Ej_min:i_Ej_max+1],axis=1))
    
    slice_e_range = [e_bin_edges[i_Ej_min],e_bin_edges[i_Ej_max+1]]
    
    if print_flag:
        print('Creating slice through bhp_e for energies from {} to {}'.format(slice_e_range[0],slice_e_range[1]))

    return bhp_e_slice, slice_e_range    
    
    
def slices_bhp_e(bhp_e, e_bin_edges, e_slices, e_slice_width = None):
    '''
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
    e_slices : ndarray
        Energy values at which to calculate bhp_e_slices
    e_slice_width : float, optional
        How wide to make enery bins? If None, use width of bin from bhp_e.
    
    Returns
    -------
    bhp_e_slices : ndarray
        Array of bhp_e slices. Dimensions: len(e_slices) x len(e_bin_centers)
    slice_e_ranges : ndarray
        Array of slice_e_ranges. Dimensions: len(e_slices) x 2 (min, max)
    '''
    e_bin_centers = bicorr_math.calc_centers(e_bin_edges)
    bhp_e_slices = np.zeros((len(e_slices),len(e_bin_centers)))
    slice_e_ranges = np.zeros((len(e_slices),2))
    

    
    for e in e_slices:
        i = e_slices.index(e) # Works as long as t_slices is unique
        
        if e_slice_width == None:
            Ej_max = None
        else:
            Ej_max = e+e_slice_width    
        
        bhp_e_slices[i,:], slice_e_ranges[i,:] = slice_bhp_e(bhp_e,e_bin_edges,e,Ej_max)
    
    return bhp_e_slices, slice_e_ranges
    
    
def calc_Eave_slices(bhp_e_slices,e_slices,e_bin_edges,E_min,E_max,norm_factor=1):
    """
    Calculate average energies as calculated from slices
    
    Parameters
    ----------
    bhp_e_slices : ndarray
        Array of bhp_e slices. Dimensions: len(e_slices) x len(e_bin_centers)
    e_slices : ndarray
        Energy values at which to calculate bhp_e_slices   
    e_bin_edges : ndarray
        One-dimensional array of energy bin edges
    E_min : float
        Minimum energy of range over which to calculate average
    E_max : float
        Maximum energy of range over which to calculate avereage
    norm_factor : float
        If providing normalized data
    
    Returns
    -------
    Eave : ndarray
        Average energies calculated
    Eave_err : ndarray
        1-sigma error calculated in Eave
    Ej : ndarray
        Dependent neutron energies
    """
    # Set up arrays to fill
    Eave = np.zeros((len(e_slices)))
    Eave_err = np.zeros((len(e_slices)))
    Ej = np.zeros((len(e_slices)))
    
    i_min = int(np.digitize(E_min,e_bin_edges))
    i_max = int(np.digitize(E_max,e_bin_edges))
    
    for i in range(len(e_slices)):
        e = e_slices[i]
        Eave[i], Eave_err[i] = bicorr_math.calc_histogram_mean(e_bin_edges[i_min:i_max+1], 
                                                               norm_factor*bhp_e_slices[i,i_min:i_max])
        Ej[i] = e
        
    return Eave, Eave_err, Ej
    