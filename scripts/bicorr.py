"""
Read cced and produce bicorr, plots (DNNG FNPC project)
PFS, Nov 2016

Instructions: Can be called from command line or run interactively
  Command line: `python bicorr.py` or `python bicorr.py folder_start folder_end`
  Interactive python session: `import bicorr.py`, then `generate` or `generate(folder_start,folder_end)`
  
Changelog:
2017_01_31 pull timeOffset.txt from subfolder
2017_02_11 add bicorr_hist_master functions
2017_03_01 add bicorr_plot functions, update comment style
2017_03_22 implement sparse matrix storage
2017_06_07 major changes to use det_df instead of dicts
"""

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

from bicorr_plot import * 
from bicorr_sums import *
from bicorr_math import *



############### SET UP SYSTEM INFORMATION ####################################      
def build_ch_lists(print_flag = False):
    """
    Generate and return lists of the channel numbers that correspond to the fission chamber and detector channels. This is built for the Chi-Nu array measurements. If the detector array changes, we need to modify this function.
    
    Note: In the Chi-Nu array, the fission chamber channels correspond to detector channels as follows:
      - fc 0 det 1-15
      - fc 16 det 17-31
      - fc 32 det 33-47
      
    Run with: chList, fcList, detList, num_dets, num_det_pairs = bicorr.build_ch_lists()
    
    Parameters
    ----------
    print_flag : bool, optional
        Print the values of all vectors?
    
    Returns
    -------
    chList : ndarray
        All channels measurements
    fcList : ndarray
        Fission chamber channels
    detList : ndarray
        Detector channels
    num_dets : int
        Number of detector channels
    num_det_pairs : int
        Number of pairs of detectors
    """
    chList  = np.arange(0,48,1)
    fcList  = np.array([0,16,32])
    detList = chList[~np.in1d(chList,fcList)]
    
    num_dets = len(detList)
    num_det_pairs = np.int((num_dets*(num_dets-1))/2)
    
    if print_flag:
        print('Fission chamber channels:', fcList)
        print('Detector channels:', detList)
        print('Number of detectors:', num_dets)
        print('Number of detector pairs:', num_det_pairs)

    return chList, fcList, detList, num_dets, num_det_pairs
    
############### CALC SINGLES RATES FROM CCED #################################
def generate_singles_hist(folder_start=1,folder_end=2, save_flag = True, print_flag=True):
    """
    Build and store singles histogram for cced# files in folders. Add the histograms together across all of the folders.
    
    Parameters
    ----------
    folder_start : int, optional
        First folder
    folder_end : int, optional
        Last folder + 1 (for example, folder_end = 2 will end at folder 1)
    save_flag : bool, optional
        Save singles_hist, dt_bin_edges, and dict_det_to_index to .npz file?
    print_flag : bool, optional
        Print status updates?        
        
    Returns
    -------
    singles_hist : ndarray
        Histogram of singles timing information
        Dimension 0: particle type, 0=n, 1=g
        Dimension 1: detector channel
        Dimension 2: dt bin
    dt_bin_edges : ndarray
        Time bin array
    dict_det_to_index : dict
        Dict from detector channel number to index in singles_hist
    """
    
    # Folders to run
    folders = np.arange(folder_start,folder_end,1)
    if print_flag: print('Generating singles histogram for cced data in folders: ', folders)
    
    # Set up singles_hist
    dt_bin_edges, num_dt_bins = build_dt_bin_edges(-300,300,0.25) # Numbers are hardcoded in build_singles_hist
    chList, fcList, detList, num_dets, num_det_pairs = build_ch_lists(print_flag=False)
    singles_hist = np.zeros((2,num_dets,num_dt_bins),dtype=np.uint64)
    
    # Produce singles_hist in each folder, add to existing singles_hist
    for folder in folders:
        if print_flag: print('Processing singles hist in folder', str(folder))
        cced_filename = 'cced'+str(folder)
        cced_path = str(folder)
        
        singles_hist_folder, dt_bin_edges, dict_det_to_index =  build_singles_hist(cced_filename,cced_path,True,str(folder)+'/fig',save_flag = False)
        
        # Combine to existing histogram
        singles_hist += singles_hist_folder
    
    # Save to file in main folder
    if save_flag:
        if print_flag: print('Saving singles_hist.npz')
        np.savez('singles_hist',singles_hist=singles_hist, dict_det_to_index=dict_det_to_index, dt_bin_edges = dt_bin_edges)
    
    return singles_hist, dt_bin_edges, dict_det_to_index

def build_singles_hist(cced_filename, cced_path, plot_flag = True, fig_folder = 'fig', show_flag = False, save_flag = False):
    """
    Parse cced file and generate histogram of singles timing information.
    
    Parameters
    ----------
    cced_filename : str
    cced_path : str
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
    dict_det_to_index : dict
        Dict from detector channel number to index in singles_hist
    
    """
    
    # Import time offset data (assume in same location)
    timeOffsetData = np.genfromtxt(os.path.join(cced_path,'timeOffset.txt'))
    
    # Build channel lists
    chList, fcList, detList, num_dets, num_det_pairs = build_ch_lists()
    
    # Set up histogram
    dt_bin_edges, num_dt_bins = build_dt_bin_edges(-300,300,0.25) # Numbers are hardcoded in generate_singles_hist
    singles_hist = np.zeros((2,num_dets,num_dt_bins),dtype=np.uint64)
    
    # Set up det -> index dictionary
    det_indices = np.arange(num_dets)
    dict_det_to_index = dict(zip(detList,det_indices))
    dict_index_to_det = dict(zip(det_indices,detList))    
    
    # Load cced file
    ccedType = np.dtype([('event', np.int32), ('detector', np.int8), ('particle_type', np.int8), ('time', np.float16), ('integral', np.float32), ('height', np.float32)])
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

    # Calculate important things about dt_bin_edges
    # Time indices
    dt_min = np.min(dt_bin_edges); dt_max = np.max(dt_bin_edges)
    dt_step = dt_bin_edges[1]-dt_bin_edges[0]
    num_dt_bins = len(dt_bin_edges)-1

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
                    par_type = ccedEvent[det_indices]['particle_type']
                    if print_flag: pass
                    
                    # Store to histogram here! (Filled in later section)
                    for d in np.arange(len(dets_present)): # Loop through verified singles
                        if print_flag: print(d,'of:',len(dt))
                        if print_flag: print(dt[d])
                        if print_flag: print(par_type[d])
                        t_i = int(np.floor((dt[d]-dt_min)/dt_step))
                        t_i_check = np.logical_and(t_i>=0, t_i<num_dt_bins) # Within range?
                        if print_flag: print('t_i:',t_i)
                        
                        if t_i_check:
                            singles_hist[par_type[d]-1,dict_det_to_index[dets_present[d]],t_i]+= 1
                            
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
        save_fig_to_folder('singles_TOF_dist.png',fig_folder)
        if show_flag: plt.show()
        plt.clf()  
    
    # Save to file
    if save_flag:
        np.savez(os.path.join(cced_path,'singles_hist'),singles_hist=singles_hist, dict_det_to_index=dict_det_to_index, dt_bin_edges = dt_bin_edges)
        
    return singles_hist, dt_bin_edges, dict_det_to_index
    
def load_singles_hist(filename='singles_hist.npz',filepath=None,plot_flag = False, fig_folder = 'fig', show_flag = False, save_flag = False):
    """
    Load existing singles histogram data.
    
    Parameters
    ----------
    filename : str, optional
        filename of singles_hist data
    filepath : str, optional
        location of singles_hist data. If None, in cwd
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
    dt_bin_edges : ndarray
        Time bin edges array
    dict_det_to_index : dict
        Dict from detector channel number to index in singles_hist
    """
    if filepath is None:
        npzfile = np.load(filename)
    else:
        npzfile = np.load(os.path.join(filepath,filename))    
   
    singles_hist = npzfile['singles_hist']
    dict_det_to_index = npzfile['dict_det_to_index'][()]
    dict_index_to_det = {v: k for k, v in dict_det_to_index.items()}
    dt_bin_edges       = npzfile['dt_bin_edges']
    
    # Make some plots
    if plot_flag:
        plot_singles_hist(singles_hist,dt_bin_edges,save_flag,fig_folder,show_flag)
  
    

    return singles_hist, dt_bin_edges, dict_det_to_index, dict_index_to_det
        
############### GENERATE BICORR FROM CCED ####################################
def generate_bicorr(folder_start=1,folder_end=2,root_path=None):
    """
	Parse cced files and produce bicorr output file in each folder from folder_start up to (not including) folder_end.     Developed in fnpc\analysis\2016_11_30_pfs_bicorrelation_plot\generate_bicorr_from_cced.ipynb
		
	Reads in cced file, format: event, detector, particle_type, time, integral, height
	Produces bicorr file, format: event, det1ch, det1par, det1t, det2ch, det2par, det2t
    
    Parameters
    ----------
    folder_start : int, optional
        First folder
    folder_end : int, optional
        Last folder + 1 (for example, folder_end = 2 will end at folder 1)
    root_path : int, optional
        Relative path to folder where data folders exist (1, 2, 3, etc.). default = cwd
    
    Returns
    ------- 
    n/a
	"""
    # If no data path provided, look for data folders here
    if root_path is None: root_path = os.getcwd()
    
    # Folders to run
    folders = np.arange(folder_start,folder_end,1)
    print('Generating bicorr file for folders: ', folders)
    
    # Set up formatting for pulling out data (copy from Matthew)
    cced_root = 'cced'
    print_root = 'bicorr'
    ccedType = np.dtype([('event', np.int32), ('detector', np.int8), ('particle_type', np.int8), ('time', np.float16), ('integral', np.float32), ('height', np.float32)])
    
    # Detector info
    chList, fcList, detList, num_dets, num_det_pairs = build_ch_lists()

    # Run through folders
    for folder in folders:    
        # Open timeOffsetData.txt in that folder
        timeOffsetData = np.genfromtxt(os.path.join(root_path + '/' + str(folder)+'/timeOffset.txt'))
    
        cced_open = os.path.join(root_path + '/' + str(folder)+'/'+cced_root+str(folder))
        print('building bicorr in:',folder)
        print('opening file:',cced_open)

        # Try reading in the entire file at once
        # In this case, I don't need to open the file using open() 
        data = np.genfromtxt(cced_open,dtype=ccedType)

        # Open a text file to write to
        print_file = open(os.path.join(root_path + '/' + str(folder)+'/'+print_root+str(folder)),'w')
        print('write to:',print_file)

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
            
            if e != eventNum:                                # Store info from current event, move onto next event.
                j = l                                        # The last index of eventNum is the previous line
                n_ints = j-i                                 # Number interactions in current event
                                                             
                if n_ints > 2:                               # If > 2 interactions in current event
                    ccedEvent = data[i:j][:]                 # Data from this event
                    chs_present = ccedEvent[:]['detector']   # What channels triggered?				
                    chs_bool = np.in1d(chs_present,detList)  # True = detector, False = fission chamber
                                                             
                    if sum(chs_bool)>1:                      # If >2 dets, did corr fc's trigger?
                        dets_present = chs_present[chs_bool] # Which det ch's triggered?
                    
                        fc_corr = (16*np.floor(dets_present/16)).astype(int) # Corr fc for each det ch
                        fc_bool = np.in1d(fc_corr,chs_present)               # Did fc corr trigger?
                        
                            # Bicorrelation events only!1
                        if sum(fc_bool)>1:                   # If >2 det + corr fc, keep those
                            dets_present = dets_present[fc_bool]
                            fc_corr      = fc_corr[fc_bool]
                            
                            det_indices = np.zeros(len(dets_present),dtype=np.int8)    # Where does chs_present = these det chs?
                            fc_indices  = np.zeros(len(fc_corr),dtype=np.int8)         # Where does chs_present = these fc chs?
                            time_offset = np.zeros(len(dets_present),dtype=np.float16) # Store time offset
                            for d in range(0,len(dets_present),1):
                                det_indices[d] = np.where(chs_present == dets_present[d])[0]
                                fc_indices[d]  = np.where(chs_present == fc_corr[d])[0]
                                time_offset[d] = timeOffsetData[fc_corr[d]][dets_present[d]]
                            
                            # Store dt and particle type for each detector event
                            dt       = ccedEvent[det_indices]['time']-ccedEvent[fc_indices]['time']+time_offset
                            par_type = ccedEvent[det_indices]['particle_type']
                            
                            # Write out event info from all detector pairs
                            for d1 in range(0,len(det_indices)-1,1):
                                for d2 in range(d1+1,len(det_indices),1):
                                    print_file.write(str(ccedEvent[0]['event'])
                                        + '  ' + str(dets_present[d1]) + '  ' + str(par_type[d1]) + '  ' + str(dt[d1]) 
                                        + '  ' + str(dets_present[d2]) + '  ' + str(par_type[d2]) + '  ' + str(dt[d2])
                                        + '\n')
                    
                eventNum = e  # Move on to next event
                i = l         # Current line is the first line for next event

        print_file.close()
        
def combine_to_bicorr_all(folder_start=1,folder_end=2):
    """
	Combine all bicorr files from folder folder_start up to (not including) folder_end
    Save as bicorr_all in main folder
	
	This function may not be used after all because bicorr_all ends up being too large to load into memory at one time. Instead, I will create the histogram of data from each bicorr# file and combine the histograms in the end. 
	"""	
    # Folders to run
    folders = np.arange(folder_start,folder_end,1)
    print('Combining bicorr files for folders: ', folders)

    with open('bicorr_all','w') as outfile:
        for folder in folders:
            bicorr_file = os.path.join(str(folder)+'/'+'bicorr'+str(folder))
            print('operating on:',bicorr_file)
            with open(bicorr_file) as infile:
                for line in infile:
                    outfile.write(line)
                    
def load_bicorr(folder_number = None, bicorr_path = None, root_path = None):
    """
    Load a data matrix of bicorr data into current python session
    
    Parameters
    ----------
    folder_number : int, optional
        Folder from which to load bicorr file
        Bicorr filename will be bicorr# (with folder_number)
    bicorr_path : str, optional
        Full path of bicorr file, including filename
    root_path : int, optional
        Relative path to folder where data folders exist (1, 2, 3, etc.). default = cwd
    
    
    Returns
    -------
    bicorr_data : ndarray
        Each element contains the following info for one bicorrelation pair
        Columns are 0: event, np.int32
                    1: det1ch, np.int8
                    2: det1par, np.int8
                    3: det1t, np.float16
                    4: det2ch, np.int8
                    5: det2par, np.int8
                    6: det2t, np.float16
    """
    # If no data path provided, look for data folders here
    if root_path is None: root_path = os.getcwd()
    
    if np.logical_and(folder_number is None,bicorr_path is None):
        print('Error: no mechanism for finding bicorr file provided. Must provide folder_number OR bicorr_path')
    elif np.logical_and(folder_number is not None,bicorr_path is not None):
        print('Error: Conflicting mechanisms for finding bicorr file provided. Must provide folder_number OR bicorr_path')

    # Set up formatting for reading in the bicorr file
    bicorrType = np.dtype([('event', np.int32), ('det1ch', np.int8), ('det1par', np.int8), ('det1t', np.float16), ('det2ch', np.int8), ('det2par', np.int8), ('det2t', np.float16)])

    # If folder_number used, set up filename. For example, folder 1 is '1/bicorr1'
    if bicorr_path is None:
        bicorr_path = os.path.join(root_path + '/' + str(folder_number)+'/bicorr'+str(folder_number))     
    
    # Load it
    bicorr_data = np.genfromtxt(bicorr_path,dtype=bicorrType)

    return bicorr_data
    


########### CONSTRUCT, STORE, LOAD BICORR_HIST_MASTER: TIME ############################
def build_dt_bin_edges(dt_min=-50,dt_max=200,dt_step=0.25,print_flag=False):
    """
    Construct dt_bin_edges for the two-dimensional bicorrelation histograms. 
    
    Use as: dt_bin_edges, num_dt_bins = bicorr.build_dt_bin_edges()
    
    Parameters
    ----------
    dt_min : int, optional
        Lower time boundary
    dt_max : int, optional
        Upper time boundary
    dt_step : float, optional
        Time bin size
    print_flag : bool, optional
        Whether to print array details
        
    Returns
    -------
    dt_bin_edges : ndarray
        One-dimensional array of time bin edges
    num_dt_bins : ndarray
        Number of bins in time dimension
    """
    dt_bin_edges = np.arange(dt_min,dt_max+dt_step,dt_step)
    num_dt_bins = len(dt_bin_edges)-1
    
    if print_flag:
        print('Built array of dt bin edges from', dt_min, 'to', dt_max, 'in', num_dt_bins,'steps of',dt_step,'ns.')
    
    return dt_bin_edges, num_dt_bins
    
    
    
def alloc_bhm(num_det_pairs, num_intn_types, num_dt_bins):
    """
    Preallocate bicorr_hist_master
    
    Four dimensions: num_det_pairs x num_intn_types x dt_nbins x dt_nbins
    Interaction type index:  (0=nn, 1=np, 2=pn, 3=pp)
    
    Parameters
    ----------
    num_det_pairs : int
    num_intn_types : int, optional
    num_dt_bins : int
    
    Returns
    -------
    bicorr_hist_master : ndarray
        Zero-filled bicorr_hist_master    
    """
    bicorr_hist_master = np.zeros((num_det_pairs,num_intn_types,num_dt_bins,num_dt_bins),dtype=np.uint32)

    return bicorr_hist_master    
    
def fill_bhm(bicorr_hist_master, bicorr_data, det_df, dt_bin_edges, disable_tqdm = False):
    """
    Fill bicorr_hist_master. Structure:
        Dimension 0: detector pair, use dictionary `dict_pair_to_index`  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
    
    Must have allocated bicorr_hist_master and loaded bicorr_data
    
    Parameters
    ----------
    bicorr_hist_master : ndarray
        Master histogram of bicorrelation histograms across all detector pairs and interaction types.
        Dimension 0: detector pair, use dictionary `dict_pair_to_index` where pair is (100*det1ch+det2ch)  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
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
    dt_bin_edges : ndarray
        One-dimensional array of time bin edges
    disable_tqdm : bool, optional
        Flag to disable tqdm progress bar
    
    Returns
    -------
    bicorr_hist_master : ndarray
        Same as input, but filled with event information from bicorr_data
    """
    # Set up dictionary for returning detector pair index
    dict_pair_to_index, dict_index_to_pair = build_dict_det_pair(det_df)[0:2]

    # Type index
    dict_type_to_index = {11:0, 12:1, 21:2, 22:3}

    # Time indices
    dt_min = np.min(dt_bin_edges); dt_max = np.max(dt_bin_edges)
    dt_step = dt_bin_edges[1]-dt_bin_edges[0]
    num_dt_bins = len(dt_bin_edges)-1
    
    for i in tqdm(np.arange(bicorr_data.shape[0]),ascii=True,disable = disable_tqdm):
        # What is the corresponding bin number for the four dimensions?
        ## Detector pair index
        pair_i = dict_pair_to_index[bicorr_data[i]['det1ch']*100+bicorr_data[i]['det2ch']]
        ## Event type index
        type_i = dict_type_to_index[bicorr_data[i]['det1par']*10+bicorr_data[i]['det2par']]
        ## Time 1 index
        t1_i = int(np.floor((bicorr_data[i]['det1t']-dt_min)/dt_step))
        t1_i_check = np.logical_and(t1_i>=0, t1_i<num_dt_bins) # Within range?
        ## Time 2 index
        t2_i = int(np.floor((bicorr_data[i]['det2t']-dt_min)/dt_step))
        t2_i_check = np.logical_and(t2_i>=0, t2_i<num_dt_bins) # Within range?
        if np.logical_and(t1_i_check, t2_i_check):    
            # Increment the corresponding bin
            bicorr_hist_master[pair_i,type_i,t1_i,t2_i] += 1 
        # Next part seems to be taking forever.... leave it out for now
        #if np.max(bicorr_hist_master) > 2e9: 
        #    print('Warning: Counts in bicorr_hist_master exceed 2e9. Will face dtype range limitations.')
    
    return bicorr_hist_master
    
    

def build_bhm(folder_start=1,folder_end=2, det_df = None, dt_bin_edges = None, checkpoint_flag = True, save_flag = True, sparse_filename = 'sparse_bhm', root_path = None, disable_tqdm = False, print_flag = True):
    """
    Load bicorr_data from folder's bicorr# file and fill histogram. Loop through folders specified by `folder_start` and `folder_end`. Built for dt_bin_edges generated using default settings in bicorr.build_dt_bin_edges().
    
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
    checkpoint_flag : bool, optional
        Generate checkpoint plots?
    save_flag : bool, optional
        Save sparse matrix to disk?
    sparse_filename : str, optional
        Filename for sparse matrix
    root_path : int, optional
        Relative path to folder where data folders exist (1, 2, 3, etc.). default = cwd
    disable_tqdm : bool, optional
        Flag to disable tqdm progress bar
    print_flag : bool, optional
        Print status updates along the way?
    
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
    """    
    # Load det_df if not provided
    if det_df is None: det_df = load_det_df()
    
    # If no data path provided, look for data folders here
    if root_path is None: root_path = os.getcwd()
    
    # Folders to run
    folders = np.arange(folder_start,folder_end,1)
    if print_flag: print('Generating bicorr histogram for bicorr data in folders: ', folders)    

    # Handle dt_bin_edges
    if dt_bin_edges is None: # Use default settings
        dt_bin_edges, num_dt_bins = build_dt_bin_edges()
    else:
        num_dt_bins = len(dt_bin_edges)-1
    
    # Set up binning
    num_det_pairs = len(det_df)
    num_intn_types = 4
    
    # Create bhm, empty
    bhm = alloc_bhm(num_det_pairs, num_intn_types, num_dt_bins)
    
    # Loop through each folder and fill the histogram
    for folder in folders:
        if print_flag: print('Loading data in folder ',folder)
        bicorr_data = load_bicorr(folder, root_path = root_path)
        if checkpoint_flag:
            fig_folder = os.path.join(root_path + '/' + str(folder) + '/fig')
            bicorr_checkpoint_plots(bicorr_data,fig_folder = fig_folder,show_flag=False)
        if print_flag: print('Building bhm in folder ',folder)
        bhm = fill_bhm(bhm,bicorr_data, det_df, dt_bin_edges, disable_tqdm = disable_tqdm)
        
    if save_flag: 
        # Generate sparse matrix
        if print_flag: print('Generating sparse matrix')
        sparse_bhm = generate_sparse_bhm(bhm,disable_tqdm = disable_tqdm)        
        if print_flag: print('Saving sparse matrix data to .npz file')
        save_sparse_bhm(sparse_bhm, dt_bin_edges, save_folder = root_path, sparse_filename = sparse_filename)
        
    if print_flag: print('Bicorr hist master bhm build complete')
                
    return bhm, dt_bin_edges 
    
def save_bicorr_hist_master(bicorr_hist_master, dict_pair_to_index, dt_bin_edges, save_folder):
    """
    Save bicorr_hist_master, dict_pair_to_index, and dt_bin_edges to .npz file in local folder. Reload using load_bicorr_hist_master function. (NOTE: This method outdated. Improved, requires much less storage space: sparse_bhm)
    
    Parameters
    ----------
    bicorr_hist_master : ndarray
        2d bicorrelation histogram
        Dimension 0: detector pair, use dictionary `dict_pair_to_index`  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
    dict_pair_to_index : dict
        Maps detector pair index (100*det1ch+det2ch) to the index of that detector pair in bicorr_hist_master 
    dt_bin_edges : ndarray
        Edges of time bin array in ns
    save_folder : str, optional
        Optional destination folder. If None, then save in current working directory
    """
    print('WARNING: This save method is outdated. Recommend using sparse_bhm method instead.')
    if save_folder is None:
        filename = 'bicorr_hist_master'
    else:
        # check if save_folder exists
        try:
            os.stat(save_folder)
        except:
            os.mkdir(save_folder)
        filename = os.path.join(save_folder, 'bicorr_hist_master')    
    
    np.savez(filename, dict_pair_to_index=dict_pair_to_index, dt_bin_edges = dt_bin_edges, bicorr_hist_master=bicorr_hist_master)
    
def load_bicorr_hist_master(filepath = None):
    """
    WARNING: This method is outdated. Use sparse matrix instead. 
    
    Load data from bicorr_hist_master.npz file into current python session. Data should include `bicorr_hist_master`, `dict_pair_to_index`, and `dt_bin_edges`.
    
    Parameters
    ----------
    filepath : str, optional
        May provide a path where the bicorr_hist_master.npz and .npy files are located. Otherwise, expect that the files are in cwd. Provide path as '../path', etc.
        
    Returns
    -------
    bicorr_hist_master : ndarray
        2d bicorrelation histogram
        Dimension 0: detector pair, use dictionary `dict_pair_to_index`  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
    dict_pair_to_index : dict
        Maps detector pair index (100*det1ch+det2ch) to the index of that detector pair in bicorr_hist_master 
    dt_bin_edges : ndarray
        Edges of time bin array in ns
    """
    print('WARNING: This save method is outdated. Recommend using sparse_bhm method instead.')
    if filepath is None:
        npzfile = np.load('bicorr_hist_master.npz')
    else:
        npzfile = np.load(filepath+r'\bicorr_hist_master.npz')    
   
    bicorr_hist_master = npzfile['bicorr_hist_master']
    dict_pair_to_index = npzfile['dict_pair_to_index'][()]
    dt_bin_edges       = npzfile['dt_bin_edges']

    return bicorr_hist_master, dict_pair_to_index, dt_bin_edges
    

def coarsen_bhm(bhm,dt_bin_edges,C,print_flag = False):
    """
    Make the time binning on bicorr_hist_master more coarse by a constant factor.
    
    Parameters
    ----------
    bhm : ndarray
        2d bicorrelation histogram
        Dimension 0: detector pair, use dictionary `dict_pair_to_index`  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
    dt_bin_edges : ndarray
        Time bin edges for bhm
    C : int    
        Coarsening factor
    print_flag : bool
        Option to print status updates
    
    Return
    ------
    bicorr_hist_master_coarse : ndarray
    dt_bin_edges_coarse : ndarray
    """
    shape = bhm.shape
    if print_flag: print('Dimensions of bicorr_hist_master: ', bhm.shape)
    bhm_coarse = np.zeros((shape[0],shape[1],int(shape[2]/C),int(shape[3]/C)))

    if print_flag: print('Width of time bin in (ns): ', dt_bin_edges[1]-dt_bin_edges[0])
    
    # Calculate new dt_bin_edges
    dt_bin_edges_coarse = dt_bin_edges[0::C]
    dt_bin_width_coarse = dt_bin_edges_coarse[1]-dt_bin_edges_coarse[0]
    if print_flag: print('Width of coarse time bin in (ns): ', dt_bin_width_coarse)
    
    if print_flag: print('Condensing bhm from shape ', shape, ' to ', bhm_coarse.shape)
    
    for bin1 in np.arange(0,bhm_coarse.shape[2]):
        for bin2 in np.arange(0,bhm_coarse.shape[3]):
            bhm_coarse[:,:,bin1,bin2] = np.sum(bhm[:,:,C*bin1:C*(bin1+1),C*bin2:C*(bin2+1)],axis=(2,3))
            
    return bhm_coarse, dt_bin_edges_coarse
    
def coarsen_bhp(bhp, dt_bin_edges, C, normalized = False, print_flag = False):
    """
    Make the time binning on bicorr_hist_plot more coarse by a constant factor.
    
    Parameters
    ----------
    bhp : ndarray
        bicorr_hist_plot. Two-dimensional with axes sizes corresponding to dt_bin_edges x dt_bin_edges.
    dt_bin_edges : ndarray
        Time bin edges for bhp
    C : int
        Coarsening factor
    normalized : bool
        Is the data normalized?
    print_flag : bool
        Option to print status updates
        
    Returns
    -------
    bhp_coarse : ndarray
        bicorr_hist_plot_coarse. bhp scaled by C in time bins.
    dt_bin_edges_coarse : ndarray
        Time bin edges for bhp_coarse
    """
    shape = bhp.shape
    if print_flag: print('Dimensions of bicorr_hist_plot: ', bhp.shape)

    dt_bin_width = dt_bin_edges[1]-dt_bin_edges[0]
    if print_flag: print('Width of time bin in (ns): ', dt_bin_width)
    
    # Preallocate coarse matrix
    bhp_coarse = np.zeros((int(shape[0]/C),int(shape[1]/C)))

    # Calculate new dt_bin_edges
    dt_bin_edges_coarse = dt_bin_edges[0::C]
    dt_bin_width_coarse = dt_bin_edges_coarse[1]-dt_bin_edges_coarse[0]
    if print_flag: print('Width of coarse time bin in (ns): ', dt_bin_width_coarse)

    if print_flag: print('Condensing bhp from shape ', shape, ' to ', bhp_coarse.shape)
    
    for bin1 in np.arange(0,bhp_coarse.shape[0]):
        for bin2 in np.arange(0,bhp_coarse.shape[1]):
            bhp_coarse[bin1,bin2] = np.sum(bhp[C*bin1:C*(bin1+1),C*bin2:C*(bin2+1)],axis=(0,1))
            
    if normalized:
        bhp_coarse = bhp_coarse / C**2 
            
    return bhp_coarse, dt_bin_edges_coarse
    

########## SPARSE BHM ############################
    
def generate_sparse_bhm(bicorr_hist_master, disable_tqdm = False):
    """
    Generate sparse version of bicorr_hist_master for storing to file. 
    
    Parameters
    ----------
    bicorr_hist_master : ndarray
        2d bicorrelation histogram
        Dimension 0: detector pair, use dictionary `dict_pair_to_index`  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
    disable_tqdm : bool, optional
        Flag to disable tqdm progress bar
        
    Returns
    -------
    sparse_bhm : ndarray
        Length is equal to the number of nonzero elements in bicorr_hist_master
        dType = ('pair_i', np.uint16)  Indices in bicorr_hist_master
                ('type_i', np.uint8)    ''
                ('det1t_i', np.uint16)  ''
                ('det2t_i', np.uint16)  ''
                ('count', np.uint32)]) Number of counts    
    """
    # Which elements in bicorr_hist_master are nonzero?
    # Note: These steps are taking a long time. I should do some time tests to see what is the slowest step.
    num_nonzero = np.count_nonzero(bicorr_hist_master)
    i_nonzero = np.nonzero(bicorr_hist_master)
    counts = bicorr_hist_master[i_nonzero]

    # Set up sparse_bhm
    sparseType = np.dtype([('pair_i', np.uint16), ('type_i', np.uint8), ('det1t_i', np.uint16), ('det2t_i', np.uint16), ('count', np.uint32)])
    sparse_bhm = np.zeros(num_nonzero,dtype=sparseType)
    
    # Fill sparse_bhm
    for i in tqdm(np.arange(0,num_nonzero),ascii=True,disable=disable_tqdm):
        sparse_bhm[i]['pair_i']  = i_nonzero[0][i]
        sparse_bhm[i]['type_i']  = i_nonzero[1][i]
        sparse_bhm[i]['det1t_i'] = i_nonzero[2][i]
        sparse_bhm[i]['det2t_i'] = i_nonzero[3][i]
        sparse_bhm[i]['count']   = counts[i]    
    
    return sparse_bhm
    
def save_sparse_bhm(sparse_bhm, dt_bin_edges, save_folder = None, sparse_filename = 'sparse_bhm', note = 'note'):
    """
    Save sparse_bhm, and dt_bin_edges to .npz file in local folder. (Reload using load_sparse_bhm function)
    
    Parameters
    ----------
    sparse_bhm : ndarray
        Length is equal to the number of nonzero elements in bicorr_hist_master
        dType = ('pair_i', np.uint16)  Indices in bicorr_hist_master
                ('type_i', np.uint8)    ''
                ('det1t_i', np.uint16)  ''
                ('det2t_i', np.uint16)  ''
                ('count', np.uint32)]) Number of counts 
    dt_bin_edges : ndarray
        Edges of time bin array in ns
    save_folder : str, optional
        Optional destination folder. If None, then save in current working directory
    sparse_filename : str, optional
        Filename to save sparse_bhm to
    note : str, optional
        Note to include in sparse_bhm.npz
    """
    if save_folder is not None:
        # check if save_folder exists
        try:
            os.stat(save_folder)
        except:
            os.mkdir(save_folder)
        sparse_filename = os.path.join(save_folder, sparse_filename)    
    
    np.savez(sparse_filename, dt_bin_edges = dt_bin_edges, sparse_bhm=sparse_bhm, note = note)
    
def load_sparse_bhm(filepath=None,filename=None):
    """
    Load .npz file containing `sparse_bhm`, `dict_pair_to_index`, and `dt_bin_edges`. This file was probably generated by the function `save_sparse_bhm`.
    
    Parameters
    ----------    
    filepath : str, optional
        Where is the `sparse_bhm.npz` file located? If = None, look for it in the current working directory
    filename : str, optional
        What is the name of the sparse_bhm file, if not sparse_bhm.npz?    
    
    Returns
    -------
    sparse_bhm : ndarray
        Length is equal to the number of nonzero elements in bicorr_hist_master
        dType = ('pair_i', np.uint16)  Indices in bicorr_hist_master
                ('type_i', np.uint8)    ''
                ('det1t_i', np.uint16)  ''
                ('det2t_i', np.uint16)  ''
                ('count', np.uint32)]) Number of counts 
    dt_bin_edges : ndarray
        Edges of time bin array in ns    
    note : str
        Note about sparse_bhm file
    """
    if filename is None:
        filename = 'sparse_bhm.npz'
    
    if filepath is None:
        npzfile = np.load(filename)
    else:
        npzfile = np.load(os.path.join(filepath,filename))    
   
    sparse_bhm = npzfile['sparse_bhm']
    dt_bin_edges = npzfile['dt_bin_edges']  
    if 'note' in npzfile:
        note = npzfile['note']
    else:
        note = 'note'
    
    
    return sparse_bhm, dt_bin_edges, note
    
def revive_sparse_bhm(sparse_bhm, det_df, dt_bin_edges, bhm = None):
    """
    Revive sparse_bhm and generate full-sized bicorr_hist_master
    
    Parameters
    ----------
    sparse_bhm : ndarray
        Length is equal to the number of nonzero elements in bicorr_hist_master
        dType = ('pair_i', np.uint16)  Indices in bicorr_hist_master
                ('type_i', np.uint8)    ''
                ('det1t_i', np.uint16)  ''
                ('det2t_i', np.uint16)  ''
                ('count', np.uint32)]) Number of counts 
    det_df : pandas dataFrame
        dataFrame of detector pair indices and angles
    dt_bin_edges : ndarray
        Edges of time bin array in ns   
    bhm : ndarray, optional
        2d bicorrelation histogram
        Dimension 0: detector pair, use dictionary `dict_pair_to_index`  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2      
    
    Returns
    -------
    bhm : ndarray
        2d bicorrelation histogram
        Dimension 0: detector pair, use dictionary `dict_pair_to_index`  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2
    """
    # If no bicorr_hist_master provided, fill a new one
    if bhm is None:
        bhm = alloc_bhm(len(det_df),4,len(dt_bin_edges)-1)
    
    for i in np.arange(0,sparse_bhm.size):
        bhm[sparse_bhm[i][0],sparse_bhm[i][1],sparse_bhm[i][2],sparse_bhm[i][3]] += sparse_bhm[i][4]    
    
    return bhm
    
    

############## DET_DF AND DETECTOR PAIR ANGLES ########################################
def load_det_df(filepath=r'../meas_info/det_df_pairs_angles.csv', remove_fc_neighbors = False, plot_flag = False):
    """
    Load pandas dataFrame containing detector pair information and angles. This was created in the notebook `detector_pair_angles`.
    
    Parameters
    ----------
    filepath : str, optional
        Path (absolute or relative) to det_df file. May be `det_df.csv` or `det_df.csv`.
        Default location is specific to pfschus folder structure
    remove_fc_neighbors : bool, optional
        Option to remove fission chamber neighbors and reset indices
    plot_flag : bool, optional
        Option to produce plots displaying basic structure of det_df
        Plots will be displayed but not stored
    
    Returns
    -------
    det_df : pandas dataFrame
        dataFrame of detector pair indices and angles    
    """
    # What kind of file is it? csv or pickle?
    if filepath[-3:] == 'csv':
        det_df = pd.read_csv(filepath)
    elif filepath[-3:] == 'pkl':
        det_df = pd.read_pickle(filepath)
    else:
        print('ERROR: File type not recognized')
        det_df = 'none'        
    
    if remove_fc_neighbors:
        pair_is = generate_pair_is(det_df,ignore_fc_neighbors_flag=True)
        det_df = det_df.loc[pair_is].reset_index().rename(columns={'index':'index_og'}).copy() 
    if plot_flag: plot_det_df(det_df, show_flag = True)
    
    return det_df
        
def d1d2_index(det_df,d1,d2):
    """
    Return the index of a given detector pair from det_df.
    
    Parameters
    ----------
    det_df : pandas dataFrame
        dataFrame of detector pair indices and angles
    d1 : int
        detector 1 channel
    d2 : int
        detector 2 channel
    
    Returns
    -------
    ind : int
        index of that detector pair in det_df    
    """
    if d2 < d1:
        print('Warning: d2 < d1. Channels inverted')
        # swap numbers
        a = d1; d1 = d2; d2 = a
        
    ind = det_df.index[(det_df['d1']==d1) & (det_df['d2']==d2)][0]
    
    return ind
        
def build_dict_det_pair(det_df):
    """
    Build the dictionary that converts from detector pair to index and angle
    
    Parameters
    ----------
    det_df : pandas dataFrame
        dataFrame of detector pair indices and angles   
    
    Returns
    -------
    dict_pair_to_index : dict
        keys: detector pair indices (100*det1ch+det2ch)
        values: index of pair in bicorr_hist_master (along first axis)
    dict_index_to_pair : dict
        Reverse version of dict_pair_to_index
    dict_pair_to_angle : dict
        keys: detector pair indices (100*det1ch+det2ch)
        values : angle of pair
    """
    dict_index_to_pair = det_df['d1d2'].to_dict()
    dict_pair_to_index = {v: k for k, v in dict_index_to_pair.items()}  

    dict_pair_to_angle = pd.Series(det_df['angle'].values,index=det_df['d1d2']).to_dict()
    
    return dict_pair_to_index, dict_index_to_pair, dict_pair_to_angle        
    

###########################################################################################

    
def generate_pair_is(det_df, th_min = None, th_max = None, i_bin = None, ignore_fc_neighbors_flag = False):
    """
    Generate list of indices of pairs within a given angle range (th_min,th_max] for bicorr_hist_master.
    
    Parameters
    ----------
    det_df : pandas dataFrame
        dataFrame of detector pair indices and angles 
    th_min : int, optional
        Exclusive lower limit (th > th_min)
    th_max : int, optional
        Inclusive upper limit (th <= th_max)  
    i_bin : int, optional
        Index of desired bin in det_df['bin']
    ignore_fc_neighbors_flag : bool, optional
        Whether to ignore channels next to fission chamber [1,17,33]
    
    Return
    ------
    pair_is : list
        Indices of detector pairs in range in bicorr_hist_master
    """
    
    # What are the conditions?    
    by_th_flag = np.logical_and(th_min is not None,th_max is not None) # True: by th range. False: by bins
    by_bin_flag = (i_bin is not None) # True: by bin range. False: by th range
    
    # If you want to ignore the channels next to the fission chamber [1,17,33], remove those lines from det_df
    if ignore_fc_neighbors_flag:
        chIgnore = [1,17,33]
        det_df = det_df[~det_df['d1'].isin(chIgnore) & ~det_df['d2'].isin(chIgnore)]
    
    if by_th_flag:
        ind_mask = (det_df['angle'] > th_min) & (det_df['angle'] <= th_max)
        pair_is = det_df.index[ind_mask].values
    elif by_bin_flag:
        pair_is = det_df[det_df['bin']==i_bin].index.values
    else: # Select all pairs, no constraints
        pair_is = det_df.index.values
            
    return pair_is
    
def unpack_dict_pair_to_angle(dict_pair_to_angle):
    """
    From the dictionary that connects the detector pair (100*det1ch+det2ch) to the corresponding angle, return vectors of det1ch, det2ch, and angle. The returned vectors are in order of increasing det1ch, then det2ch, and elements at the same index correspond across the three arrays (angle[i] is the angle between det1ch[i] and det2ch[i]).
    
    Parameters
    ----------
    dict_pair_to_angle : dict
        Dictionary mapping detector pair to angle
        keys: Detector pair (100*det1ch+det2ch)
        values: Angle between detectors in degrees
    
    Returns
    -------
    det1ch : ndarray of ints
        Channel of detector 1
    det2ch : ndarray of ints
        Channel of detector 2
    angle : ndarray of floats
        Angle between detectors in degrees
    """
    det_pairs = sorted(list(dict_pair_to_angle.keys()))
    det1ch = []
    det2ch = []
    angle = []

    for pair in det_pairs:
        det1ch.append(int(np.floor(np.divide(pair,100))))
        det2ch.append(int(pair-100*det1ch[-1]))
        angle.append(dict_pair_to_angle[pair])

    # Convert to array 
    det1ch = np.asarray(det1ch)
    det2ch = np.asarray(det2ch)
    angle = np.asarray(angle)
        
    return det1ch, det2ch, angle
    
    
def build_bhp(bhm, dt_bin_edges, num_fissions = None,
              pair_is = 'all', type_is = 'all', print_flag = False):
    """
    Build the bicorr_hist_plot by selecting events from bhm and applying normalization factor. The normalization factor is only applied if norm_factor is provided. If not, norm_factor remains at default value 1 and the units are in number of counts. 
    
    Parameters
    ----------
    bhm : ndarray
        Master histogram of bicorrelation histograms across all detector pairs and interaction types.
        Dimension 0: detector pair, use dictionary `dict_pair_to_index` where pair is (100*det1ch+det2ch)  
        Dimension 1: interaction type, length 4. (0=nn, 1=np, 2=pn, 3=pp)  
        Dimension 2: dt bin for detector 1
        Dimension 3: dt bin for detector 2 
    dt_bin_edges : ndarray
        One-dimensional array of time bin edges
    num_fissions : float, optional
        Number of fissions for normalization. If provided, then proceed with normalization. If not provided, then no normalization performed
    pair_is : list, optional
        Indices of selected detector pairs in bhm
    type_is : list, optional  
        Indices of selected interaction types in bhm (0=nn, 1=np, 2=pn, 3=pp) 
    print_flag : bool, optional
        If set to True, print out some information about array size and selections
    
    Returns
    -------
    bicorr_hist_plot : ndarray
        Array to plot. Two-dimensional with axes sizes corresponding to dt_bin_edges x dt_bin_edges.
    norm_factor : float
        Normalization factor to translate to counts per fission-pair
    """
    if print_flag:
        print('Creating bicorr_hist_plot for...')
        print('pair_is = ',pair_is)
        print('type_is = ',type_is)    
    
    # If plotting all indices for pair_is or type_is, generate those indices
    if pair_is is 'all':
        pair_is = np.arange(0,bhm.shape[0])
    if type_is is 'all':
        type_is = np.arange(0,bhm.shape[1])   
        
        
    # If normalizing, calculate normalization factor
    if num_fissions is None:
        norm_factor = 1 # No normalization
    else:               # Normalize by number fissions, detector pairs, and time bin size
        norm_factor = num_fissions * len(pair_is) * np.power((dt_bin_edges[1]-dt_bin_edges[0]),2)
        
    # Produce bicorr_hist_plot
    bicorr_hist_plot = np.sum(bhm[pair_is,:,:,:][:,type_is,:,:],axis=(0,1)) / norm_factor
    
    if print_flag:
        print('time bin width (ns) = ', (dt_bin_edges[1]-dt_bin_edges[0]))
        print('length of pair_is = ', len(pair_is))
        print('norm_factor = ',norm_factor)               
    
    return bicorr_hist_plot, norm_factor
    
def slice_bhp(bhp, dt_bin_edges, delta_tj_min, delta_tj_max = None, print_flag = False):
    """
    Produce counts vs. \Delta t_i at constant \Delta t_j from bhp

    Parameters
    ----------
    bhp : ndarray
        Array to plot. Two-dimensional with axes sizes corresponding to dt_bin_edges x dt_bin_edges.
    dt_bin_edges : ndarray
        One-dimensional array of time bin edges
    delta_tj_min : float
        Time at which to slice bhp- lower boundary
    delta_tj_max : float
        Time at which to slice bhp- upper boundary. If not provided, only use one bin in which delta_tj_min exists
    print_flag : bool
        Option to print status updates
        
    Returns
    -------
    bhp_slice : ndarray
        Slice through bhp at delta_tj_min
    slice_dt_range : list
        Two-element list
        Lower and upper bound of slice time window
        slice_dt_range[0] = lower time bound, slice_dt_range[1] = upper time bound
    """
    i_tj_min = np.digitize(delta_tj_min,dt_bin_edges)-1
    
    if delta_tj_max is None:
        i_tj_max = i_tj_min
    else:
        if delta_tj_max < delta_tj_min:
            print('ERROR in slice_bhp: t_max < t_min')
        i_tj_max = np.digitize(delta_tj_max,dt_bin_edges)-1
        
    bhp_slice = (np.sum(bhp[i_tj_min:i_tj_max+1,:],axis=0) +
                 np.sum(bhp[:,i_tj_min:i_tj_max+1],axis=1))
    
    slice_dt_range = [dt_bin_edges[i_tj_min],dt_bin_edges[i_tj_max+1]]
    
    if print_flag:
        print('Creating slice through bhp for times from {} to {}'.format(slice_dt_range[0],slice_dt_range[1]))

    return bhp_slice, slice_dt_range

    
def slices_bhp(bhp, dt_bin_edges, t_slices):
    '''
    Parameters
    ----------
    bhp : ndarray
        Bicorr hist plot. Two-dimensional with axes sizes corresponding to dt_bin_edges x dt_bin_edges.
    dt_bin_edges : ndarray
        One-dimensional array of time bin edges
    t_slices : ndarray
        Time values at which to calculate bhp_slices
    
    Returns
    -------
    bhp_slices : ndarray
        Array of bhp slices. Dimensions: len(t_slices) x len(dt_bin_centers)
    slice_dt_ranges : ndarray
        Array of slice_dt_ranges. Dimensions: len(t_slices) x 2 (min, max)
    '''
    dt_bin_centers = calc_centers(dt_bin_edges)
    bhp_slices = np.zeros((len(t_slices),len(dt_bin_centers)))
    slice_dt_ranges = np.zeros((len(t_slices),2))
    
    for t in t_slices:
        i = t_slices.index(t) # Works as long as t_slices is unique
        bhp_slices[i,:], slice_dt_ranges[i,:] = slice_bhp(bhp,dt_bin_edges,t)
    
    return bhp_slices, slice_dt_ranges

    
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
    
def calc_nn_sum(bicorr_hist_plot, dt_bin_edges, emin = 0.62, emax = 12, return_real_energies_flag = False):
    """
    Calculate the total number of neutron-neutron counts in the bicorrelation plot.
    
    Parameters
    ----------
    bicorr_hist_plot : ndarray
        Array to plot. Two-dimensional with axes sizes corresponding to dt_bin_edges x dt_bin_edges.
    dt_bin_edges : ndarray
        One-dimensional array of time bin edges
    emin : float
        Lower energy boundary for neutron event selection in MeV
        Default 0.62 MeV ~ 70 keVee
    emax : float, optional
        Upper energy boundary for neutron event selection in MeV
    
    Returns
    -------
    nn_sum : float
        Number of neutron counts (same units as bicorr_hist_plot)
    indices : list, optional
        Indices corresponding to sum
    energies_real : list, optional
        Actual energy limits used in calculation (due to discrete binning)
    """
    
    # Find the corresponding lower and upper time boundaries
    tmin = convert_energy_to_time(emax)
    tmax = convert_energy_to_time(emin)

    # Find bins
    # Energy boundaries are rounded down. Time boundaries are rounded up. 
    i_min = np.min(np.argwhere(tmin < dt_bin_edges))
    i_max = np.min(np.argwhere(tmax < dt_bin_edges))
    indices = [i_min, i_max]
    
    # What are the energy bin limits that correspond to the bins?
    emin_real = convert_time_to_energy(dt_bin_edges[i_min])
    emax_real = convert_time_to_energy(dt_bin_edges[i_max])
    energies_real = [emin_real,emax_real]
    
    nn_sum = np.sum(bicorr_hist_plot[i_min:i_max, i_min:i_max])
    
    if return_real_energies_flag:    
        return nn_sum, indices, energies_real
    else: 
        return nn_sum
    
def calc_nn_sum_br(bhp_nn_pos, bhp_nn_neg, dt_bin_edges_pos, norm_factor=None, emin=0.62, emax=12, return_real_energies_flag = False):
    """
    Calculate the number of counts in a given time range after background subtraction.
    
    Parameters
    ---------
    bhp_nn_pos : ndarray
        Positive time range nn bicorr_hist_plot
    bhp_nn_neg : ndarray
        Negative time range nn bicorr_hist_plot (not flipped)
    dt_bin_edges_pos : ndarray
        One-dimensional array of time bin edges
        (Negative array should have same bin edges but negative)
    norm_factor : float, optional
        Normalization factor to translate to counts per fission-pair
        If None, then bhp is in number of counts, not normalized
    emin : float
        Lower energy boundary for neutron event selection in MeV
        Default 0.62 MeV ~ 70 keVee
    emax : float, optional
        Upper energy boundary for neutron event selection in MeV
    
    
    Returns
    -------
    Cp or Np : int or float
        Positive counts
        Normalized if norm_factor provided as input
    Cn or Nn : int or float
        Negative counts
        Normalized if norm_factor provided as input
    Cd or Nd : int or float
        Background-subtracted counts.
        Normalized if norm_factor provided as input
    Cd_err or Nd_err
        1-sigma error in background-subtracted counts
        Normalized if norm_factor provided as input
    energies_real
        Actual energy bin limits (due to discrete energy bins)
    """
    if norm_factor is None:
        Cp, indices, energies_real = calc_nn_sum(bhp_nn_pos, dt_bin_edges_pos, emin, emax, return_real_energies_flag = True)
        Cn = calc_nn_sum(bhp_nn_neg[::-1,::-1], dt_bin_edges_pos, emin, emax, return_real_energies_flag = True)[0]
        Cd = Cp-Cn
        Cd_err = np.sqrt(Cp+Cn)        
        if return_real_energies_flag: 
            return Cp, Cn, Cd, Cd_err, energies_real
        else: 
            return Cp, Cn, Cd, Cd_err
    else:
        Np, indices, energies_real = calc_nn_sum(bhp_nn_pos, dt_bin_edges_pos, emin, emax, return_real_energies_flag = True)
        Nn = calc_nn_sum(bhp_nn_neg[::-1,::-1], dt_bin_edges_pos, emin, emax, return_real_energies_flag = True)[0]
        Nd = Np-Nn
        Nd_err = np.sqrt((Np+Nn)/norm_factor)
        if return_real_energies_flag:
            return Np, Nn, Nd, Nd_err, energies_real
        else:
            return Np, Nn, Nd, Nd_err
        
def calc_n_sum_br(singles_hist, dt_bin_edges_sh, det_i, emin=0.62, emax=12):
    """
    Calculate background-subtracted number of neutron events within a given time range in the singles histogram. Analogous to calc_nn_sum and calc_nn_sum_br for singles events.
    
    NOTE: I AM NOT NORMALIZING THIS. PLAN ACCORDINGLY WHEN USING TOGETHER WITH CALC_NN_SUM
    
    Parameters
    ----------
    singles_hist : ndarray
        Histogram of singles timing information
        Dimension 0: particle type, 0=n, 1=g
        Dimension 1: detector channel
        Dimension 2: dt bin    
    dt_bin_edges_sh : ndarray
        Singles time bin edges array
    det_i : int
        Index of detector in singles_hist. Use dict_det_to_index from `load_singles_hist`
    emin : float
        Lower energy boundary for neutron event selection in MeV
        Default 0.62 MeV ~ 70 keVee
    emax : float, optional
        Upper energy boundary for neutron event selection in MeV    
    
    Returns
    -------
    Sp : int
        Positive counts
    Sn : int
        Negative counts
    Sd : int
        Background-subtracted counts.
    Sd_err : int
        1-sigma error in background-subtracted counts
    """
    # Calculate time window indices
    # Positive time range
    tmin = convert_energy_to_time(emax)
    tmax = convert_energy_to_time(emin)
    i_min = np.min(np.argwhere(tmin<dt_bin_edges_sh))
    i_max = np.min(np.argwhere(tmax<dt_bin_edges_sh))
    # Negative time range
    i_min_neg = np.min(np.argwhere(-tmax<dt_bin_edges_sh))
    i_max_neg = np.min(np.argwhere(-tmin<dt_bin_edges_sh))
    
    # Add optional plotting later?
    
    Sp = np.sum(singles_hist[[0],det_i,i_min:i_max])
    Sn = np.sum(singles_hist[[0],det_i,i_min_neg:i_max_neg])
    Sd = Sp-Sn
    
    Sd_err = np.sqrt(Sp+Sn)        
    return Sp, Sn, Sd, Sd_err
    
    
        
    
####### MAIN ###########################
def main(folder_start = 1,folder_end = 2, option = [1,2]):
    """
    NOTE: For now only build bicorr#, don't build bicorr_hist_master
    
    Main function to run. Can be called within interactive session or directly from command land.
    
	If I call this `.py` file directly (not from an interactive session), it will run all steps in the bicorr process for the folders specified. It will produce a bicorr# file in each folder and a bicorr_hist_master .npy array in the main folder.
    
    Syntax: "python bicorr.py" OR "python bicorr.py folder_start folder_end"
    Default is to only run folder 1   


    
    Parameters
    ----------
    folder_start : int, optional

        Folder to start with
    folder_end : int, optional
        Folder to end at (don't run this one)
    option : ndarray, optional
        Which step(s) in the analysis to run?
        
    Returns
    -------
    n/a
    """    
    if 1 in option:
        print('********* Generate bicorr from cced files *********')
        generate_bicorr(folder_start,folder_end)
    if 2 in option:
        print('********* Build bhm: Positive time range **********')
        build_bhm(folder_start,folder_end,dt_bin_edges = np.arange(0.0,200.25,0.25))
        print('********* Build bhm: Negative time range **********')
        build_bhm(folder_start,folder_end,dt_bin_edges = np.arange(-200.0,0.25,0.25),sparse_filename = 'sparse_bhm_neg')
    if 3 in option:
        print('********* Build singles_hist **************')
        generate_singles_hist(folder_start,folder_end)
        
if __name__ == "__main__":
    """
    Calls main()
    
    Parameters
    ----------
    folder_start : int, optional
        Folder to start with
    folder_end : int, optional
        Folder to end at (don't run this one)
    option_code : int, optional
        Which options to include? Write all as one integer. Ex: option = [1,2] corresponds to option_code = 12
        1: generate_bicorr
        2: build_bhm pos and neg, store sparse_bhm
        3: generate_singles_hist, store singles_hist.npz
        Default: Run everything
    
	"""
    # print(len(sys.argv))
    
    if len(sys.argv) == 3:
        folder_start = int(sys.argv[1])
        folder_end   = int(sys.argv[2])

        option = [1,2,3]
        
        print('folder_start = ', folder_start)
        print('folder_end = ', folder_end)
        
        main(folder_start,folder_end,option)
        
    elif len(sys.argv) == 4:
        folder_start = int(sys.argv[1])
        folder_end   = int(sys.argv[2])
        option_code  = int(sys.argv[3]); option = [int(i) for i in str(option_code)]
        
        print('folder_start = ', folder_start)
        print('folder_end = ', folder_end)
        print('option = ', option)
        
        main(folder_start,folder_end,option)    
        
    else:

        main()