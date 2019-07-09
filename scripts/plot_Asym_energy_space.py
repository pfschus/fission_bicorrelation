# # Calculate `Asym` vs. `Emin` from `bhm_e`
# 
# Rewriting `calc_Asym_vs_emin_energies` for `bhm_e`.
# This will calculate Asym_df.csv for a single data set. Run this for each of the datasets.
# 
# P. Schuster  
# July 18, 2018  
# Coded in methods/calculate_Asym_energy_space, downloading to .py here to run easily

# Import packages ----------------------------------------------------------------
import matplotlib
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

# Import custom scripts
sys.path.append('../scripts/')
import bicorr as bicorr
import bicorr_math as bicorr_math
import bicorr_plot as bicorr_plot
import bicorr_e as bicorr_e
import bicorr_sums as bicorr_sums

# Load data- experimental system-----------------------------------------------------
det_df = bicorr.load_det_df('../../meas_info/det_df_pairs_angles.csv')
chList, fcList, detList, num_dets, num_det_pairs = bicorr.build_ch_lists()
dict_pair_to_index, dict_index_to_pair, dict_pair_to_angle = bicorr.build_dict_det_pair(det_df)

# Load data- singles histogram, bicorr histogram--------------------------------------
singles_hist_e_n, e_bin_edges, dict_det_to_index, dict_index_to_det = bicorr_e.load_singles_hist_both(filepath = 'datap/',plot_flag=True, save_flag=True)
bhm_e, e_bin_edges, note = bicorr_e.load_bhm_e('datap')

# Perform analysis---------------------------------------------------------------------
# Create bicorr hist plot, just neutrons
bhp_e = np.zeros((len(det_df),len(e_bin_edges)-1,len(e_bin_edges)-1))
for index in det_df.index.values: # index is same as in `bhm`
    bhp_e[index,:,:] = bicorr_e.build_bhp_e(bhm_e,e_bin_edges,pair_is=[index])[0]

# Set up energy ranges for calculating sums
emins = np.arange(1,4,.2)
emax = 4
# Set up angle bin edges for grouping detector pairs
angle_bin_edges = np.arange(10.01,181,10)

# Calculate Asym vs energy threshold
Asym_df = bicorr_sums.calc_Asym_vs_emin_energies(det_df, dict_index_to_det, singles_hist_e_n, e_bin_edges, bhp_e, e_bin_edges, emins, emax, angle_bin_edges)

# Write to .csv file
Asym_df.to_csv('datap/Asym_df.csv')