# # Analysis of combined data sets: Counts vs. angle
# 
# 7/18/2018
# 
# Doing this in energy space because that is more accurate.

# Import packages ------------------------------
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
import seaborn as sns
sns.set(style='ticks')

# Custom scripts ---------------------------------
sys.path.append('../../scripts')

import bicorr as bicorr
import bicorr_e as bicorr_e
import bicorr_plot as bicorr_plot
import bicorr_sums as bicorr_sums
import bicorr_math as bicorr_math

# Specify energy range
e_min = 1
e_max = 4


# Load data- experimental setup --------------------
det_df = bicorr.load_det_df('../../meas_info/det_df_pairs_angles.csv')
chList, fcList, detList, num_dets, num_det_pairs = bicorr.build_ch_lists()
dict_pair_to_index, dict_index_to_pair, dict_pair_to_angle = bicorr.build_dict_det_pair(det_df)

# Load singles hist and bicorr histogram -----------
singles_hist_e_n, e_bin_edges, dict_det_to_index, dict_index_to_det = bicorr_e.load_singles_hist_both(filepath = 'datap/',plot_flag=True, save_flag=True)

bhm_e, e_bin_edges, note = bicorr_e.load_bhm_e('datap')

# Calculate bicorr hist plot -----------------------
bhp_e = np.zeros((len(det_df),len(e_bin_edges)-1,len(e_bin_edges)-1))
for index in det_df.index.values: # index is same as in `bhm`
    bhp_e[index,:,:] = bicorr_e.build_bhp_e(bhm_e,e_bin_edges,pair_is=[index])[0]

# Singles sums -------------------------------------
singles_e_df = bicorr_sums.init_singles_e_df(dict_index_to_det)
singles_e_df = bicorr_sums.fill_singles_e_df(dict_index_to_det, singles_hist_e_n, e_bin_edges, e_min, e_max)
bicorr_plot.Sd_vs_ch_all(singles_e_df, show_flag=False)

# Append sums to det_df ---------------------------------
det_df = bicorr_sums.init_det_df_sums(det_df)
det_df, energies_real = bicorr_sums.fill_det_df_doubles_e_sums(det_df, bhp_e, e_bin_edges, e_min, e_max, True)
det_df = bicorr_sums.fill_det_df_singles_sums(det_df, singles_e_df)
det_df = bicorr_sums.calc_det_df_W(det_df)

# Plot W vs angle --------------------------------------
chIgnore = [1,17,33]
det_df_ignore = det_df[~det_df['d1'].isin(chIgnore) & ~det_df['d2'].isin(chIgnore)]
bicorr_plot.W_vs_angle_all(det_df_ignore, save_flag=True, show_flag=False)

# Group into angle bins --------------------------------
angle_bin_edges = np.arange(10.01,181,10)
by_angle_df = bicorr_sums.condense_det_df_by_angle(det_df_ignore, angle_bin_edges)
bicorr_plot.W_vs_angle(det_df_ignore, by_angle_df, save_flag=True, show_flag=False)

# Store to datap ---------------------------------------
singles_e_df.to_csv('datap/singles_e_df_filled.csv')
det_df_ignore.to_csv(r'datap/det_df_e_ignorefc_filled.csv')
det_df.to_csv(r'datap/det_df_e_filled.csv')
by_angle_df.to_csv(r'datap/by_angle_e_df.csv')

