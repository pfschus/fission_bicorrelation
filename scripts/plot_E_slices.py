# Calculate Esum_df.csv for a given dataset

# Import packages -----------------------------------------------
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


# Import custom functions -------------------------------------
sys.path.append('../../scripts/')
import bicorr as bicorr
import bicorr_math as bicorr_math
import bicorr_plot as bicorr_plot
import bicorr_e as bicorr_e
import bicorr_sums as bicorr_sums


# Load data- experimental setup -------------------------------
det_df = bicorr.load_det_df('../../meas_info/det_df_pairs_angles.csv')
chList, fcList, detList, num_dets, num_det_pairs = bicorr.build_ch_lists()
dict_pair_to_index, dict_index_to_pair, dict_pair_to_angle = bicorr.build_dict_det_pair(det_df)

# Load data- singles and bicorr histograms
singles_hist_e_n, e_bin_edges, dict_det_to_index, dict_index_to_det = bicorr_e.load_singles_hist_both(filepath = 'datap/',plot_flag=True, save_flag=True)
bhm_e, e_bin_edges, note = bicorr_e.load_bhm_e('datap')

# Set up analysis parameters
e_slices = list(np.arange(1,6,.5))
e_slice_width = 0.1
print(e_slices)

th_bin_edges = np.arange(10.01,181,10)
th_bin_centers = bicorr_math.calc_centers(th_bin_edges)
print(th_bin_edges)

# Create bhp_e
bhp_e = np.zeros([len(th_bin_edges)-1,len(e_bin_edges)-1,len(e_bin_edges)-1])

Eave = np.zeros([len(th_bin_edges-1),len(e_slices)])
Eave_err = np.zeros([len(th_bin_edges),len(e_slices)])
Ej = np.zeros(len(e_slices))

E_min = 1
E_max = 4

# Calculate slices
bhp_e_slices = np.zeros([len(th_bin_edges),len(e_slices),len(e_bin_edges)-1])
for th_i in range(len(th_bin_edges)-1):
    th_min = th_bin_edges[th_i]
    th_max = th_bin_edges[th_i+1]
    print(th_min,th_max)
    
    pair_is = bicorr.generate_pair_is(det_df,th_min=th_min,th_max=th_max)
    if len(pair_is) == 0: continue
        
    bhp_e[th_i,:,:] = bicorr_e.build_bhp_e(bhm_e,e_bin_edges,pair_is=pair_is)[0]
    bhp_e_slices[th_i,:,:], slice_e_ranges = bicorr_e.slices_bhp_e(bhp_e[th_i,:,:],e_bin_edges,e_slices,e_slice_width=e_slice_width)
    Eave[th_i,:], Eave_err[th_i,:], Ej = bicorr_e.calc_Eave_slices(bhp_e_slices[th_i,:,:],e_slices,e_bin_edges,E_min,E_max)
    
    save_filename = r'Eave_{0:.2f}_{0:.2f}'.format(th_min,th_max)
    plt.figure(figsize=(4,3))
    plt.errorbar(Ej,Eave[th_i,:],yerr=Eave_err[th_i,:],fmt='.')
    plt.xlabel('$E_j$ (MeV)')
    plt.ylabel('Average $E_i$ (MeV)')
    plt.xlim([.3,6])
    plt.ylim([2.1,2.4])
    # plt.title('{} to {} degrees'.format(th_min,th_max))
    sns.despine(right=False)
    bicorr_plot.save_fig_to_folder(save_filename,'fig/animate_Eave',['png'])
    plt.clf()

    
# Now calculate average energy sum
i_E_min = np.digitize(E_min,e_bin_edges)-1
i_E_max = np.digitize(E_max,e_bin_edges)-1

centers = bicorr_math.calc_centers(e_bin_edges)[i_E_min:i_E_max]
X, Y = np.meshgrid(centers, centers)

Esum_df = pd.DataFrame({'th_bin_center':th_bin_centers})
Esum_df['Eave'] = np.nan
Esum_df['Eave_err'] = np.nan

for th_i in range(len(th_bin_edges)-1):
    th_min = th_bin_edges[th_i]
    th_max = th_bin_edges[th_i+1]
    
    pair_is = bicorr.generate_pair_is(det_df,th_min=th_min,th_max=th_max)
    if len(pair_is) > 5: 
        bhp_e = bicorr_e.build_bhp_e(bhm_e,e_bin_edges,pair_is=pair_is)[0]
        H = bhp_e[i_E_min:i_E_max,i_E_min:i_E_max] # Range of nn energy blob to average
        Esum_df.loc[th_i,'Eave'], Esum_df.loc[th_i,'Eave_err'] = bicorr_math.calc_histogram_mean((X+Y)/2,H,False,True)
print(Esum_df)
Esum_df.to_csv(r'datap/Esum_df.csv')