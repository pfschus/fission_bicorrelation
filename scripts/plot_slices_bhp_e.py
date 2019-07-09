# Energy slice analysis

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os
import scipy.io as sio
import sys
import time
import inspect
import pandas as pd
from tqdm import *


sys.path.append('../scripts/')
import bicorr as bicorr
import bicorr_plot as bicorr_plot
import bicorr_e as bicorr_e
import bicorr_math as bicorr_math


bhm_e, e_bin_edges, note = bicorr_e.load_bhm_e('datap')
det_df = bicorr.load_det_df('../../meas_info/det_df_pairs_angles.csv')
dict_pair_to_index, dict_index_to_pair, dict_pair_to_angle = bicorr.build_dict_det_pair(det_df)

num_fissions = float(np.squeeze(sio.loadmat('datap/num_fissions.mat')['num_fissions']))

angle_bin_edges = np.arange(10.01,181,10)
angle_bin_centers = bicorr_math.calc_centers(angle_bin_edges)

e_slices = list(np.arange(0.5,6,.5))
E_min = 1; E_max = 4;

# Allocate matrices
bhp_e = np.zeros((len(angle_bin_centers),len(e_bin_edges)-1,len(e_bin_edges)-1))
norm_factor = np.zeros(len(angle_bin_centers))

bhp_e_slices = np.zeros((len(angle_bin_centers),len(e_slices),len(e_bin_edges)-1))

Eave = np.zeros((len(angle_bin_centers),len(e_slices)))
Eave_err = np.zeros((len(angle_bin_centers),len(e_slices)))

# Do the calculations
for i in range(len(angle_bin_centers)):
    angle_min = angle_bin_edges[i]
    angle_max = angle_bin_edges[i+1]
    
    pair_is = bicorr.generate_pair_is(det_df, angle_min, angle_max)
    bhp_e[i,:,:], norm_factor[i] = bicorr_e.build_bhp_e(bhm_e,e_bin_edges,pair_is=pair_is,num_fissions = num_fissions,print_flag=True)
    bhp_e_slices[i,:,:],slice_e_ranges = bicorr_e.slices_bhp_e(bhp_e[i,:,:],e_bin_edges,e_slices,0.224)
    Eave[i,:], Eave_err[i,:], _ = bicorr_e.calc_Eave_slices(bhp_e_slices[i,:,:],e_slices,e_bin_edges,E_min,E_max,norm_factor=norm_factor[i])

# Calculate ranges
vmin = np.min(bhp_e[np.nonzero(bhp_e)])
vmax = np.max(bhp_e)

Eave_min = np.min(Eave[np.nonzero(Eave)])
Eave_max = np.max(Eave)

# Make the plots
filenames_bhp_e = []
filenames_Eave  = []  
for i in range(len(angle_bin_centers)): #range(len(angle_bin_centers)):
    print('Plotting in angle bin ', i, ' of ', range(len(angle_bin_centers)))
    angle_min = angle_bin_edges[i]
    angle_max = angle_bin_edges[i+1]   
    
    title = '{:d} to {:d} degrees'.format(int(angle_min),int(angle_max))  
    filename_bhp_e = 'bhp_e_{:d}_{:d}_deg'.format(int(angle_min),int(angle_max)); filenames_bhp_e.append(filename_bhp_e);
    bicorr_plot.bhp_e_plot(bhp_e[i,:,:], e_bin_edges, zoom_range = [0,6], 
                           vmin=vmin, vmax=vmax,
                           title=title, show_flag = False,
                           save_flag = True, save_filename = filename_bhp_e)        
    #bicorr_plot.plot_bhp_e_slices(bhp_e_slices[i,:,:],e_bin_edges,slice_e_ranges,
    #                               E_min = E_min, E_max = E_max, title=title,
    #                               save_filename = 'bhp_e_slices_{}_{}_degrees'.format(angle_min,angle_max))
    filename_Eave = 'Eave_{:d}_{:d}_degrees'.format(int(angle_min),int(angle_max)); filenames_Eave.append(filename_Eave);
    bicorr_plot.plot_Eave_vs_Ej(Eave[i,:], Eave_err[i,:], e_slices, title=title,
                                y_range = [Eave_min,Eave_max],
                                show_flag = False,
                                save_flag = True, save_filename = filename_Eave)


# Save data to file
np.savez('datap/slices_analysis', 
         angle_bin_edges = angle_bin_edges,
         angle_bin_centers = angle_bin_centers,
         e_slices = e_slices,
         E_min = E_min, E_max = E_max,
         bhp_e = bhp_e, norm_factor=norm_factor,
         bhp_e_slices = bhp_e_slices,
         Eave=Eave, Eave_err = Eave_err
         )


# # Animate it all
import imageio

images_bhp_e = []
for filename in filenames_bhp_e:
    images_bhp_e.append(imageio.imread(os.path.join('fig',filename + '.png')))
imageio.mimsave('fig/animate_bhp_e.gif',images_bhp_e, fps=1)

images_Eave = []
for filename in filenames_Eave:
    images_Eave.append(imageio.imread(os.path.join('fig',filename + '.png')))
imageio.mimsave('fig/animate_Eave.gif',images_Eave, fps=1)

