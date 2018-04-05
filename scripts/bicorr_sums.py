"""
Calculate sums of bicorrelation distribution
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

from bicorr import *
from bicorr_plot import * 

# --SINGLES SUMS TO SINGLES_DF ---------------------------------------------
def init_singles_df(dict_index_to_det):
    '''
    Build empty singles dataframe
    
    Load dict_index_to_det with `singles_hist, dt_bin_edges_sh, dict_det_to_index, dict_index_to_det = bicorr.load_singles_hist(filepath='datap')`
    '''
    singles_df = pd.DataFrame.from_dict(dict_index_to_det,orient='index',dtype=np.int8).rename(columns={0:'ch'})
    chIgnore = [1,17,33]
    singles_df = singles_df[~singles_df['ch'].isin(chIgnore)].copy()
    
    singles_df['Sp']= np.nan
    singles_df['Sn']= np.nan
    singles_df['Sd']= np.nan
    singles_df['Sd_err'] = np.nan
    
    return singles_df
    
def fill_singles_df(dict_index_to_det, singles_hist, dt_bin_edges_sh, emin, emax):
    '''
    Calculate singles sums and fill singles_df
    '''    
    singles_df = init_singles_df(dict_index_to_det)
    
    for index in singles_df.index.values:
        Sp, Sn, Sd, Sd_err = calc_n_sum_br(singles_hist, dt_bin_edges_sh, index, emin=emin, emax=emax)
        singles_df.loc[index,'Sp'] = Sp
        singles_df.loc[index,'Sn'] = Sn
        singles_df.loc[index,'Sd'] = Sd
        singles_df.loc[index,'Sd_err'] = Sd_err
    
    return singles_df
        
# --DOUBLES SUMS TO DOUBLES_DF---------------------------------------------------
def init_det_df_sums(det_df):
    '''
    Add more columns (empty right now) to det_df, which I will fill with sums
    '''
    det_df['Cp'] = np.nan
    det_df['Cn'] = np.nan
    det_df['Cd'] = np.nan
    det_df['Cd_err'] = np.nan
    det_df['Np'] = np.nan
    det_df['Nn'] = np.nan
    det_df['Nd'] = np.nan
    det_df['Nd_err'] = np.nan
    
    det_df['Sd1']     = np.nan
    det_df['Sd1_err'] = np.nan
    det_df['Sd2']     = np.nan
    det_df['Sd2_err'] = np.nan

    det_df['W']     = np.nan
    det_df['W_err'] = np.nan
    
    return det_df
    
def fill_det_df_singles_sums(det_df, singles_df):
    '''
    Map singles sums from (filled) singles_df to det_df
    '''
    # Fill S columns in det_df
    for index in singles_df.index.values:
        ch = singles_df.loc[index,'ch']
        
        d1_indices = (det_df[det_df['d1'] == ch]).index.tolist()
        d2_indices = (det_df[det_df['d2'] == ch]).index.tolist()
        
        det_df.loc[d1_indices,'Sd1']     = singles_df.loc[index,'Sd']
        det_df.loc[d1_indices,'Sd1_err'] = singles_df.loc[index,'Sd_err']
        det_df.loc[d2_indices,'Sd2']     = singles_df.loc[index,'Sd']
        det_df.loc[d2_indices,'Sd2_err'] = singles_df.loc[index,'Sd_err']
        
    return det_df
    
def fill_det_df_doubles_sums(det_df, bhp_nn_pos, bhp_nn_neg, dt_bin_edges, emin, emax, num_fissions):
    '''
    Calculate and fill det_df doubles sums C and N
    '''
    for index in det_df.index.values:
        Cp, Cn, Cd, err_Cd = calc_nn_sum_br(bhp_nn_pos[index,:,:],
                                                   bhp_nn_neg[index,:,:],
                                                   dt_bin_edges,
                                                   emin=emin, emax=emax)
        det_df.loc[index,'Cp'] = Cp
        det_df.loc[index,'Cn'] = Cn
        det_df.loc[index,'Cd'] = Cd
        det_df.loc[index,'Cd_err'] = err_Cd
        Np, Nn, Nd, err_Nd = calc_nn_sum_br(bhp_nn_pos[index,:,:],
                                                   bhp_nn_neg[index,:,:],
                                                   dt_bin_edges,
                                                   emin=emin, emax=emax,
                                                   norm_factor = num_fissions)
        det_df.loc[index,'Np'] = Np
        det_df.loc[index,'Nn'] = Nn
        det_df.loc[index,'Nd'] = Nd
        det_df.loc[index,'Nd_err'] = err_Nd

    return det_df

        
def calc_det_df_W(det_df):
    '''
    Calculate W values in det_df. Requires that doubles and singles count rates are already filled
    '''
    det_df['W'] = det_df['Cd']/(det_df['Sd1']*det_df['Sd2'])
    det_df['W_err'] = det_df['W'] * np.sqrt((det_df['Cd_err']/det_df['Cd'])**2 + 
                                        (det_df['Sd1_err']/det_df['Sd1'])**2 + 
                                        (det_df['Sd2_err']/det_df['Sd2'])**2)
    return det_df
        
        
        
        
        
        
# ------- CONDENSE BY ANGLES ---------------------
def condense_det_df_by_angle(det_df,angle_bin_edges, C_flag=False, plot_flag = False, show_flag = False):
    '''
    Condense anisotropy distribution by discrete angle bins.
    
    C_flag: Option to calculate average total count (before singles correction)
    '''
    angle_bin_centers = calc_centers(angle_bin_edges)
        
    # Set up by_angle_df
    by_angle_df = pd.DataFrame({'angle_bin_min':angle_bin_edges[:-1],
    'angle_bin_max':angle_bin_edges[1:],'angle_bin_centers':angle_bin_centers})
    by_angle_df['len pair_is'] = np.nan
    by_angle_df['W'] = np.nan
    by_angle_df['W_err'] = np.nan
    by_angle_df['std W'] = np.nan
    
    if C_flag:    
        by_angle_df['Cd'] = np.nan
        by_angle_df['Cd_err'] = np.nan
        by_angle_df['std_Cd'] = np.nan
        
    for index in np.arange(len(angle_bin_edges)-1):
        pair_is = generate_pair_is(det_df,angle_bin_edges[index],angle_bin_edges[index+1],True)
        if len(pair_is) > 0:
            by_angle_df.loc[index,'len pair_is'] = len(pair_is)            
            by_angle_df.loc[index,'W']=    np.sum(det_df.loc[pair_is,'W'])/len(pair_is)
            by_angle_df.loc[index,'W_err']=np.sqrt(np.sum(det_df.loc[pair_is,'W_err']**2))/len(pair_is)
            by_angle_df.loc[index,'std W']=np.std(det_df.loc[pair_is,'W'])  
            if C_flag:
                by_angle_df.loc[index,'Cd']=    np.sum(det_df.loc[pair_is,'Cd'])/len(pair_is)
                by_angle_df.loc[index,'Cd_err']=np.sqrt(np.sum(det_df.loc[pair_is,'Cd_err']**2))/len(pair_is)
                by_angle_df.loc[index,'std_Cd']=np.std(det_df.loc[pair_is,'Cd'])   
    if plot_flag:    
        plt.figure(figsize=(4,3))
        plt.errorbar(det_df['angle'],det_df['W'],yerr=det_df['W_err'],
                         fmt='.',color='r',markersize=5,elinewidth=.5, zorder=1)
        plt.errorbar(by_angle_df['angle_bin_centers'],by_angle_df['W'],yerr=by_angle_df['std W'],fmt='.',color='k',zorder=3)
        step_plot(angle_bin_edges,by_angle_df['W'],linewidth=1,zorder=2)
        plt.xlabel('Angle (degrees)')
        plt.ylabel('W (relative doubles rate)')
        sns.despine(right=False)
        if show_flag: plt.show()  
        
        
    return by_angle_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    