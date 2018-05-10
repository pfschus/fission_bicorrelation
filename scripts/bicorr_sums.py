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
from bicorr_math import *

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
    
def fill_det_df_doubles_sums(det_df, bhp_nn_pos, bhp_nn_neg, dt_bin_edges, emin, emax, num_fissions, return_real_energies_flag = False):
    '''
    Calculate and fill det_df doubles sums C and N
    '''
    for index in det_df.index.values:
        Cp, Cn, Cd, err_Cd, energies_real = calc_nn_sum_br(bhp_nn_pos[index,:,:],
                                                   bhp_nn_neg[index,:,:],
                                                   dt_bin_edges,
                                                   emin=emin, emax=emax, return_real_energies_flag = True)
        det_df.loc[index,'Cp'] = Cp
        det_df.loc[index,'Cn'] = Cn
        det_df.loc[index,'Cd'] = Cd
        det_df.loc[index,'Cd_err'] = err_Cd
        Np, Nn, Nd, err_Nd, energies_real = calc_nn_sum_br(bhp_nn_pos[index,:,:],
                                                   bhp_nn_neg[index,:,:],
                                                   dt_bin_edges,
                                                   emin=emin, emax=emax,
                                                   norm_factor = num_fissions, return_real_energies_flag = True)
        det_df.loc[index,'Np'] = Np
        det_df.loc[index,'Nn'] = Nn
        det_df.loc[index,'Nd'] = Nd
        det_df.loc[index,'Nd_err'] = err_Nd

    if return_real_energies_flag:
        return det_df, energies_real
    else:
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
    by_angle_df['std_angle'] = np.nan
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
            by_angle_df.loc[index,'std_angle'] = np.std(det_df.loc[pair_is,'angle'])
            by_angle_df.loc[index,'W']=    np.sum(det_df.loc[pair_is,'W'])/len(pair_is)
            by_angle_df.loc[index,'W_err']=np.sqrt(np.sum(det_df.loc[pair_is,'W_err']**2))/len(pair_is)
            by_angle_df.loc[index,'std W']=np.std(det_df.loc[pair_is,'W'])  
            if C_flag:
                by_angle_df.loc[index,'Cd']=    np.sum(det_df.loc[pair_is,'Cd'])/len(pair_is)
                by_angle_df.loc[index,'Cd_err']=np.sqrt(np.sum(det_df.loc[pair_is,'Cd_err']**2))/len(pair_is)
                by_angle_df.loc[index,'std_Cd']=np.std(det_df.loc[pair_is,'Cd'])   

        
        
    return by_angle_df
        
def perform_W_calcs(det_df,
                    dict_index_to_det, singles_hist, dt_bin_edges_sh,
                    bhp_nn_pos, bhp_nn_neg, dt_bin_edges,
                    num_fissions, emin, emax, angle_bin_edges, return_real_energies_flag = False):

    """
    Perform all operations for calculating W for each detector pair and in each angle bin
    """
    singles_df  = fill_singles_df(dict_index_to_det, singles_hist, dt_bin_edges_sh, emin, emax)
    det_df      = init_det_df_sums(det_df)
    det_df      = fill_det_df_singles_sums(det_df, singles_df)
    det_df, energies_real = fill_det_df_doubles_sums(det_df, bhp_nn_pos, bhp_nn_neg, dt_bin_edges, emin, emax, num_fissions, return_real_energies_flag = True)
    det_df      = calc_det_df_W(det_df)
    by_angle_df = condense_det_df_by_angle(det_df,angle_bin_edges)
    
    if return_real_energies_flag:
        return singles_df, det_df, by_angle_df, energies_real
    else:
        return singles_df, det_df, by_angle_df
        
# ------------ ASYM CALCULATIONS -------
def calc_Asym(by_angle_df, std_flag = True):
    """
    Errors propagated from std(W), not W_err
    
    if std_flag = True: Propagate errors from std(W)
    if std_flag = False: Propagate errosr from W_err
    """

    angle_bin_edges = [by_angle_df.loc[0,'angle_bin_min']]+by_angle_df['angle_bin_max'].values.tolist()

    series_180 = by_angle_df.loc[np.int(np.digitize(180,angle_bin_edges))-1]
    series_90 = by_angle_df.loc[np.int(np.digitize(90,angle_bin_edges))-1]
    
    series_180 = by_angle_df.loc[np.int(np.digitize(180,angle_bin_edges))-1]
    series_90 = by_angle_df.loc[np.int(np.digitize(90,angle_bin_edges))-1]
    
    num = series_180['W']
    denom = series_90['W']
    if std_flag:  
        num_err = series_180['std W']
        denom_err = series_90['std W']
    else:
        num_err = series_180['W_err']
        denom_err = series_90['W_err']
    
    Asym, Asym_err = prop_err_division(num,num_err,denom,denom_err)
    
    return Asym, Asym_err
        
def calc_Asym_vs_emin(det_df,
                    dict_index_to_det, singles_hist, dt_bin_edges_sh,
                    bhp_nn_pos, bhp_nn_neg, dt_bin_edges,
                    num_fissions, emins, emax, angle_bin_edges,
                    plot_flag=True, save_flag=True):
    """
    Calculate Asym for variable emin values. The input parameter emins is an array of emin values. emax constant.
    """
    
        
        
    # Initialize Asym_df
    Asym_df = pd.DataFrame(data = {'emin': emins})
    Asym_df['emax'] = emax
    Asym_df['emin_real'] = np.nan
    Asym_df['emax_real'] = np.nan
    Asym_df['Asym'] = np.nan
    Asym_df['Asym_err'] = np.nan
    
    # Fill Asym_df
    for index, row in Asym_df.iterrows():    
        singles_df, det_df, by_angle_df, energies_real = perform_W_calcs(det_df,
                        dict_index_to_det, singles_hist, dt_bin_edges_sh,
                        bhp_nn_pos, bhp_nn_neg, dt_bin_edges,
                        num_fissions, row['emin'], row['emax'], angle_bin_edges,
                        return_real_energies_flag = True)
        Asym, Asym_err = calc_Asym(by_angle_df)
        
        Asym_df.loc[index,'emin_real'] = energies_real[1]
        Asym_df.loc[index,'emax_real'] = energies_real[0]
        Asym_df.loc[index,'Asym'] = Asym
        Asym_df.loc[index,'Asym_err'] = Asym_err
        
    if plot_flag:
        plt.figure(figsize=(4,3))
        plt.errorbar(Asym_df['emin'],Asym_df['Asym'],yerr=Asym_df['Asym_err'],fmt='.',color='k')
        plt.xlabel('$E_{min}$ (MeV)')
        plt.ylabel('$A_{sym}$')
        plt.title('Errors from std(W)')
        sns.despine(right=True)
        if save_flag: bicorr_plot.save_fig_to_folder('Asym_vs_emin')
        plt.show()
        
    return Asym_df    
        
def calc_Asym_vs_ebin(det_df,
                    dict_index_to_det, singles_hist, dt_bin_edges_sh,
                    bhp_nn_pos, bhp_nn_neg, dt_bin_edges,
                    num_fissions, e_bin_edges, angle_bin_edges,
                    plot_flag=True, save_flag=True):
    """
    Calculate Asym for variable emin values. The input parameter ebins is an array of energy values. Each calculation will use emin = ebins[i], emax = ebins[i+1].
    """
    
        
        
    # Initialize Asym_df
    Asym_df = pd.DataFrame(data = {'emin':e_bin_edges[:-1],'emax':e_bin_edges[1:]})
    Asym_df['Asym'] = np.nan
    Asym_df['Asym_err'] = np.nan
    
    # Fill Asym_df
    for index, row in Asym_df.iterrows():    
        singles_df, det_df, by_angle_df = perform_W_calcs(det_df,
                        dict_index_to_det, singles_hist, dt_bin_edges_sh,
                        bhp_nn_pos, bhp_nn_neg, dt_bin_edges,
                        num_fissions, row['emin'], row['emax'], angle_bin_edges)
        Asym, Asym_err = calc_Asym(by_angle_df)
        
        Asym_df.loc[index,'Asym'] = Asym
        Asym_df.loc[index,'Asym_err'] = Asym_err
        
    if plot_flag:
        plt.figure(figsize=(4,3))
        plt.errorbar(Asym_df['emin_real'],Asym_df['Asym'],yerr=Asym_df['Asym_err'],fmt='.',color='k')
        plt.xlabel('$E_{min}$ (MeV)')
        plt.ylabel('$A_{sym}$')
        plt.title('Errors from std(W)')
        sns.despine(right=True)
        if save_flag: bicorr_plot.save_fig_to_folder('Asym_vs_emin')
        plt.show()
        
    return Asym_df  
    