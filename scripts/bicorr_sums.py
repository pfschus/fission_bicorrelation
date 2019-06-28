"""
Calculate sums of bicorrelation distribution
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

import bicorr as bicorr
import bicorr_math as bicorr_math
import bicorr_plot as bicorr_plot

# --Calculate sums on energy histograms ------------------------------------
def calc_n_sum_e(singles_hist_e_n, e_bin_edges, det_i, e_min=1, e_max=4):
    """
    Calculate background-subtracted number of neutron events within a given time range in the singles histogram. Analogous to calc_nn_sum and calc_nn_sum_br for singles events.
    
    NOTE: I AM NOT NORMALIZING THIS. PLAN ACCORDINGLY WHEN USING TOGETHER WITH CALC_NN_SUM
    
    Parameters
    ----------
    singles_hist_e_n : ndarray
        Histogram of singles timing information for neutrons
        Dimension 0: detector channel
        Dimension 1: dt bin    
    e_bin_edges : ndarray
        Energy bin edges array
    det_i : int
        Index of detector in singles_hist. Use dict_det_to_index from `load_singles_hist`
    e_min : float
        Lower energy boundary for neutron event selection in MeV
        Default 0.62 MeV ~ 70 keVee
    e_max : float, optional
        Upper energy boundary for neutron event selection in MeV    
    
    Returns
    -------
    Se : int
    Se_err : int
        1-sigma error in counts
    """
    # Calculate energy window indices
    i_min = np.digitize(e_min,e_bin_edges) - 1
    i_max = np.digitize(e_max,e_bin_edges) - 1
    
    # What range am I actually integrating over?
    e_range = [e_bin_edges[i_min],e_bin_edges[i_max]]
    
    Se = np.sum(singles_hist_e_n[det_i,i_min:i_max])    
    Se_err = np.sqrt(Se)        
    return Se, Se_err, e_range
    
def calc_nn_sum_e(bhp_nn_e, e_bin_edges, e_min=1, e_max=4, return_real_energies_flag = False):
    """
    Calculate the number of counts in a given time range after background subtraction.
    
    Parameters
    ---------
    bhp_nn_e : ndarray
        Energy nn bicorr_hist_plot
    e_bin_edges_e : ndarray
        Energy bin edges
    emin : float
        Lower energy boundary for neutron event selection in MeV
        Default 0.62 MeV ~ 70 keVee
    emax : float, optional
        Upper energy boundary for neutron event selection in MeV
    
    
    Returns
    -------
    Ce : int or float
        Counts
    Ce_err
        1-sigma error in counts
    energies_real
        Actual energy bin limits (due to discrete energy bins)
    """
    # Calculate energy window indices
    i_min = np.digitize(e_min,e_bin_edges) - 1
    i_max = np.digitize(e_max,e_bin_edges) - 1
    
    # What range am I actually integrating over?
    e_range = [e_bin_edges[i_min],e_bin_edges[i_max]]

    Ce = np.sum(bhp_nn_e[i_min:i_max,i_min:i_max])
    Ce_err = np.sqrt(Ce)
    
    if return_real_energies_flag: 
        return Ce, Ce_err, e_range
    else: 
        return Ce, Ce_err


# --SINGLES SUMS TO SINGLES_DF: TIME ---------------------------------------------
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
        Sp, Sn, Sd, Sd_err = bicorr.calc_n_sum_br(singles_hist, dt_bin_edges_sh, index, emin=emin, emax=emax)
        singles_df.loc[index,'Sp'] = Sp
        singles_df.loc[index,'Sn'] = Sn
        singles_df.loc[index,'Sd'] = Sd
        singles_df.loc[index,'Sd_err'] = Sd_err
    
    return singles_df

# --SINGLES SUMS TO SINGLES_DF: ENERGY ---------------------------------------------    
def init_singles_e_df(dict_index_to_det):
    '''
    Build empty singles dataframe
    
    Load with 
    singles_hist_e_n, e_bin_edges, dict_det_to_index, dict_index_to_det = bicorr_e.load_singles_hist_both()
    '''
    singles_e_df = pd.DataFrame.from_dict(dict_index_to_det,orient='index',dtype=np.int8).rename(columns={0:'ch'})
    chIgnore = [1,17,33]
    #singles_e_df = singles_e_df[~singles_e_df['ch'].isin(chIgnore)].copy()
    
    singles_e_df['Sd']= np.nan
    singles_e_df['Sd_err'] = np.nan
    
    return singles_e_df
    
def fill_singles_e_df(dict_index_to_det, singles_hist_e_n, e_bin_edges, e_min, e_max):
    '''
    Calculate singles sums and fill singles_df
    '''    
    singles_e_df = init_singles_e_df(dict_index_to_det)
    
    for index in singles_e_df.index.values:
        Se, Se_err, e_range = calc_n_sum_e(singles_hist_e_n, e_bin_edges, index, e_min=e_min, e_max=e_max)
        singles_e_df.loc[index,'Sd'] = Se
        singles_e_df.loc[index,'Sd_err'] = Se_err
    
    return singles_e_df
    
        
# --DOUBLES SUMS TO DOUBLES_DF---------------------------------------------------
def init_det_df_sums(det_df, t_flag = False):
    '''
    Add more columns (empty right now) to det_df, which I will fill with sums
    '''
    if t_flag: # For background subtraction
        det_df['Cp'] = np.nan
        det_df['Cn'] = np.nan
    det_df['Cd'] = np.nan
    det_df['Cd_err'] = np.nan
    
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

########### FILL DOUBLES SUMS -> DET DF ###################################
def fill_det_df_doubles_t_sums(det_df, bhp_nn_pos, bhp_nn_neg, dt_bin_edges, emin, emax, return_real_energies_flag = False):
    '''
    Calculate and fill det_df doubles sums C and N
    '''
    for index in det_df.index.values:
        Cp, Cn, Cd, err_Cd, energies_real = bicorr.calc_nn_sum_br(bhp_nn_pos[index,:,:],
                                                   bhp_nn_neg[index,:,:],
                                                   dt_bin_edges,
                                                   emin=emin, emax=emax, return_real_energies_flag = True)
        det_df.loc[index,'Cp'] = Cp
        det_df.loc[index,'Cn'] = Cn
        det_df.loc[index,'Cd'] = Cd
        det_df.loc[index,'Cd_err'] = err_Cd

    if return_real_energies_flag:
        return det_df, energies_real
    else:
        return det_df
    
def fill_det_df_doubles_e_sums(det_df, bhp_nn_e, e_bin_edges, e_min, e_max, return_real_energies_flag = False):
    '''
    Calculate and fill det_df doubles sums C and N
    
    
    det_df : pandas DataFrame
        detector pair data, after running init_det_df_sums
    bhp_nn_e : ndarray
        bhp for all detector pairs. First dimension is detector pair index
    e_bin_edges : ndarray
    e_min: float
        In MeV
    e_max : float
        In MeV
    '''
    for index in det_df.index.values:
        Cd, Cd_err, energies_real = calc_nn_sum_e(bhp_nn_e[index,:,:], e_bin_edges, e_min=e_min, e_max=e_max, return_real_energies_flag = True)
        
        det_df.loc[index,'Cd'] = Cd
        det_df.loc[index,'Cd_err'] = Cd_err

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
def condense_det_df_by_angle(det_df,angle_bin_edges, C_flag=False):
    '''
    Condense anisotropy distribution by discrete angle bins.
    
    C_flag: Option to calculate average total count (before singles correction)
    '''
    angle_bin_centers = bicorr_math.calc_centers(angle_bin_edges)
        
    # Set up by_angle_df
    by_angle_df = pd.DataFrame({'angle_bin_min':angle_bin_edges[:-1],
    'angle_bin_max':angle_bin_edges[1:],'angle_bin_centers':angle_bin_centers})
    by_angle_df['len pair_is'] = np.nan
    by_angle_df['std_angle'] = np.nan
    by_angle_df['Sd1'] = np.nan
    by_angle_df['Sd1_err'] = np.nan
    by_angle_df['Sd2'] = np.nan
    by_angle_df['Sd2_err'] = np.nan
    by_angle_df['Cd'] = np.nan
    by_angle_df['Cd_err'] = np.nan
    by_angle_df['W'] = np.nan
    by_angle_df['W_err'] = np.nan
    by_angle_df['std W'] = np.nan
    
    if C_flag:    
        by_angle_df['Cd'] = np.nan
        by_angle_df['Cd_err'] = np.nan
        by_angle_df['std_Cd'] = np.nan
        
    for index in np.arange(len(angle_bin_edges)-1):
        print('Generating data in angle bin', str(angle_bin_edges[index]), 'to', str(angle_bin_edges[index+1]))    
        pair_is = bicorr.generate_pair_is(det_df,angle_bin_edges[index],angle_bin_edges[index+1])
        if len(pair_is) > 0:        
            
            by_angle_df.loc[index,'len pair_is'] = int(len(pair_is))
            by_angle_df.loc[index,'std_angle'] = np.std(det_df.loc[pair_is,'angle'])
            by_angle_df.loc[index,'Sd1']=    np.mean(det_df.loc[pair_is,'Sd1'])
            by_angle_df.loc[index,'Sd1_err']=np.std(det_df.loc[pair_is,'Sd1']) 
            by_angle_df.loc[index,'Sd2']=    np.mean(det_df.loc[pair_is,'Sd2'])
            by_angle_df.loc[index,'Sd2_err']=np.std(det_df.loc[pair_is,'Sd2']) 
            by_angle_df.loc[index,'Cd']=    np.mean(det_df.loc[pair_is,'Cd'])
            by_angle_df.loc[index,'Cd_err']=np.std(det_df.loc[pair_is,'Cd']) 
            by_angle_df.loc[index,'W']=    np.mean(det_df.loc[pair_is,'W'])
            by_angle_df.loc[index,'W_err']=np.sqrt(np.sum(det_df.loc[pair_is,'W_err']**2))/len(pair_is)
            by_angle_df.loc[index,'std W']=np.std(det_df.loc[pair_is,'W'])  
            if C_flag:
                by_angle_df.loc[index,'Cd']=np.sum(det_df.loc[pair_is,'Cd'])/len(pair_is)
                by_angle_df.loc[index,'Cd_err']=np.sqrt(np.sum(det_df.loc[pair_is,'Cd_err']**2))/len(pair_is)
                by_angle_df.loc[index,'std_Cd']=np.std(det_df.loc[pair_is,'Cd'])   

        
        
    return by_angle_df


def perform_W_calcs_energies(det_df,
                    dict_index_to_det, singles_hist_e_n, e_bin_edges_sh,
                    bhp_e, e_bin_edges,
                    e_min, e_max, angle_bin_edges, return_real_energies_flag = False):

    """
    Perform all operations for calculating W for each detector pair and in each angle bin
    """
    singles_e_df = fill_singles_e_df(dict_index_to_det, singles_hist_e_n, e_bin_edges, e_min, e_max) 
    
    det_df = init_det_df_sums(det_df)    
    det_df, energies_real = fill_det_df_doubles_e_sums(det_df, bhp_e, e_bin_edges, e_min, e_max, True)
    det_df = fill_det_df_singles_sums(det_df, singles_e_df)
    det_df = calc_det_df_W(det_df)

    chIgnore = [1,17,33]
    det_df_ignore = det_df[~det_df['d1'].isin(chIgnore) & ~det_df['d2'].isin(chIgnore)]
    
    by_angle_df = condense_det_df_by_angle(det_df_ignore, angle_bin_edges)
    
    if return_real_energies_flag:
        return singles_e_df, det_df_ignore, by_angle_df, energies_real
    else:
        return singles_e_df, det_df_ignore, by_angle_df



    
def perform_W_calcs(det_df,
                    dict_index_to_det, singles_hist, dt_bin_edges_sh,
                    bhp_nn_pos, bhp_nn_neg, dt_bin_edges,
                    num_fissions, emin, emax, angle_bin_edges, return_real_energies_flag = False):

    """
    Perform all operations for calculating W for each detector pair and in each angle bin
    """
    singles_df  = fill_singles_df(dict_index_to_det, singles_hist, dt_bin_edges_sh, emin, emax)
    det_df      = init_det_df_sums(det_df, t_flag = True)
    det_df      = fill_det_df_singles_sums(det_df, singles_df)
    det_df, energies_real = fill_det_df_doubles_t_sums(det_df, bhp_nn_pos, bhp_nn_neg, dt_bin_edges, emin, emax, return_real_energies_flag = True)
    det_df      = calc_det_df_W(det_df)
    by_angle_df = condense_det_df_by_angle(det_df,angle_bin_edges)
    
    if return_real_energies_flag:
        return singles_df, det_df, by_angle_df, energies_real
    else:
        return singles_df, det_df, by_angle_df
        
# ------------ ASYM CALCULATIONS -------
def calc_Asym(by_angle_df, std_flag = True, min_flag = False):
    """
    Errors propagated from std(W), not W_err
    
    if std_flag = True: Propagate errors from std(W)
    if std_flag = False: Propagate errosr from W_err
    """

    angle_bin_edges = [by_angle_df.loc[0,'angle_bin_min']]+by_angle_df['angle_bin_max'].values.tolist()

    series_180 = by_angle_df.loc[np.int(np.digitize(180,angle_bin_edges))-1]
    series_90 = by_angle_df.loc[np.int(np.digitize(90,angle_bin_edges))-1]
    series_min = by_angle_df.loc[by_angle_df['W'].idxmin()]
    
    num = series_180['W']
    denom = series_90['W']
    minval = series_min['W']
    if std_flag:  
        num_err = series_180['std W']
        denom_err = series_90['std W']
        minval_err = series_min['std W']
    else:
        num_err = series_180['W_err']
        denom_err = series_90['W_err']
        minval_err = series_min['W_err']
    
    Asym, Asym_err = bicorr_math.prop_err_division(num,num_err,denom,denom_err)
    Asym_min, Asym_min_err = bicorr_math.prop_err_division(num,num_err,minval,minval_err)
    
    if min_flag:
        return Asym, Asym_err, Asym_min, Asym_min_err
    else:
        return Asym, Asym_err
    
def calc_Asym_vs_emin_energies(det_df,
                    dict_index_to_det, singles_hist_e_n, e_bin_edges_sh,
                    bhp_nn_e, e_bin_edges,
                    emins, emax, angle_bin_edges,
                    plot_flag=True, show_flag = False, save_flag=True):
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
    Asym_df['Asym_min'] = np.nan
    Asym_df['Asym_min_err'] = np.nan
    
    # Fill Asym_df
    for index, row in Asym_df.iterrows():    
        singles_df, det_df_ignore, by_angle_df, energies_real = perform_W_calcs_energies(det_df, 
                        dict_index_to_det, singles_hist_e_n, 
                        e_bin_edges, 
                        bhp_nn_e, e_bin_edges, 
                        row['emin'], row['emax'], angle_bin_edges, return_real_energies_flag = True)
        Asym, Asym_err, Asym_min, Asym_min_err = calc_Asym(by_angle_df,min_flag=True)
        
        Asym_df.loc[index,'emin_real'] = energies_real[1]
        Asym_df.loc[index,'emax_real'] = energies_real[0]
        Asym_df.loc[index,'Asym'] = Asym
        Asym_df.loc[index,'Asym_err'] = Asym_err
        Asym_df.loc[index,'Asym_min'] = Asym_min
        Asym_df.loc[index,'Asym_min_err'] = Asym_min_err
        
    if plot_flag:
        plt.figure(figsize=(4,3))
        plt.errorbar(Asym_df['emin'],Asym_df['Asym'],yerr=Asym_df['Asym_err'],fmt='.',color='k')
        plt.xlabel('$E_{min}$ (MeV)')
        plt.ylabel('$A_{sym}$')
        # plt.title('Errors from std(W)')
        sns.despine(right=True)
        if save_flag: bicorr_plot.save_fig_to_folder('Asym_vs_emin')
        if show_flag: plt.show()
        plt.clf()
        
        plt.figure(figsize=(4,3))
        plt.errorbar(Asym_df['emin'],Asym_df['Asym_min'],yerr=Asym_df['Asym_min_err'],fmt='.',color='k')
        plt.xlabel('$E_{min}$ (MeV)')
        plt.ylabel('$A_{sym}$')
        # plt.title('Errors from std(W)')
        sns.despine(right=True)
        if save_flag: bicorr_plot.save_fig_to_folder('Asym_min_vs_emin')
        if show_flag: plt.show()
        plt.clf()
        
        
        
    return Asym_df    
    
    
        
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
    

    