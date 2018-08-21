"""
Main function for calling bicorr analysis from command line
""" 
import sys
import bicorr as bicorr
import bicorr_e as bicorr_e
import bicorr_sim as bicorr_sim

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
        bicorr.generate_bicorr(folder_start,folder_end)
    if 2 in option:
        print('********* Build bhm: Positive time range **********')
        bicorr.build_bhm(folder_start,folder_end,dt_bin_edges = np.arange(0.0,200.25,0.25))
        print('********* Build bhm: Negative time range **********')
        bicorr.build_bhm(folder_start,folder_end,dt_bin_edges = np.arange(-200.0,0.25,0.25),sparse_filename = 'sparse_bhm_neg')
    if 3 in option:
        print('********* Build singles_hist **************')
        bicorr.generate_singles_hist(folder_start,folder_end)
    if 4 in option:
        print('********* Build bhm_e *********************')
        bicorr_e.build_bhm_e(folder_start,folder_end)
        
        
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
        MEASUREMENTS:
        1: generate_bicorr
        2: build_bhm pos and neg, store sparse_bhm
        3: generate_singles_hist, store singles_hist.npz
        4: generate bhm_e        
        
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