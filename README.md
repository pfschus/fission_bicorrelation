# fission_bicorrelation
Project for investigating the angular correlation of prompt neutrons and gamma-rays emitted from fission. All of the work presented in this repository is preliminary and subject to modification or removal at any time.

# Project overview
The work presented in this repository aims to characterize the angular distribution of prmopt neutrons and gamma-rays emitted from fission. We have performed measurements and built simulation tools for investigating this on Cf-252. 

My collaborator, Matthew Marcath, took measurements of a Cf-252 source positioned at the focal point of the Chi-Nu detector array at Los Alamos National Laboratory (LANL). A model of the detector array is shown below:

![LANL Chi-Nu array](fig/setup.png)

We were able to use 47 of the detectors in the array (limited due to our electronics setup). In the bicorrelation analysis detailed in this respository, we look for events in which a trigger was observed in the fission chamber and in two detectors. Such an event in which a neutron and a gamma-ray were detected is illustrated below:

![Bicorrelation event](fig/diagram.png)

In this event, we have three time stamps:

* `t_0` from the fission chamber. This is the "start" time and our reference time for when the fission occurred.
* `t_A` from detector A
* `t_B` from detector B

These times can be used to calculate the time of flight to detectors A and B:

* `Delta t_A = t_A-t_0`
* `Delta t_B = t_B-t_0`

The distribution of either of these time of flight values is expected to look something like this:

![1D time of flight](fig/1D_tof.PNG)

The inital peak is from prompt gamma-ray interactions. Since all gamma-rays travel at the speed of light, they should all arrive at the detector at the same time of flight. The neutrons travel with a distribution of energies, so they arrive at the detector across a range of time of flight values.

When looking at a two-dimensional distribution of these time of flight values across two detectors, we expect to see the following features:

![2D time of flight](fig/2D_tof.PNG)

There are four main features observed from detecting two prompt fission emission particles:

1. nn "blob": Extends across a range of `Delta t_A` and `Delta t_B` values according to the neutron time of flight distribution.
2. ng "band": `Delta t_B` ranges over neutron time of flight, `Delta t_A` fixed at gamma time of flight
3. gn "band": Opposite that of ng
4. gg peak: `Delta t_A` and `Delta t_B` fixed at their gamma-ray time of flight values

We will build analysis tools to produce this distribution for our measurements and simulations under various conditions. 

# Analysis tools

All of my functions are in the [bicorr.py](scripts/bicorr.py) file, which is extensively documented. The following Jupyter notebooks contain my work in developing those functions and demonstrating how they are used.

## Parse `cced`, calculate `singles_hist`

[singles_histogram.ipynb](analysis/singles_histogram.ipynb): Here I calculate a histogram of singles count rate events vs. $\Delta t$, the time between the fission chamber event and the corresponding detector interaction. The histogram has the following structure: 

* Dimension 0: particle type (0=n,1=g)  
* Dimension 1: detector channel (need to build a dict_det_to_index for this), 45 in length  
* Dimension 2: dt bin for detector, dt_bin_edges = -300 to 300 in .25 ns steps

## Parse `cced`, produce `bicorr`

[generate_bicorr_from_cced.ipynb](analysis/generate_bicorr_from_cced.ipynb): This notebook identifies bicorrelation events from the `cced` list-mode file of all interactions and generates a `bicorr` file, which is a list-mode log of all bicorrelation events. These two files look like this:

`cced` file (columns are `event`, `detector`, `particle_type`, `time`, `integral`, `height`):

    1    7 2 60.43050003051757812500    1.65172    0.20165
    1    40 1 -237.56460189819335937500    0.36266    0.03698
    2    0 2 56.02870178222656250000    1.00000    0.37657
    2    32 2 55.86729812622070312500    1.00000    0.38003
    2    33 2 76.49300003051757812500    0.17479    0.03698
    3    24 2 58.41870117187500000000    0.90767    0.12300
    3    31 2 68.34080123901367187500    0.25033    0.04415
    4    10 1 60.55670166015625000000    6.73639    0.82892

`bicorr` file (columns are `event`, `det1ch`, `det1par`, `det1t`, `det2ch`, `det2par`, `det2t`):

    16  22  2  -0.143900418213  30  2  -143.509488085
    16  22  2  -0.143900418213  43  2  11.6941293987
    16  30  2  -143.509488085  43  2  11.6941293987
    44  29  1  40.8700744099  37  1  46.0955163235
    52  10  2  238.164701554  30  2  -0.665989855469
    76  10  2  -80.4918002156  14  2  -3.35047062158
    87  4  1  33.0194278652  38  2  -1.08306861172
    93  19  1  54.4448179592  20  2  -0.818649601318
    98  7  2  -86.5709508137  34  2  -0.624976884033
    
* Produce `bicorr` from `cced` in folder `1`: `bicorr.generate_bicorr()`
* Produce `bicorr` from `cced` files in folders `1` to `5`: `bicorr.generate_bicorr(1,6)`

## Parse `bicorr`, produce `bicorr_hist_master`

[build_bicorr_hist_master.ipynb](analysis/build_bicorr_hist_master.ipynb): Produce the two-dimensional bicorrelation histogram, `bicorr_hist_master` of counts vs. $\Delta_1$ vs. $\Delta_2$ from the `bicorr` file. `bicorr_hist_master` has four dimensions:

* 0: 990 in length: detector pair index (from `det_df`)  
* 1: 4 in length: interaction type (0=nn, 1=np, 2=pn, 3=pp)  
* 2: Default 1000 in length: $\Delta t$ for detector 1, from `dt_bin_edges`  
* 3: Default 1000 in length: $\Delta t$ for detector 2, from `dt_bin_edges`  

The default time range and in `bicorr_hist_master` is 0.25 ns over a time range of -50 to 200 ns. This produces an array approximately 15 GB in size. It can be shrunk and stored to disk using the sparse matrix technique (described next). 

To run:

* Load `det_df`: `det_df = bicorr.load_det_df(filepath=r'filepath')`
* If different time bins are desired, generate with `dt_bin_edges, num_dt_bins = bicorr.build_dt_bin_edges(dt_min=-50,dt_max=200,dt_step=0.25,print_flag=False)`
* Run it, specifying folders
    * Run for folder `1`: `bicorr_hist_master = bicorr.build_bhm(det_df, dt_bin_edges = dt_bin_edges)[0]`
    * Run for folders `1` to `5`: `bicorr_hist_master = bicorr.build_bhm(det_df, 1, 6, dt_bin_edges = dt_bin_edges)[0]`
* If `save_flag = True`, will store a sparse matrix version to disk as explained in next section

[implement_sparse_matrix.ipynb](analysis/implement_sparse_matrix.ipynb): Convert `bicorr_hist_master` from a numpy array to a sparse matrix, in which I only store the indices and values of each nonzero element. This cuts down the file size from 1 GB to 30 MB for 1 ns time binning and from 15 GB to 0.5 GB for 0.25 ns time binning.

* Store `bicorr_hist_master` as a sparse matrix `sparse_bhm`: `sparse_bhm = bicorr.generate_sparse_bhm(bicorr_hist_master)`
* Save `sparse_bhm` to file: `bicorr.save_sparse_bhm(sparse_bhm, det_df, dt_bin_edges, 'sparse_folder')`
* Load `sparse_bhm` from file: `sparse_bhm, det_df, dt_bin_edges = bicorr.load_sparse_bhm(filepath = 'folder')`
* Revive `bicorr_hist_master` from `sparse_bhm`: `bicorr_hist_master = bicorr.revive_sparse_bhm(sparse_bhm, det_df, dt_bin_edges)`

## Visualize, animate `bicorr_hist_master`

[plot_bicorr_hist.ipynb](analysis/plot_bicorr_hist.ipynb): Plot the two-dimensional histogram which we call a "bicorrelation plot." This is a histogram of the time of flight to detectors A and B, `Delta t_A` and `Delta t_B`. The following `.gif` shows the bicorrelation plot vs. angle between detectors.

![analysis/fig/all.gif](analysis/fig/all.gif)

[build_det_df_angles_pairs.ipynb](analysis/build_det_df_angles_pairs.ipynb): Build, store, reload the `pandas` dataframe that organizes detector channel number and stores angle for each pair. Indices correspond to `bicorr_hist_master`. `.csv` of dataframe: [det_df_pairs_angles.csv](analysis/det_df_pairs_angles.csv).

[detector_pair_angles.ipynb](analysis/detector_pair_angles.ipynb): Explore the distribution of angles between detector pairs and produce a function that selects detector pairs within a given angle range. The angles between detector pairs are shown here:

![fig/angle_btw_pairs.png](fig/angle_btw_pairs.png)

[coarsen_time_and_normalization.ipynb](analysis/coarsen_time_and_normalization.ipynb): Coarsen the time binning for bins wider than the default 0.25 ns, and normalize the number of counts in the bicorrelation plot to the number of fissions, the number of detector pairs, and the time bin width. The following two bicorrelation plots compare normalized distributions with 0.25 and 4 ns bin widths. 

![analysis/fig/compare_coarse_normd.png](analysis/fig/compare_coarse_normd.png)





