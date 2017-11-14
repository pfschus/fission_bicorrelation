# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input File for Measurement data - Dual_Read, Cross_correlation, Multiplet, Quick_clan programs
# Version 1.3 - 04/03/2013
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
base_name				dataFile		# Base file name for the detector files, digitizer V1720-option
ending					.dat			# Extentsion for the detector files (include dot! i.e. ".bin") V1720 option
starting_folder				1			# folder to start with, default=1, good for rerunning  folder e.g. 11
number_of_folders    	        	12			# Number of folders starting from "starting_folder"
number_of_detectors 		        48			# Number of detectors (digitizer channels)
first_chan				0			# first channel to analyse (default=0!) (not used by cross_correlation.exe)
number_of_pulses    			100000000			# Nr of pulses to analyze, choose number well above biggest file to run ALL data
number_of_points			144			# Number of points in a pulse waveform
minthreshold				0.02824			# Min accaptable pulse (V!)
psd_coefficients	-0.002419     0.19107    0.020134	# Three values A B C : Ax^2+Bx+C, Additional PSD options in PSD section below
psd_coefficients2		7.0749e-3 1.9266e-1 4.682e-2	# Three values A B C : Ax^2+Bx+C
det_psd_coeffs			1 1 1 2 2 1 1 1	1 1 2 1		# 1 number for each channel, assumes 1 if missing. 1=first set, 2=second set
psd_per_chan	1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1	1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1	1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1	# Option to give psd coeffs for each channel in the file psdcoeff.i (default=0).
number_of_headers            3                                  # Number of headers, default=2 (DNNG software); use 6 for CAEN software
volt_dynamic_range           2                                  # Dynamic range in volts (2(def) or 0.5)
bit_dynamic_range           14                                  # Dynamic range in bits (12(def) or 14)
time_per_sample              2                                  # From the sampling rate of the digitizer (4(def) or 2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# USB digitizer options below!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

usb_digitizer			0				# 0 default, 1=using the 4-chan USB digitzer
usb_base_name			wave_data_ch_			# Name base for USB digitizer files
usb_ending			.dat			        # file ending for USB! digitizer
number_of_headers_usb		2				# USB file headers (default=2!)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pulse and program options
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CFD					0.5		# Constant Fraction Discriminator (0.5 default)
start_past_max				12			# Steps past max for start of integration of the tail
steps_past_max				90			# Total number of steps past the max to integrate
tot_start_bef_max			10			# Steps before max for START of TOTAL integration (default = 5! (EJ309))
psd_normalized_time                     2                       # use 1 to multiply the sum of sample points by the time step (default = 0)
points_for_baseline			10			# Num. of points used to determine the offset
num_points_for_cfd			20			# Num. of points backwards to find cfd-level, (default 50, EJ309, higher for HpGe)
double_pulse_fraction			0.20			# Incremental datapoint fraction of max that is considered a double pulse
datapoint_jump				1			# jump for double point cleaning comparison (default 1)
double_pulse_threshold                  0.1                     # Pulse height threshold under which double pulse cleaning uses the fixed value below.
double_pulse_fixed                      0.1                     # Fixed pulse height value used instead of the fraction of the pulse max
neg_pulses				1			# negative pulses (EJ309, default!) (1) or positive (-1) (NaI, CLYC amplified)
write_event_nr				0			# default=0, use 1 to write event trigger nr. in file "eventX" for each chan. X
ph_lower_bin				0			# Lower bin for pulse height histogram
ph_uppper_bin				2			# Upper bin for pulse height histogram
ph_bin_incr				0.01		        # Size of step for pulse height histogram
phd_mevee				0			# 1 to Use MeVee binning on PHD, 0=don't use (uses digitizer Volts) (0 = defualt)
cs_compton				0.300			# given in VOlts, REQUIRED for option 1 above!
pulse_smooth				0			# option to turn on pulse smoothing (1=on!, 0=off!, default=0), (turns on for ALL channels)
smoothp					5			# number of points used in smoothing, e.g. 3-p running average.


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PSD method options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

psd_method			0				# PSD Method to use ('1' enables 2 alternative methods), default=0!
pulse_jitter_clean		0.001			        # value in volts for noise suppression on PSD methods (default=0.001 (V))
psd_meth2_coeff			0.018803   -0.064434   0.22804	# quadratic values A, B, C: Ax^2+Bx+C
psd_meth3_coeff			1.85 23.4982	 		# logarithmic func, 2 parameters [A, B]: log(x^A/B)
psd_conf_range_meth1		0.03				# PSD confidence buffer method 1 (1 value) (default=0.03)
psd_conf_range_meth2		0.035				# PSD confidence buffer method 2 (1 value) (default=0.035)
psd_conf_range_meth3		0.15				# PSD confidence buffer method 3 (1 value) (default=0.15)
psd_ignore			1				# ignore pulse psd analysis (default=0=no), '1' ignores psd, use next row to select what chanel
psd_ignore_chan			1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0	# channels to inactivate PSD analysis on (only used when "psd_ignore=1")
doub_clean_ignore		1				# if "psd_ignore=1" and the channel is selected above then this deactivates double pulse removal (default='1')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADDED OPTIONS FOR DUAL_READ_CFD, cross_correlation_multi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

manual_offset			0				# 0 to NOT use manual offset on one channel, 1 for doing so (default = 0!)
offset_chan			5				# channel to use offset on. Ignored if "manual_offset" = 0
offset_value			96				# offset value to use for manual offset, ignored if "manual_offset" = 0
use_doub_fixed_val		0				# 1 for using a fixed voltage value as a double pulse cleaning, (default=0, uses fraction above)
doub_pul_fixed_val		0.05				# Only used if "1" on line above, (default=0.05(V))
save_CC_phd			0				# write PHD for CC (cross correlation program only!)
keep_bad_pulses			0				# keep clipped pulses for a certain channel (default 0)
keep_bad_channel		0				# channel number for keeping clipped pulses (ToF measurements)
write_double_pulses		0				# (0 = no; 1 = yes) - warning: could produce large file
write_good_pulses		0				# (0 = no; 1 = yes) - warning: WILL! produce large file
integ_ref_point			0				# use ('1') to define a data point around which integration happens INSTEAD of pulse_max (default=0)
ref_dpoint			20				# data point into the waveform to use for integr. bounds, useful for CLYC, NaI (Mark B)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# multi board options below!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

multi_board_conv			0			# 1 for converting multi-board setup data (default=0, meaning no multi-board)
run_dual_read				0			# 1 for dual_reads pulse analysis after conversion (default=1!), 0 to only convert multi-board
multi_base_name				dataFile		# name of data file per board (default=acq_data)
first_board				0			# first board channel (default = 0)
num_boards				3			# number of boards used!
file_per_board				1			# default="1", legacy data used single file for all boards (use "0")
zero_supp				0			# apply zero suppression on pulses (default=0 - No ZS, 1=use ZS)
zs_thresh				0.02			# threshold in Volts to use if using zero suppression (ignored if zero_supp=0)
multi_zs_offset				50			# offset in digitizer units! used on all channels ZS (ignored if zero_supp=0)
run_multi_direct			1			# Run dual_read directly form single board-files ("acq_data_1.dat"), defualt ="0" ("1"=On).
npulses_board				50000000		# Pulses to read in single board file (only active if run_multi_direct="1").


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Accellerator and logical NIM pulses options
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

acc_use				0				# Was a beam pick-off signal used? (normally periodically reoccurring)
beam_ch				5				# channel number that beam signal was on.
beam_period			449				# number of data points between beam pulsing: period(in ns)/(ns per point) (integer!)
use_slope			1				# use predefined pulse slope near CFD edge, can improve timing! (default=0)
slope				0.3301				# slope to use if the option above is 1!


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Multiplet.exe program options! No effect on other programs for the options below!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mult_nfolder			2				# Number of folders to process
mult_ndet			4				# Number of detectors (digitizer channels)
mult_firstch			0				# first channel to analyse (default=0!)
mult_npulses			100000				# Number of pulses (waveforms) to process
mult_twindow			100				# time window (in ns) for correlating pulses to a mutliple (defualt ~100 (ns))
mult_thresh			0				# pulse threshold in digitizer volts (V). Can be used to apply arbitrary thresholds
mult_order_mult			4				# highest order of detailed mixed multiples, "3" or "4" valid options
mult_true_mult			1				# "1" uses true mutliplicity, while "0" used He-3 equivalent with a trigger not counted
mult_write_mult			0				# "1" to write the multiplicity vector in ASCII format (default="0")
mult_write_time			0				# "1" to write the sorted time-stamp file, default="0".


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Quick_clean.exe program options! No effect on other programs for the options below!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clean_file_base			wave_data_ch_			# File name base for the quick_clean program (to apply ZS)
clean_file_end			.dat				# File ending (including the dot, so ".dat" for example)
clean_nfolder			2				# Number of folders to process
clean_ndet			2				# Number of detectors (digitizer channels)
clean_npulses			3000000				# Number of pulses (waveforms) to process
clean_npoints			100				# Number of data points in a waveform
clean_nheader			2				# Number of headers in data (DNNG format, 2 is default!)
clean_ZS_thresh			0.02				# Pulse threshold to be used for zero suppression and data rejection purpose
clean_offset1			96				# Main pulse offset to use for channels identified by a "1" below
clean_offset2			46				# Offset to be used on channels iudentified by a "2" below
clean_pick_off			1 1 1 1 1 2 2 2			# Selects which of "offset1" or "offset2" rto use on each data channel.
clean_write_bad			0				# write all rejected waveforms to separate file for error-control (Default="0")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OPTIONS FOR TCHP ANALYSIS - Eric M. Runs in the Cross_correlation program.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tcph_on			        0				# 1 = on, 0 = off
tcph_all_pairs                  1				# 1 = on, 0 = off
tcph_time_low		        0				# ns, lower time bin
tcph_time_high		        100				# ns, upper time bin
tcph_time_incr		        1				# ns, time incrament
tcph_erg_low		        0				# MeVee, lower light limit
tcph_erg_high		        3				# MeVee, Upper light limit
tcph_erg_incr		        .1				# MeVee, light incrament
