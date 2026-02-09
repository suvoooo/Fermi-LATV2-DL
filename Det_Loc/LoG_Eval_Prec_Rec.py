'''
# once we ran 'LoG_r8_S6_MaskCenter.py'
# we have a list of sources with source centers 
# here we systematically find TP. FP. FN
# followed same structure as our previous work 
# and implemented by Boris, Gulli, Sascha (https://github.com/bapanes/AutoSourceID)
'''
import sys, os
import numpy as np
import pandas as pd

from numpy import genfromtxt
import math as mt
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

##########################################

#############################################
#astropy physical distance in degrees
#############################################

def distance_degrees(lon1, lat1, lon2, lat2):
    ra_1 = lon1*u.degree
    de_1 = lat1*u.degree
                    
    ra_2 = lon2*u.degree
    de_2 = lat2*u.degree
                    
    c1 = SkyCoord(ra = ra_1, dec = de_1)
    c2 = SkyCoord(ra = ra_2, dec = de_2)
        
    distance_c1c2 = c1.separation(c2).degree

    return distance_c1c2 

#####################################
#cartesian distance in pixels
#####################################
def distance_pixel(pred_yc, pred_xc, test_yc, test_xc):
    return mt.sqrt(pow(pred_yc - test_yc, 2) + pow(pred_xc - test_xc, 2))


#################################################################
#stats about tp, fp and fn for the pair test and pred
#################################################################

def stats_tp_fp_fn(test, pred, probability_threshold = 0.5, 
                   distance_degrees_threshold = 0.3, bl_alg=False):

    #we copy the test array in order to delete elements without changing
    #the original test array 
    test_aux = np.copy(test)

    #in the following list we want to save relevant info for plots
    list_source_tp_fp_fn = []

    #These indices correspond to the last version of our data generation
    #and unet-kmeans predictions, which are the same after removal of source duplicates which
    #appear in overlapping patches
    # ids below are col ids
    test_patch_idx = 0
    
    test_xmin_idx = 1
    test_xmax_idx = 2
    test_ymin_idx = 3
    test_ymax_idx = 4

    test_lon_patch_idx = 6
    test_lat_patch_idx = 7
    
    #without snr
    test_flux_1000_ps_idx = 8
    test_lon_ps_idx = 9
    test_lat_ps_idx = 10
    test_cat_idx = 11

    test_flux_10000_ps_idx = 12

    #SNR column, created by this algorithm
    test_snr_box_idx = 13
    test_sbr_box_idx = 14
    
    #class extras
    test_class_ps_idx = 5
    
    #with snr
    #test_flux_ps_idx = 9
    #test_lon_ps_idx = 10
    #test_lat_ps_idx = 11
    #test_cat_idx = 12

    #prediction indices in the format uk-full
    #we have to figure out how to introduce this format in centroidnet prediction output
    pred_patch_idx = 1

    pred_yc_idx = 2 #1
    pred_xc_idx = 3 #2

    pred_classId_idx = 4 #3

    pred_probability_idx = 5 #4
    
    pred_lon_ps_idx = 8 #7
    pred_lat_ps_idx = 9 #8
    pred_cat_idx = 10 #9

    pred_snr_box_idx = 11 #10
    pred_sbr_box_idx = 12 #11
    
    #distance threshold in degrees
    #distance_degrees_threshold = 0.3
    #distance_degrees_threshold = 0.6

    #global counters for tp, fp and fn
    tp_con = 0
    fp_con = 0
    fn_con = 0

    #list of border lines in the test are defined as those without a proper box around
    #such that we can evaluate them with the classification algorithm
    #these points are eliminated in advance in order to avoid false negatives
    #from border lines too
    list_of_bl = []

    # xsize = 64 # should this change ? 
    xsize = 128
    r_b = 9
    box_inf = 4
    box_sup = 5
    for aux_con in range(len(test_aux)):

        test_yc = int((test_aux[aux_con, test_ymin_idx] + test_aux[aux_con, test_ymax_idx])//2)
        test_xc = int((test_aux[aux_con, test_xmin_idx] + test_aux[aux_con, test_xmax_idx])//2)

        xmin_b, xmax_b, ymin_b, ymax_b = max(0,test_xc-box_inf), min(xsize, test_xc+box_sup),\
                                         max(0,test_yc-box_inf), min(xsize, test_yc+box_sup)
        
        #we get rid of test border line stars since we do not want to count them as false negatives
        if (((xmax_b-xmin_b) != r_b) or ((ymax_b-ymin_b) != r_b)):
            list_of_bl.append(aux_con)

    if bl_alg:
        test_aux = np.delete(test_aux, list_of_bl, 0)

    #we loop over each one of the predictions in the ps basis
    #pred_con is not the patch number
    for pred_con in range(len(pred)):

        pred_cat = pred[pred_con, pred_cat_idx]
        pred_patch = pred[pred_con, pred_patch_idx]

        # print ('pred cat and pred patch: ', pred_cat, pred_patch) # just for check bug
        
        pred_lon = pred[pred_con, pred_lon_ps_idx]
        pred_lat = pred[pred_con, pred_lat_ps_idx]

        # print ('pred lon and lat: ', pred_lon, pred_lat) # bug check

        pred_id = pred[pred_con, pred_classId_idx] 
        
        pred_yc = pred[pred_con, pred_yc_idx]
        pred_xc = pred[pred_con, pred_xc_idx]

        # print ('pred yc and xc: ', pred_yc, pred_xc) # bug check

        pred_snr_box = pred[pred_con, pred_snr_box_idx]
        pred_sbr_box = pred[pred_con, pred_sbr_box_idx]
        
        #borderline in pred is just skipped
        xmin_b, xmax_b, ymin_b, ymax_b = max(0,int(pred_xc)-box_inf),min(xsize,int(pred_xc)+box_sup),\
                                         max(0,int(pred_yc)-box_inf),min(xsize,int(pred_yc)+box_sup)
        
        #we get rid of test border line stars since we do not want to count them as false negatives
        if bl_alg:
            if (((xmax_b-xmin_b) != r_b) or ((ymax_b-ymin_b) != r_b)):
                continue
        
        pred_probability = pred[pred_con, pred_probability_idx]
                    
        #we initialize the true positive flag as 1 in order
        #to indicate that there is not a tp yet
        tp_ban = 1

        #here we define the variables that we want to record for TP MAXIMUM flux                 
        pred_yc_max = pred_yc
        pred_xc_max = pred_xc
        pred_probability_max = pred_probability
        pred_snr_box_max = pred_snr_box
        pred_sbr_box_max = pred_sbr_box
        
        test_cat_max = pred_cat
        test_patch_max = pred_patch
        
        #the following variables only make sense when we find a potential TP
        pred_test_dg_max = 0
        pred_test_px_max = 0

        test_lon_max = 0
        test_lat_max = 0
                        
        test_flux_1000_max = 0
        test_flux_10000_max = 0

        test_yc_max = 0
        test_xc_max = 0

        test_lon_patch_max = 0
        test_lat_patch_max = 0

        test_id_max = -1

        test_snr_box_max = 0
        test_sbr_box_max = 0
        
        aux_con_flux_max = 0

        #here we define the variables for FP matches with MINIMUM distance variables
        pred_yc_min = pred_yc
        pred_xc_min = pred_xc
        pred_probability_min = pred_probability
        pred_snr_box_min = pred_snr_box
        pred_sbr_box_min = pred_sbr_box

        test_cat_min = pred_cat
        test_patch_min = pred_patch

        #the variables below only make sense when we find the nearest FP
        pred_test_dg_min = 1000
        pred_test_px_min = 1000

        test_lon_min = 0
        test_lat_min = 0
                        
        test_flux_1000_min = 0   
        test_flux_10000_min = 0       
                        
        test_yc_min = 0
        test_xc_min = 0

        test_lon_patch_min = 0
        test_lat_patch_min = 0

        test_id_min = -1
        
        test_snr_box_min = 0
        test_sbr_box_min = 0
        
        aux_con_flux_min = 0

        #for each prediction we loop over each of the true sources in test
        #until we find a true positive

        for aux_con in range(len(test_aux)):

            test_cat = test_aux[aux_con, test_cat_idx]
            test_patch = test_aux[aux_con, test_patch_idx]

            # print ('test cat and patch: ', test_cat, test_patch) # bug check
            
            test_lon = test_aux[aux_con, test_lon_ps_idx]
            test_lat = test_aux[aux_con, test_lat_ps_idx]

            # print ('test lon and lat: ', test_lon, test_lat) # bug check
            
            test_flux_1000  = test_aux[aux_con, test_flux_1000_ps_idx]
            test_flux_10000 = test_aux[aux_con, test_flux_10000_ps_idx]
            
            test_yc = int((test_aux[aux_con, test_ymin_idx] + test_aux[aux_con, test_ymax_idx])/2)
            test_xc = int((test_aux[aux_con, test_xmin_idx] + test_aux[aux_con, test_xmax_idx])/2)

            # print ('test yc and xc: ', test_yc, test_xc) # bug check
            
            test_id = test_aux[aux_con, test_class_ps_idx]

            test_lon_patch = test_aux[aux_con, test_lon_patch_idx]
            test_lat_patch = test_aux[aux_con, test_lat_patch_idx]

            # print ('test lon and lat: ', test_lon, test_lat) # bug check

            test_snr_box = test_aux[aux_con, test_snr_box_idx]
            test_sbr_box = test_aux[aux_con, test_sbr_box_idx]
            
            #print(pred_cat == test_cat)
                    
            #first, we have to ensure that we are comparing between pred and true sources
            #in the same catalogs
            #if (pred_cat == test_cat):
            if (pred_patch == test_patch): 

            #if True:
                #here we compute the geometrical distance between true and predicted sources
                pred_test_dg = distance_degrees(pred_lon, pred_lat, test_lon, test_lat)

                # print ('pred test deg: ', pred_test_dg) # bug check

                #distance in pixels
                pred_test_px = distance_pixel(pred_yc, pred_xc, test_yc, test_xc)

                # print ('pred test pix: ', pred_test_px) # bug check

                #print(pred_con, pred_test_dg, pred_probability)
                if (pred_test_dg <= distance_degrees_threshold and pred_probability >= probability_threshold and pred_id >= 0):

                    #swith the true positive banner to tp = 0
                    tp_ban = 0

                    #for more than one TP match we choose the source with the highest flux
                    if test_flux_1000 > test_flux_1000_max:
                        pred_test_dg_max = pred_test_dg
                        pred_test_px_max = pred_test_px

                        test_lon_max = test_lon
                        test_lat_max = test_lat
                        
                        test_flux_1000_max = test_flux_1000
                        test_flux_10000_max = test_flux_10000

                        test_cat_max = test_cat
                        test_patch_max = test_patch
                        
                        pred_yc_max = pred_yc
                        pred_xc_max = pred_xc

                        test_yc_max = test_yc
                        test_xc_max = test_xc

                        test_lon_patch_max = test_lon_patch
                        test_lat_patch_max = test_lat_patch

                        test_id_max = test_id

                        test_snr_box_max = test_snr_box
                        pred_snr_box_max = pred_snr_box

                        test_sbr_box_max = test_sbr_box
                        pred_sbr_box_max = pred_sbr_box
                        
                        pred_probability_max = pred_probability

                        aux_con_flux_max = aux_con
                else:
                    #for FP matches (dg>0.3) we consider the nearest one
                    if (pred_test_dg < pred_test_dg_min):

                        pred_test_dg_min = pred_test_dg
                        pred_test_px_min = pred_test_px

                        test_lon_min = test_lon
                        test_lat_min = test_lat
                        
                        test_flux_1000_min = test_flux_1000   
                        test_flux_10000_min = test_flux_10000
                        
                        test_cat_min = test_cat
                        test_patch_min = test_patch
                        
                        pred_yc_min = pred_yc
                        pred_xc_min = pred_xc

                        test_yc_min = test_yc
                        test_xc_min = test_xc

                        test_lon_patch_min = test_lon_patch
                        test_lat_patch_min = test_lat_patch

                        test_id_min = test_id
                        
                        test_snr_box_min = test_snr_box
                        pred_snr_box_min = pred_snr_box

                        test_sbr_box_min = test_sbr_box
                        pred_sbr_box_min = pred_sbr_box
                      
                        pred_probability_min = pred_probability
            
        #if the tp_ban does not change from 1, the predicted source is a false positive
        #anyway, we save the information of the closest source from the true
        #be careful since the distance in pixel can be computed through different patches
        if (tp_ban == 1):

            test_lon = test_lon_min
            test_lat = test_lat_min
            
            test_flux_1000  = test_flux_1000_min
            test_flux_10000 = test_flux_10000_min
            
            pred_test_dg = pred_test_dg_min
            pred_test_px = pred_test_px_min

            test_cat = test_cat_min
            test_patch = test_patch_min
            
            pred_yc = pred_yc_min
            pred_xc = pred_xc_min

            test_yc = test_yc_min
            test_xc = test_xc_min

            test_lon_patch = test_lon_patch_min
            test_lat_patch = test_lat_patch_min

            test_snr_box = test_snr_box_min
            pred_snr_box = pred_snr_box_min
            
            test_sbr_box = test_sbr_box_min
            pred_sbr_box = pred_sbr_box_min
            
            pred_probability_min = pred_probability
            
            #false positive should be classified as fake
            test_id = test_id_min
            test_id_ps_fake = 4

            #add 1 to false possitives
            fp_con = fp_con + 1
            
            list_source_tp_fp_fn.append([tp_ban, pred_con, pred_lon, pred_lat, test_lon, test_lat,\
                                         pred_test_dg, pred_test_px, test_flux_1000,\
                                         test_cat, test_patch, pred_yc, pred_xc, test_id_ps_fake, test_yc, test_xc,\
                                         test_lon_patch, test_lat_patch, pred_probability, test_flux_10000, pred_snr_box,\
                                         test_id, test_snr_box, pred_sbr_box, test_sbr_box])

            print("false positive: %d %d (%1.2f, %1.2f) (%1.2f, %1.2f) %1.2f %1.2f %1.2e %1.2e"%(tp_ban, pred_con,\
                   pred_lon, pred_lat, test_lon, test_lat,\
                   pred_test_dg, pred_test_px, test_flux_1000, pred_snr_box)) 

        #true positives matched to the highest flux
        if (tp_ban == 0):
            pred_test_dg = pred_test_dg_max
            pred_test_px = pred_test_px_max

            test_lon = test_lon_max
            test_lat = test_lat_max
                        
            test_flux_1000  = test_flux_1000_max
            test_flux_10000 = test_flux_10000_max

            test_cat = test_cat_max
            test_patch = test_patch_max
                        
            pred_yc = pred_yc_max
            pred_xc = pred_xc_max

            test_yc = test_yc_max
            test_xc = test_xc_max

            test_lon_patch = test_lon_patch_max
            test_lat_patch = test_lat_patch_max

            test_snr_box = test_snr_box_max
            pred_snr_box = pred_snr_box_max

            test_sbr_box = test_sbr_box_max
            pred_sbr_box = pred_sbr_box_max

            pred_probability = pred_probability_max

            aux_con_flux = aux_con_flux_max

            #false positive should be classified as fake
            test_id = test_id_max
            test_id_ps_fake = test_id

            #add 1 to false possitives
            tp_con = tp_con + 1
            
            list_source_tp_fp_fn.append([tp_ban, pred_con, pred_lon, pred_lat, test_lon, test_lat,\
                                         pred_test_dg, pred_test_px, test_flux_1000,\
                                         test_cat, test_patch, pred_yc, pred_xc, test_id_ps_fake, test_yc, test_xc,\
                                         test_lon_patch, test_lat_patch, pred_probability, test_flux_10000, pred_snr_box,\
                                         test_id, test_snr_box, pred_sbr_box, test_sbr_box])

            print("true positive: %d %d (%1.2f, %1.2f) (%1.2f, %1.2f) %1.2f %1.2f %1.2e %1.2e"%(tp_ban, pred_con,\
                   pred_lon, pred_lat, test_lon, test_lat,\
                   pred_test_dg, pred_test_px, test_flux_1000, pred_snr_box)) 

            #We get rid of true positives in order to assume that the rest are false negatives
            test_aux = np.delete(test_aux, aux_con_flux, 0)

    #finally, we count as false negatives all the surviving sources from the true 
    fn_con = len(test_aux)

    tp_ban = 2
    pred_con = 0
    pred_lon = 0
    pred_lat = 0
    pred_test_dg = 0
    pred_test_px = 0

    pred_yc = 0
    pred_xc = 0
    
    for aux_con in range(len(test_aux)):

        test_lon = test_aux[aux_con, test_lon_ps_idx]
        test_lat = test_aux[aux_con, test_lat_ps_idx]
        
        test_flux_1000  = test_aux[aux_con, test_flux_1000_ps_idx]                 
        test_flux_10000 = test_aux[aux_con, test_flux_10000_ps_idx]
        
        test_id = test_aux[aux_con, test_class_ps_idx]
        test_cat =  test_aux[aux_con, test_cat_idx]
        test_patch = test_aux[aux_con, test_patch_idx]

        test_yc = int((test_aux[aux_con, test_ymin_idx] + test_aux[aux_con, test_ymax_idx])/2)
        test_xc = int((test_aux[aux_con, test_xmin_idx] + test_aux[aux_con, test_xmax_idx])/2)

        test_lon_patch = test_aux[aux_con, test_lon_patch_idx]
        test_lat_patch = test_aux[aux_con, test_lat_patch_idx]

        test_snr_box = test_aux[aux_con, test_snr_box_idx]
        test_sbr_box = test_aux[aux_con, test_sbr_box_idx]

        pred_probability = 1
        
        list_source_tp_fp_fn.append([tp_ban, pred_con, pred_lon, pred_lat, test_lon, test_lat,\
                                     pred_test_dg, pred_test_px, test_flux_1000,\
                                     test_cat, test_patch, pred_yc, pred_xc, test_id, test_yc, test_xc,\
                                     test_lon_patch, test_lat_patch, pred_probability, test_flux_10000, test_snr_box,\
                                     test_id, test_snr_box, test_sbr_box, test_sbr_box])

        #print("false negative: %d %d (%1.2f, %1.2f) (%1.2f, %1.2f) %1.2f %1.2f %1.2e %1.2e"%(tp_ban, pred_con,\
        #    pred_lon, pred_lat, test_lon, test_lat,\
        #    pred_test_dg, pred_test_px, test_flux_1000, test_snr_box))

        print("false negative:", (tp_ban, pred_con,\
            pred_lon, pred_lat, test_lon, test_lat,\
            pred_test_dg, pred_test_px, test_flux_1000, test_snr_box, test_sbr_box)) 
             
    return np.array(list_source_tp_fp_fn), tp_con, fp_con, fn_con


###################
# collecting counts
###################
def collect_counts_at_loc(inp_ims, comp, en_bins_fac, loc_y, loc_x):
    '''
    :param inp_ims: list of input arrays for different bins ordered from highest to lowest bins
    :param comp: 0: iem, 1: bll, 2: fsrq, 3: pwn, 4: psr
    :param en_bins_fac: list of energy bin factors: [2, 1, 0.5, 0.25, 0.25]
    '''
    comp_7_20 = inp_ims[0][comp, int(loc_y*en_bins_fac[0]), int(loc_x*en_bins_fac[0]), :].sum()
    comp_2_7 = inp_ims[1][comp, int(loc_y*en_bins_fac[1]), int(loc_x*en_bins_fac[1]), :].sum()
    comp_1_2 = inp_ims[2][comp, int(loc_y*en_bins_fac[2]), int(loc_x*en_bins_fac[2]), :].sum()
    comp_0d5_1 = inp_ims[3][comp, int(loc_y*en_bins_fac[3]), int(loc_x*en_bins_fac[3]), :].sum()
    comp_0d3_0d5 = inp_ims[4][comp, int(loc_y*en_bins_fac[4]), int(loc_x*en_bins_fac[4]), :].sum()
    all_bins_comp = comp_7_20 + comp_2_7 + comp_1_2 + comp_0d5_1 + comp_0d3_0d5
    return all_bins_comp

def collect_counts_at_loc_varPos(inp_ims, comp, loc_y_min, loc_y_max, loc_x_min, loc_x_max):
    '''
    same as before but we use different location lists
    from true csvs; so no multiplicative factors, just true locations
    '''
    comp_7_20 = inp_ims[0][comp, int((loc_y_min[0] + loc_y_max[0])//2), 
                           int((loc_x_min[0] + loc_x_max[0])//2), :].sum()
    comp_2_7 = inp_ims[1][comp, int((loc_y_min[1] + loc_y_max[1])//2), 
                           int((loc_x_min[1] + loc_x_max[1])//2), :].sum()
    comp_1_2 = inp_ims[2][comp, int((loc_y_min[2] + loc_y_max[2])//2), 
                           int((loc_x_min[2] + loc_x_max[2])//2), :].sum()
    comp_0d5_1 = inp_ims[3][comp, int((loc_y_min[3] + loc_y_max[3])//2), 
                           int((loc_x_min[3] + loc_x_max[3])//2), :].sum()
    comp_0d3_0d5 = inp_ims[4][comp, int((loc_y_min[4] + loc_y_max[4])//2), 
                           int((loc_x_min[4] + loc_x_max[4])//2), :].sum()
    all_bins_comp = comp_7_20 + comp_2_7 + comp_1_2 + comp_0d5_1 + comp_0d3_0d5
    return all_bins_comp


path_to_file='/d6/CAC/sbhattacharyya/Downloads/ps_data_Roberto'
im2_7_path = "test_im_iem_psr_bll2_7_patch768"
im7_20_path = "test_im_iem_psr_bll_fsrq_pwn7_20_patch768_rad8"

cat_id = '910'
# path_to_F1_csvs = path_to_file + "/test_catalog_F1_Roberto_csvs"

test_file_name7_20 = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs" + "/test_catalog%s_R_7_20_iem_bll_f_pw_ps_rad8.csv"%(cat_id)
test_file_name2_7 = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs"  + "/test_catalog%s_R_2_7_iem_bll_f_pw_ps.csv"%(cat_id)
test_file_name1_2 = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs" + "/test_catalog%s_R_1_2_iem_bll_f_pw_ps.csv"%(cat_id)

test_data = pd.read_csv(test_file_name7_20) 
test_data1_2 = pd.read_csv(test_file_name1_2)

test_data_aux = np.array(test_data.iloc[:, :])
test_data_aux1_2 = np.array(test_data1_2.iloc[:, :])

print ('test data aux shape: ', np.shape(test_data_aux), np.shape(test_data_aux1_2))


test_data_array = np.empty((test_data_aux.shape[0], test_data_aux.shape[1] + 2), dtype=object)
test_data_array1_2 = np.empty((test_data_aux1_2.shape[0], test_data_aux1_2.shape[1] + 2), dtype=object)

print ('shape of test data array: ', np.shape(test_data_array), np.shape(test_data_array1_2))

test_data_array[:, :-2] = test_data_aux[:, :]
test_data_array1_2[:,:-2] = test_data_aux1_2[:, :]

print ('check test data array, len', test_data_array[8:10], len(test_data_array), '\n', test_data_array1_2[8:10], len(test_data_array1_2))

###############################################
# Main Part
###############################################

th_s = '0d1'
overlap_s = '0d2'
min_sigma, max_sigma = 6.0, 6.0

def main():
    #path_to_data = f"/home/bapanes/Research-Now/Gamma-Ray-Point-Source-Detector"
    #path_to_file='/content/drive/My Drive/Colab Notebooks/Segmentation/Fermi_train'
    path_to_pred = f"{path_to_file}/outputs-csv"
    

    #summary 
    #summarize overall precision, recall for a full test catalog
    global_stats_file = "global_stats_LoG_MinMaxSig%d_overlap%s_th0d10_%sR_weight0_40_1_60_rad8_allSources_Mask2_7GeV.csv"%(max_sigma, 
                                                                                                                            overlap_s, 
                                                                                                                            cat_id)
    header_line = "dataset, precision, recall\n"
    
    # f1 = open(os.path.join(path_to_pred, global_stats_file), "a+")
    f1 = open(os.path.join(path_to_pred, global_stats_file), "a+")
    f1.writelines(header_line)
    f1.close()


    # this is common for UNEK 
    # path_to_test = f"{path_to_data}/data-test/data-768-F0-B1"
    # test_file_name = f"{path_to_test}/test.csv"
    test_file_name7_20 = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs" + "/test_catalog%s_R_7_20_iem_bll_f_pw_ps_rad8.csv"%(cat_id)	
    test_file_name = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs"  + "/test_catalog%s_R_2_7_iem_bll_f_pw_ps.csv"%(cat_id)
    test_file_name1_2 = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs"  + "/test_catalog%s_R_1_2_iem_bll_f_pw_ps.csv"%(cat_id)
    test_file_name0d5_1 = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs"  + "/test_catalog%s_R_0d5_1_iem_bll_f_pw_ps.csv"%(cat_id)
    test_file_name0d3_0d5 = path_to_file + "/patch_bll_iem_psr_fsrq_pwn_csvs"  + "/test_catalog%s_R_0d3_0d5_iem_bll_f_pw_ps.csv"%(cat_id)


    stat_file_name = f"{path_to_pred}/LoG_eval_test_catalog{cat_id}R_5inputs_th{th_s}_weight0_40_1_60_rad8_MinMaxSig{int(max_sigma)}_overlap{overlap_s}_allSources_Mask2_7GeV.csv"
    #uneLoG prediction file from previous run of LoG_r8_S6_MaskCenter.py
    pred_file_name = path_to_file + "/unek_prediction_csvs/LoG_preds_labscore%s_MinMaxSig%d_overlap%s_p768_iem_bll_psr_fsrq_pwn1_%sR_weight0_40_1_60_Mask2_7GeV_rad8.csv"%(th_s, max_sigma, 
                                                                                                                                                                           overlap_s, 
                                                                                                                                                                           cat_id) 
    


    
    #############################################################
    #open csv files and transform to arrays
    #############################################################
    
    test_data7_20 = pd.read_csv(test_file_name7_20)
    test_data = pd.read_csv(test_file_name) 
    test_data1_2 = pd.read_csv(test_file_name1_2)
    test_data0d5_1 = pd.read_csv(test_file_name0d5_1)
    test_data0d3_0d5 = pd.read_csv(test_file_name0d3_0d5)

    test_data_aux7_20 = np.array(test_data7_20.iloc[:, :]) 	
    test_data_aux = np.array(test_data.iloc[:, :])
    test_data_aux1_2 = np.array(test_data1_2.iloc[:, :])
    test_data_aux0d5_1 = np.array(test_data0d5_1.iloc[:, :])
    test_data_aux0d3_0d5 = np.array(test_data0d3_0d5.iloc[:, :])

    #now we add the SNR column
    test_data_array7_20 = np.empty((test_data_aux7_20.shape[0], test_data_aux7_20.shape[1] + 2), dtype=object)        
    test_data_array7_20[:,:-2] = test_data_aux7_20[:, :]	

    test_data_array = np.empty((test_data_aux.shape[0], test_data_aux.shape[1] + 2), dtype=object)        
    test_data_array[:,:-2] = test_data_aux[:, :]

    test_data_array1_2 = np.empty((test_data_aux1_2.shape[0], test_data_aux1_2.shape[1] + 2), dtype=object)        
    test_data_array1_2[:,:-2] = test_data_aux1_2[:,:]

    test_data_array0d5_1 = np.empty((test_data_aux0d5_1.shape[0], test_data_aux0d5_1.shape[1] + 2), dtype=object)        
    test_data_array0d5_1[:,:-2] = test_data_aux0d5_1[:, :]

    test_data_array0d3_0d5 = np.empty((test_data_aux0d3_0d5.shape[0], test_data_aux0d3_0d5.shape[1] + 2), dtype=object)        
    test_data_array0d3_0d5[:,:-2] = test_data_aux0d3_0d5[:, :]

    print ('shape of test_data array: ', test_data_array7_20.shape)    
    for con_test in range(len(test_data_array7_20)):

        # print ('inside for loop : test data array')
        # X_input = np.load(f"{path_to_test}/{test_data_array[con_test,0]}")
        X_input7_20 = np.load(f"{path_to_file}/test_im_iem_psr_bll_fsrq_pwn7_20_patch768_rad8/{test_data_array[con_test, 0]}")
        X_input2_7 = np.load(f"{path_to_file}/test_im_iem_psr_bll_fsrq_pwn2_7_patch768/{test_data_array[con_test, 0]}")
        X_input1_2 = np.load(f"{path_to_file}/test_im_iem_psr_bll_fsrq_pwn1_2_patch768/{test_data_array[con_test, 0]}")
        X_input0d5_1 = np.load(f"{path_to_file}/test_im_iem_psr_bll_fsrq_pwn0d5_1_patch768/{test_data_array[con_test, 0]}")
        X_input0d3_0d5 = np.load(f"{path_to_file}/test_im_iem_psr_bll_fsrq_pwn0d3_0d5_patch768/{test_data_array[con_test, 0]}")    

        xmin7_20 = test_data_array7_20[con_test, 1]
        xmax7_20 = test_data_array7_20[con_test, 2]
        ymin7_20 = test_data_array7_20[con_test, 3]
        ymax7_20 = test_data_array7_20[con_test, 4]
        
        xmin = test_data_array[con_test, 1]
        xmax = test_data_array[con_test, 2]
        ymin = test_data_array[con_test, 3]
        ymax = test_data_array[con_test, 4]

        
        xmin1_2 = test_data_array1_2[con_test, 1]
        xmax1_2 = test_data_array1_2[con_test, 2]
        ymin1_2 = test_data_array1_2[con_test, 3]
        ymax1_2 = test_data_array1_2[con_test, 4]

        xmin0d5_1 = test_data_array0d5_1[con_test, 1]
        xmax0d5_1 = test_data_array0d5_1[con_test, 2]
        ymin0d5_1 = test_data_array0d5_1[con_test, 3]
        ymax0d5_1 = test_data_array0d5_1[con_test, 4]


        xmin0d3_0d5 = test_data_array0d3_0d5[con_test, 1]
        xmax0d3_0d5 = test_data_array0d3_0d5[con_test, 2]
        ymin0d3_0d5 = test_data_array0d3_0d5[con_test, 3]
        ymax0d3_0d5 = test_data_array0d3_0d5[con_test, 4]

        ps_class = test_data_array7_20[con_test, 5]

        all_y_max_locs = [ymax7_20, ymax, ymax1_2, ymax0d5_1, ymax0d3_0d5]
        all_y_min_locs = [ymin7_20, ymin, ymin1_2, ymin0d5_1, ymin0d3_0d5]
        all_x_min_locs = [xmin7_20, xmin, xmin1_2, xmin0d5_1, xmin0d3_0d5]
        all_x_max_locs = [xmax7_20, xmax, xmax1_2, xmax0d5_1, xmax0d3_0d5]

        IEM_ph_true = collect_counts_at_loc_varPos(inp_ims=[X_input7_20, X_input2_7, X_input1_2, 
                                                    X_input0d5_1, X_input0d3_0d5], comp=0, 
                                                    loc_y_min=all_y_min_locs, loc_y_max=all_y_max_locs, 
                                                    loc_x_min=all_x_min_locs, loc_x_max=all_x_max_locs)

        

        AGN_ph_true = collect_counts_at_loc_varPos(inp_ims=[X_input7_20, X_input2_7, X_input1_2, 
                                                    X_input0d5_1, X_input0d3_0d5], comp=1, 
                                                    loc_y_min=all_y_min_locs, loc_y_max=all_y_max_locs, 
                                                    loc_x_min=all_x_min_locs, loc_x_max=all_x_max_locs)


        FSRQ_ph_true = collect_counts_at_loc_varPos(inp_ims=[X_input7_20, X_input2_7, X_input1_2, 
                                                    X_input0d5_1, X_input0d3_0d5], comp=2, 
                                                    loc_y_min=all_y_min_locs, loc_y_max=all_y_max_locs, 
                                                    loc_x_min=all_x_min_locs, loc_x_max=all_x_max_locs)



        PWN_ph_true = collect_counts_at_loc_varPos(inp_ims=[X_input7_20, X_input2_7, X_input1_2, 
                                                    X_input0d5_1, X_input0d3_0d5], comp=3, 
                                                    loc_y_min=all_y_min_locs, loc_y_max=all_y_max_locs, 
                                                    loc_x_min=all_x_min_locs, loc_x_max=all_x_max_locs)
        

        PSR_ph_true = collect_counts_at_loc_varPos(inp_ims=[X_input7_20, X_input2_7, X_input1_2, 
                                                    X_input0d5_1, X_input0d3_0d5], comp=3, 
                                                    loc_y_min=all_y_min_locs, loc_y_max=all_y_max_locs, 
                                                    loc_x_min=all_x_min_locs, loc_x_max=all_x_max_locs)
            
        SNR_ph_true = 0
        SBR_ph_true = 0
        if IEM_ph_true > 0:
            #AGN point source
            if ps_class==0:
                SNR_ph_true = AGN_ph_true/np.sqrt(PSR_ph_true + IEM_ph_true + AGN_ph_true + FSRQ_ph_true + PWN_ph_true)
                # SNR_ph = AGN_ph/np.sqrt(IEM_ph + AGN_ph)
                SBR_ph_true = AGN_ph_true/(PSR_ph_true + IEM_ph_true + FSRQ_ph_true + PWN_ph_true)    
            #PSR point source
            if ps_class==1:
                SNR_ph_true = PSR_ph_true/np.sqrt(AGN_ph_true + IEM_ph_true + PSR_ph_true + FSRQ_ph_true + PWN_ph_true)

                SBR_ph_true = PSR_ph_true/(AGN_ph_true + IEM_ph_true + FSRQ_ph_true + PWN_ph_true)
            #FSRQ point source
            if ps_class==2:
                SNR_ph_true = FSRQ_ph_true/np.sqrt(AGN_ph_true + IEM_ph_true + PSR_ph_true + FSRQ_ph_true + PWN_ph_true)

                SBR_ph_true = FSRQ_ph_true/(AGN_ph_true + IEM_ph_true + PSR_ph_true + PWN_ph_true)
            #PWN point source
            if ps_class==3:
                SNR_ph_true = PWN_ph_true/np.sqrt(AGN_ph_true + IEM_ph_true + PSR_ph_true + FSRQ_ph_true + PWN_ph_true)

                SBR_ph_true = PWN_ph_true/(AGN_ph_true + IEM_ph_true + PSR_ph_true + FSRQ_ph_true)        

        test_data_array7_20[con_test, -2] = SNR_ph_true
        test_data_array7_20[con_test, -1] = SBR_ph_true
        # print('check test snr sbr: ', SNR_ph, SBR_ph)
    
    #pred data read
    pred_data = pd.read_csv(pred_file_name)

    #select only the 0 and 1 class as potential sources
    pred_data = pred_data[pred_data["class_id"]>=0]          
    pred_data_aux = np.array(pred_data.iloc[:,:])

    #now we add the SNR column
    pred_data_array = np.empty((pred_data_aux.shape[0], pred_data_aux.shape[1] + 2), dtype=object)        
    pred_data_array[:,:-2] = pred_data_aux[:,:]
        
    for con_pred in range(len(pred_data_array)):

        # print ('inside pred data array loop')

        # X_input = np.load(f"{path_to_test}/test_image_{int(pred_data_array[con_pred,0])}.npy")
        im_path7_20 = '/d6/CAC/sbhattacharyya/Downloads/ps_data_Roberto/test_im_iem_psr_bll_fsrq_pwn7_20_patch768_rad8/'
        im_path2_7 = path_to_file + '/test_im_iem_psr_bll_fsrq_pwn2_7_patch768'
        im_path1_2 = path_to_file + '/test_im_iem_psr_bll_fsrq_pwn1_2_patch768'
        im_path0d5_1 = path_to_file + '/test_im_iem_psr_bll_fsrq_pwn0d5_1_patch768'
        im_path0d3_0d5 = path_to_file + '/test_im_iem_psr_bll_fsrq_pwn0d3_0d5_patch768'

       # X_input2_7 = np.load(f"{im_path2_7}/test_image_{int(pred_data_array[con_pred, 1])}.npy")
       # X_input1_2 = np.load(f"{im_path1_2}/test_image_{int(pred_data_array[con_pred, 1])}.npy")
       # X_input0d5_1 = np.load(f"{im_path0d5_1}/test_image_{int(pred_data_array[con_pred, 1])}.npy")
       # X_input0d3_0d5 = np.load(f"{im_path0d3_0d5}/test_image_{int(pred_data_array[con_pred, 1])}.npy")
        X_input7_20 = np.load(f"{im_path7_20}/{pred_data_array[con_pred, 1]}") 
        X_input2_7 = np.load(f"{im_path2_7}/{pred_data_array[con_pred, 1]}")
        X_input1_2 = np.load(f"{im_path1_2}/{pred_data_array[con_pred, 1]}")
        X_input0d5_1 = np.load(f"{im_path0d5_1}/{pred_data_array[con_pred, 1]}")
        X_input0d3_0d5 = np.load(f"{im_path0d3_0d5}/{pred_data_array[con_pred, 1]}")		

        #im_name = 'test_image_%d.npy'%(int(pred_data_array[con_pred, 1]))
        im_name = '%s'%(pred_data_array[con_pred, 1])
        #print(X_input.shape)
            
        yc = int(pred_data_array[con_pred, 2])
        xc = int(pred_data_array[con_pred, 3])
        
        
        IEM_ph = collect_counts_at_loc([X_input7_20, X_input2_7, X_input1_2, X_input0d5_1, X_input0d3_0d5], comp=0, 
                                       en_bins_fac=[2, 1, 0.5, 0.25, 0.25], loc_y=yc, loc_x=xc)
        AGN_ph = collect_counts_at_loc([X_input7_20, X_input2_7, X_input1_2, X_input0d5_1, X_input0d3_0d5], comp=1, 
                                       en_bins_fac=[2, 1, 0.5, 0.25, 0.25], loc_y=yc, loc_x=xc)
        FSRQ_ph = collect_counts_at_loc([X_input7_20, X_input2_7, X_input1_2, X_input0d5_1, X_input0d3_0d5], comp=2, 
                                       en_bins_fac=[2, 1, 0.5, 0.25, 0.25], loc_y=yc, loc_x=xc)
        PWN_ph = collect_counts_at_loc([X_input7_20, X_input2_7, X_input1_2, X_input0d5_1, X_input0d3_0d5], comp=3, 
                                       en_bins_fac=[2, 1, 0.5, 0.25, 0.25], loc_y=yc, loc_x=xc)
        PSR_ph = collect_counts_at_loc([X_input7_20, X_input2_7, X_input1_2, X_input0d5_1, X_input0d3_0d5], comp=4, 
                                       en_bins_fac=[2, 1, 0.5, 0.25, 0.25], loc_y=yc, loc_x=xc)

        

            
        #notice that predicted positions are not directly associated with a given class
        SNR_ph = 0
        SBR_ph = 0
            
        if IEM_ph > 0:
               
            if AGN_ph >= PSR_ph + FSRQ_ph + PWN_ph:
                # SNR_ph = AGN_ph/np.sqrt(PSR_ph + IEM_ph + PSR_ph)
                SNR_ph = AGN_ph/np.sqrt(PSR_ph + IEM_ph + AGN_ph + FSRQ_ph + PWN_ph)
                SBR_ph = AGN_ph/(IEM_ph + PSR_ph + FSRQ_ph + PWN_ph)
                    
            if PSR_ph > AGN_ph + FSRQ_ph + PWN_ph:
                SNR_ph = PSR_ph/np.sqrt(PSR_ph + IEM_ph + AGN_ph + FSRQ_ph + PWN_ph)
                SBR_ph = PSR_ph/(AGN_ph + IEM_ph + FSRQ_ph + PWN_ph)

            if FSRQ_ph > AGN_ph + PSR_ph + PWN_ph:
                SNR_ph = FSRQ_ph/np.sqrt(PSR_ph + IEM_ph + AGN_ph + FSRQ_ph + PWN_ph)
                SBR_ph = FSRQ_ph/(AGN_ph + IEM_ph + PWN_ph + PSR_ph)

            if PWN_ph > AGN_ph + FSRQ_ph + PSR_ph:
                SNR_ph = PWN_ph/np.sqrt(PSR_ph + IEM_ph + AGN_ph + FSRQ_ph + PWN_ph)
                SBR_ph = PWN_ph/(AGN_ph + IEM_ph + FSRQ_ph + PSR_ph)        
                
        pred_data_array[con_pred, -2] = SNR_ph
        pred_data_array[con_pred, -1] = SBR_ph
        # print('check pred snr, sbr: ', SNR_ph, SBR_ph)

    print("test and pred: ",len(test_data), len(pred_data))
        
    #here we could transform the filename to numeric code and then ask
    #for the same patch comparison

    for con in range(test_data_array.shape[0]):
        filename = test_data_array[con, 0]
        test_data_array[con, 0] = int((filename.split('_')[2]).split('.')[0])
    
    for con_h in range(test_data_array7_20.shape[0]):
        filename7_20 = test_data_array7_20[con_h, 0]
        test_data_array7_20[con_h, 0] = int((filename7_20.split('_')[2]).split('.')[0])
    
    for co in range(pred_data_array.shape[0]):
        pred_filename = pred_data_array[co, 1]
        pred_data_array[co, 1] = int((pred_filename.split('_')[2]).split('.')[0])

            
    print("length of test: ", len(test_data_array))
    print("length of pred: ", len(pred_data_array))
    
    ############################################################################
    #calling external function that compute the metrics TP-FP-FN in the ps basis
    ############################################################################

    stat_array, tp_tot, fp_tot, fn_tot = stats_tp_fp_fn(test_data_array, pred_data_array, \
                                                        probability_threshold = 0.0, distance_degrees_threshold = 0.5,\
                                                        bl_alg=False) 
    

    ###############################################
    #Consistency check, both lines should be equal
    ###############################################
    
    print(tp_tot, fp_tot, fn_tot)
    print(stat_array.shape)
    print(len(np.where(stat_array[:, 0]==0)[0]), len(np.where(stat_array[:, 0]==1)[0]), len(np.where(stat_array[:, 0]==2)[0]))
    
    #################################
    #global metrics 	
    #################################
    
    print("Global metric values")
    
    precision = round(tp_tot/(tp_tot + fp_tot)*100, 2)
    recall = round(tp_tot/(tp_tot + fn_tot)*100, 2)

    print('precision: ', precision)
    print('recall:', recall)       
        
    f1 = open(os.path.join(path_to_pred, global_stats_file), "a+")
    value_line = f"test_f0_b1_r4, {precision}, {recall}\n"
    f1.writelines(value_line)
    f1.close()
        
    #############################################################
    #file for plots with from-performance-file-to-figures.ipynb
    #############################################################

    #stat_code
    #0=tp
    #1=fp
    #2=fn

        
    stat_col_names = ["stat_code", "pred_con", "pred_lon", "pred_lat", "test_lon", "test_lat",\
                      "distance_degree", "distance_pixel", "test_flux_1000", "catalog", "patch_number",\
                      "pred_y", "pred_x", "class", "test_y", "test_x", "test_lon_patch", "test_lat_patch",\
                      "probability", "test_flux_10000", "snr_center_pred", "class_ps", "snr_center_test",\
                      "sbr_center_pred", "sbr_center_test"]

    stat_data_output = {stat_col_names[0]:stat_array[:,0], stat_col_names[1]:stat_array[:,1],\
                        stat_col_names[2]:stat_array[:,2], stat_col_names[3]:stat_array[:,3],\
                        stat_col_names[4]:stat_array[:,4], stat_col_names[5]:stat_array[:,5],\
                        stat_col_names[6]:stat_array[:,6], stat_col_names[7]:stat_array[:,7],\
                        stat_col_names[8]:stat_array[:,8], stat_col_names[9]:stat_array[:,9],\
                        stat_col_names[10]:stat_array[:,10], stat_col_names[11]:stat_array[:,11],\
                        stat_col_names[12]:stat_array[:,12], stat_col_names[13]:stat_array[:,13],\
                        stat_col_names[14]:stat_array[:,14], stat_col_names[15]:stat_array[:,15],\
                        stat_col_names[16]:stat_array[:,16], stat_col_names[17]:stat_array[:,17],\
                        stat_col_names[18]:stat_array[:,18], stat_col_names[19]:stat_array[:,19],\
                        stat_col_names[20]:stat_array[:,20], stat_col_names[21]:stat_array[:,21],\
                        stat_col_names[22]:stat_array[:,22], stat_col_names[23]:stat_array[:,23],\
                        stat_col_names[24]:stat_array[:,24]}

    stat_data_frame = pd.DataFrame(data=stat_data_output)
        
    stat_data_frame.to_csv(stat_file_name, sep=',', index=False)

    print("stats output file: ", stat_file_name)


##################################
# Call the Main 
##################################

if __name__ == '__main__':
    main()
