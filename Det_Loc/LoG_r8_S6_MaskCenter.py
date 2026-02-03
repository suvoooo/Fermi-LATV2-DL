'''
once we trained the segmentation network; 
here we load the best segmentation model (based on validation loss and dice coefficient)
to predict masks and find source center with LoG

Train using: train_unet_LAT_MultiBins.py
save the best model.
load it here and use LoG to get predicted source centers. 
'''

import numpy as np
import matplotlib.pyplot as plt

import math as mt
import pandas as pd



from scipy.stats import poisson



from skimage.feature import blob_log # laplacian of gaussian
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K

import glob


###################################
# force tf to use cpu
# if you want!
##################################
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


###################################################
# whether we are using weighted loss to predict
###################################################

weighted_loss = True

def weighted_binary_cross_entropy(weights: dict, from_logits: bool = False):
    #assert 0 in weights
    #assert 1 in weights
    def weighted_cross_entropy_fn(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)

        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(tf_y_true, tf_y_pred, from_logits=from_logits)
        loss = K.mean(tf.multiply(ce, weights_v))
        return loss

    return weighted_cross_entropy_fn



def dice_coeff(y_true, y_pred, smooth=1e-4):
    intersection = K.sum(y_true * y_pred, axis=(0, 1))
    union = K.sum(y_true, axis=(0, 1)) + K.sum(y_pred, axis=(0, 1))
    dice = (2. * intersection + smooth)/(union + smooth)
    return dice




def distance(x0, y0, x1, y1):
    return mt.sqrt((mt.pow(x1-x0, 2) + mt.pow(y1-y0, 2)))




###############################################
# Geometric Utilities
###############################################

def RotMatrixY(psi, isdeg=True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), 0.0, -np.sin(np.radians(psi))], [0.0, 1.0, 0.0],
                         [np.sin(np.radians(psi)), 0.0, np.cos(np.radians(psi))]])
    else:
        return np.array([[np.cos(psi), 0.0, -np.sin(psi)], [0.0, 1.0, 0.0], [np.sin(psi), 0.0, np.cos(psi)]])


def RotMatrixZ(psi, isdeg=True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0],
                         [-np.sin(np.radians(psi)), np.cos(np.radians(psi)), 0.0], [0.0, 0.0, 1.0]])
    else:
        return np.array([[np.cos(psi), np.sin(psi), 0.0], [-np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]])


def sph2xyz(r, theta, phi, isdeg=True):
    if isdeg:
        return np.array([r*np.sin(np.radians(theta))*np.cos(np.radians(phi)),
                         r*np.sin(np.radians(theta))*np.sin(np.radians(phi)), 
                         r*np.cos(np.radians(theta))])
    else:
        return np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), 
                         r*np.cos(theta)])


def xyz2sph(x, y, z, isdeg=True, is_lat=False):
    r = np.sqrt(x*x + y*y + z*z)
    if isdeg:
        phi = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
        if is_lat:
            return np.array([r, lat, phi])
        else:
            return np.array([r, 90. - lat, phi])
    else:
        phi = np.arctan2(y, x)
        lat = np.arctan2(z, np.sqrt(x*x + y*y))
        if is_lat:
            return np.array([r, lat, phi])
        else:
            return np.array([r, np.pi/2.0 - lat, phi])


def get_lb_from_pixel(pixel_id, lb_centre, xsize, isdeg=True, is_lat=True):
    # if input angles are in degree use 'isdeg = True'
    ######### Generate (l,b) coordinate map of 10x10deg patch ######

    # Following the suggestions of CA mail
    # kept as it is from Boris' version
    if (xsize == 100):
        coord_range = np.linspace(-4.95, 4.95, xsize)

    if (xsize == 128):
        coord_range = np.linspace(-4.9609375, 4.9609375, xsize)


    X, Y = np.meshgrid(coord_range, coord_range)
    lonlat_patch = list(zip(np.flip(X.flatten()), Y.flatten()))
    print('wtf lonlat patch: ', len(lonlat_patch))
    ######### Get rotation matrix used to rotate the original centre to (0., 0.) #########
    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))
    #########

    lon_PS_rotated, lat_PS_rotated = lonlat_patch[pixel_id]

    #x_pixel_normal = int(pixel_id % xsize)
    #y_pixel_normal = int(pixel_id / xsize)

    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r.T, xyz_PS_rotated), dtype='float32')
    #xyz_centre = sph2xyz(1., 90. - b_centre, l_centre)
    r, b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS, isdeg=isdeg, is_lat=is_lat)

    if l_PS < 0:
        l_PS = 360 + l_PS

    return l_PS, b_PS


def pixel_id(row, col, xsize_patch):

    return xsize_patch*row + col


catalog_id = '910'

##########################################
# Load Files and Saved U-Net Model
##########################################

path_to_file = '/d6/CAC/sbhattacharyya/Downloads/ps_data_Roberto/'



test_file7_20 = path_to_file + 'patch_bll_iem_psr_fsrq_pwn_csvs/' + 'test_catalog%s_R_7_20_iem_bll_f_pw_ps_rad8.csv'%(catalog_id)
test_data7_20 = pd.read_csv(test_file7_20)

test_file2_7 = path_to_file + 'patch_bll_iem_psr_fsrq_pwn_csvs/' + 'test_catalog%s_R_2_7_iem_bll_f_pw_ps.csv'%(catalog_id)
test_data2_7 = pd.read_csv(test_file2_7)



# list of file names in test data

list_of_names2_7 = sorted(test_data2_7["filename"].unique().tolist())
list_of_names7_20 = sorted(test_data7_20["filename"].unique().tolist())


print('order of names: ', list_of_names2_7[3:6], list_of_names7_20[3:6],)



unet_model_file_poisson = path_to_file + 'checkpoints/unet_mask_try_iem_bll_psr_fsrq_pwn_poisson_rvs_5inputs_p768_fcji0_40_1_60_250epochs_normed_rad8_check_mk2_7GeV.h5'


unet_model = load_model(unet_model_file_poisson, 
                        custom_objects={ 'weighted_cross_entropy_fn': weighted_binary_cross_entropy({0:0.40, 1:0.60}, from_logits = False), 
                                                                 'dice_coeff': dice_coeff})
#


input_bins = 1  # each energy bin seperate, previously it's 5
output_bins = 2
label_score_uk = 0.1  # 0.2
th_s = '0d1'
overlap = 0.2
overlap_s= '0d2'
xsize = 128 # 2-7 GeV

###########
##tested for several sigmas
###########
min_sigma = 6.0
max_sigma = 6.0

intensity_penalty_points_uk = -10

# min and max number of sources to search for
nmin = 1
nmax = 40


###############################################
# Define the Main Function
###############################################


def main():

    #####################################################################################
    # INPUT and OUTPUT files
    #####################################################################################

    # path to results
    path_to_results_file = path_to_file + "unek_prediction_csvs/" + "LoG_preds_labscore%s_MinMaxSig%d_overlap%s_p768_iem_bll_psr_fsrq_pwn1_%sR_weight0_40_1_60_Mask2_7GeV_rad8.csv"%(th_s, 
                                                                                                                                                                                     max_sigma, 
                                                                                                                                                                                     overlap_s, 
                                                                                                                                                                                     catalog_id)

    ########################################################################################
    con_50_plus = 0
    number_of_test_files = len(list_of_names2_7)

    list_file_number = []
    list_catalog = []
    list_lon_c = []
    list_lat_c = []

    list_xc = []
    list_yc = []
    list_lon_ps = []
    list_lat_ps = []
    list_class_id = []
    list_probability = []
    image_file_names = []

    print("evaluation of %d input images" % (number_of_test_files))

    for con_test_file in range(number_of_test_files):
        # read the files
        image_file = list_of_names2_7[con_test_file]

        # get the information about this image_file in particular
        
        sub_df2_7 = test_data2_7[test_data2_7["filename"] == image_file]
        sub_df7_20 =test_data7_20[test_data7_20["filename"] == image_file] 
        

        # for each file in test data there is one lon_c, lat_C
        
        
        lon_c_file2_7 = sub_df2_7["lon_c"].iloc[0]
        lon_c_file7_20 = sub_df7_20["lon_c"].iloc[0]

        
        lat_c_file2_7 = sub_df2_7["lat_c"].iloc[0]
        lat_c_file7_20 = sub_df7_20["lat_c"].iloc[0]

        
        catalog_file2_7 = sub_df2_7["catalog"].iloc[0]
        catalog_file7_20 = sub_df7_20["catalog"].iloc[0]

        filename_ims2_7 = sub_df2_7["filename"].iloc[0]
        filename_ims7_20 = sub_df7_20["filename"].iloc[0]

        print(image_file)
        # image_file_names.append(image_file)

        masks_file = image_file.replace("image", "masks")

        print("image file: ", image_file)
        print("masks file: ", masks_file)

        X1_7_20 = np.zeros((1, 5, int(xsize*2), int(xsize*2), input_bins), dtype=float)
        X1_2_7 = np.zeros((1, 5, int(xsize/1.), int(xsize/1.), input_bins), dtype=float)
        X1_1_2 = np.zeros((1, 5, int(xsize/2.), int(xsize/2.), input_bins), dtype=float) # 5 because agn, psr, iem, fsrq, pwn
        X1_0d5_1 = np.zeros((1, 5, int(xsize/4.), int(xsize/4.), input_bins), dtype=float)
        X1_0d3_0d5 = np.zeros((1, 5, int(xsize/4.), int(xsize/4.), input_bins), dtype=float)

        Xa_0d5_1 = np.zeros((1, int(xsize/4), int(xsize/4), input_bins), dtype=float)
        Xp_0d5_1 = np.zeros((1, int(xsize/4), int(xsize/4), input_bins), dtype=float)

        Xa_0d3_0d5 = np.zeros((1, int(xsize/4), int(xsize/4), input_bins), dtype=float)
        Xp_0d3_0d5 = np.zeros((1, int(xsize/4), int(xsize/4), input_bins), dtype=float)

        Xa_1_2 = np.zeros((1, int(xsize/2.), int(xsize/2.), input_bins), dtype=float)
        Xp_1_2 = np.zeros((1, int(xsize/2.), int(xsize/2.), input_bins), dtype=float)

        Xa_2_7 = np.zeros((1, int(xsize/1.), int(xsize/1.), input_bins), dtype=float)
        Xp_2_7 = np.zeros((1, int(xsize/1.), int(xsize/1.), input_bins), dtype=float)
        Y_2_7 = np.zeros((1, int(xsize/1.), int(xsize/1.), output_bins), dtype=float)

        Xa_7_20 = np.zeros((1, int(xsize*2), int(xsize*2), input_bins), dtype=float)
        Xp_7_20 = np.zeros((1, int(xsize*2), int(xsize*2), input_bins), dtype=float)

        # read new format X data and transform to usual shapes
        # X1[0,:,:,:,:] = np.load(f'{path_to_test}/{image_file}') # previous version
        X1_0d3_0d5[0, :, :, :, :] = np.load(path_to_file+'test_im_iem_psr_bll_fsrq_pwn0d3_0d5_patch768/'+image_file)
        X1_0d5_1[0, :, :, :, :] = np.load(path_to_file+'test_im_iem_psr_bll_fsrq_pwn0d5_1_patch768/'+image_file)
        X1_1_2[0, :, :, :, :] = np.load(path_to_file+'test_im_iem_psr_bll_fsrq_pwn1_2_patch768/'+image_file)
        X1_2_7[0, :, :, :, :] = np.load(path_to_file+'test_im_iem_psr_bll_fsrq_pwn2_7_patch768/'+image_file)
        X1_7_20[0, :, :, :, :] = np.load(path_to_file+'test_im_iem_psr_bll_fsrq_pwn7_20_patch768_rad8/'+image_file)
        
        Y_2_7[0, :, :, :] = np.load(path_to_file+'test_mk_iem_psr_bll_fsrq_pwn2_7_patch768_rad8/'+masks_file)

        Xa_0d3_0d5[0, :, :, :] = X1_0d3_0d5[0, 0, :, :, :] + X1_0d3_0d5[0, 1, :, :, :] + X1_0d3_0d5[0, 2, :, :, :] + X1_0d3_0d5[0, 3, :, :, :] + X1_0d3_0d5[0, 4, :, :, :]
        Xa_0d5_1[0, :, :, :] = X1_0d5_1[0, 0, :, :, :] + X1_0d5_1[0, 1, :, :, :] + X1_0d5_1[0, 2, :, :, :] + X1_0d5_1[0, 3, :, :, :] + X1_0d5_1[0, 4, :, :, :]
        Xa_1_2[0, :, :, :] = X1_1_2[0, 0, :, :, :] + X1_1_2[0, 1, :, :, :] + X1_1_2[0, 2, :, :, :] + X1_1_2[0, 3, :, :, :] + X1_1_2[0, 4, :, :, :]
        Xa_2_7[0, :, :, :] = X1_2_7[0, 0, :, :, :] + X1_2_7[0, 1, :, :, :] + X1_2_7[0, 2, :, :, :] + X1_2_7[0, 3, :, :, :] + X1_2_7[0, 4, :, :, :]
        Xa_7_20[0, :, :, :] = X1_7_20[0, 0, :, :, :] + X1_7_20[0, 1, :, :, :] + X1_7_20[0, 2, :, :, :] + X1_7_20[0, 3, :, :, :] + X1_7_20[0, 4, :, :, :]

        Xp_0d3_0d5[0, :, :, :] = poisson.rvs(Xa_0d3_0d5[0, :, :, :]*10)
        Xp_0d5_1[0, :, :, :] = poisson.rvs(Xa_0d5_1[0, :, :, :]*10)
        Xp_1_2[0, :, :, :] = poisson.rvs(Xa_1_2[0, :, :, :]*10)
        Xp_2_7[0, :, :, :] = poisson.rvs(Xa_2_7[0, :, :, :]*10)
        Xp_7_20[0, :, :, :] = poisson.rvs(Xa_7_20[0, :, :, :]*10)	
        print ('check counts in patch 0.3-0.5: ', np.sum(Xa_0d3_0d5[0, :, :, :]*10))
        print('check counts in patch 0.5-1: ', np.sum(Xa_0d5_1[0, :, :, :]*10))
        print('check counts in patch 1-2: ', np.sum(Xa_1_2[0, :, :, :]*10))
        print('check counts in patch 2-7: ', np.sum(Xa_2_7[0, :, :, :]*10))
        print('check counts in patch 7-20: ', np.sum(Xa_7_20[0, :, :, :]*10))
        

        Xp_0d3_0d5 = Xp_0d3_0d5/(1e-9 + np.max(Xp_0d3_0d5))
        Xp_0d5_1 = Xp_0d5_1/(1e-9 + np.max(Xp_0d5_1))
        Xp_1_2 = Xp_1_2/(1e-9 + np.max(Xp_1_2))
        Xp_2_7 = Xp_2_7/(1e-9 + np.max(Xp_2_7)) 
        Xp_7_20 = Xp_7_20 /(1e-9 + np.max(Xp_7_20)) 
	
        
        unet_pred = unet_model.predict([Xp_7_20, Xp_2_7, Xp_1_2, Xp_0d5_1, Xp_0d3_0d5], verbose=1)
        print('prediction shape: ', unet_pred.shape)

        # layer with the source probs
        grid2D_pred = unet_pred[0, :, :, 0]
        print('grid2D_pred.shape: ', grid2D_pred.shape)
        #print('grid2D_pred: ', grid2D_pred)
        print('grid2D avg: ', np.mean(grid2D_pred))

        

        #lap of gaussian is here

        blobs_log = blob_log(grid2D_pred, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=1, 
                             threshold=label_score_uk, exclude_border=False, overlap=overlap)

        

        print ('number of sources: ', len(blobs_log))


        # pred_nsources, pred_centers = try_kmeans_with_error_method(grid2D_pred, outer_radius, inner_radius, nmin, nmax)
        # from before not used anymore

        


        for blob in blobs_log:
            y, x, r = blob
            list_file_number.append(con_test_file)
            list_lon_c.append(lon_c_file2_7)
            list_lat_c.append(lat_c_file2_7)
            image_file_names.append(filename_ims2_7)
            ycentroid = int(y)
            xcentroid = int(x)
            r_sigma = r
            l_ps,b_ps = get_lb_from_pixel(pixel_id(ycentroid*1, xcentroid*1, int(xsize*1.)), 
                                          [lon_c_file2_7, lat_c_file2_7], xsize)

            pixel_val = pixel_id(ycentroid*1, xcentroid*1, 128)
            print ('pixel_val: ', pixel_val)

            prob_at_centroid = grid2D_pred[ycentroid, xcentroid]
            class_id=0

            list_yc.append(ycentroid*1.0)
            list_xc.append(xcentroid*1.0)

            list_class_id.append(class_id*1.0)
            list_probability.append(prob_at_centroid)

            list_lon_ps.append(l_ps)
            list_lat_ps.append(b_ps)

            list_catalog.append(catalog_file7_20)

            print (con_test_file, ycentroid, xcentroid, 
                   class_id, prob_at_centroid, len(blobs_log))
            

            if len(blobs_log)>=nmax:
                con_50_plus +=1
            print ('check file name: ', image_file_names[-1])

            output_data_frame = pd.DataFrame(data={"image_nr": list_file_number, "filename":image_file_names, 
                                                   "centroid_y": list_yc, "centroid_x": list_xc, 
                                                   "class_id":list_class_id, "probability":list_probability, 
                                                   "lon_c":list_lon_c, "lat_c":list_lat_c, 
                                                   "lon_ps": list_lon_ps, "lat_ps": list_lat_ps, 
                                                   "catalog":list_catalog})    
            
            output_data_frame.to_csv(f"{path_to_results_file}", sep=',', index=False) 
    print (con_50_plus)
    return 0




if __name__ == "__main__":
    main()
