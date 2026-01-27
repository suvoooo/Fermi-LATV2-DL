import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

import pandas as pd


###############################
# data-reader
# works great when all folders have the same files (numbers)
# test_image_123.npy, test_masks_123.npy
###############################

# make sure the file numbers are the same

def read_npy_data(base_dir:str, train:bool, norm:bool, add_poisson:bool, 
                  mask_f:bool, train_size: int = 8000, seed: int = 42):
    

    all_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".npy")])
    all_numbers = [int(f.split('_')[2].split('.')[0]) for f in all_files]

    # Shuffle numbers with fixed seed
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(all_numbers))

    if train: 
        # select randomly 13000 images
        selected_indices = shuffled_indices[:train_size]
    else: 
        # select the remaining 2000 images    
        selected_indices = shuffled_indices[train_size:]


    selected_numbers = set([all_numbers[i] for i in selected_indices])

    if not mask_f:

        im_dict, im_files = {}, {}
        




        
        
        for filename in all_files:
            
            # Extract the number from the filename
            number = int(filename.split('_')[2].split('.')[0])
            if number not in selected_numbers:
                continue


            file_path = os.path.join(base_dir, filename)
            im_data = np.load(file_path)
            
            im_tot = (im_data[0, :, :, :] + im_data[1, :, :, :] + im_data[2, :, :, :] 
                    + im_data[3, :, :, :] + im_data[4, :, :, :]*1.7)
            # iem , bll, fsrq, pwn, psr
            # im_tot = np.sum(im_data[1:], axis=0) # for later 

            if add_poisson:
                im_tot = poisson.rvs(im_tot*10)

            if norm:
                im_tot = im_tot/(np.max(im_tot + 1e-9))    


            

            im_dict[number] = im_tot
            im_files[number] = filename

                    
        # Ensure consistent ordering
        valid_nums = sorted(set(im_dict))
        print(f"[INFO] Loaded {len(valid_nums)} valid samples from {base_dir}")

        im_list = [im_dict[num] for num in valid_nums]
        im_files_list = [im_files[num] for num in valid_nums]

    else:
        mk_dict, mk_files = {}, {}    

        for f_name in all_files:
            number = int(f_name.split('_')[2].split('.')[0])
            if number not in selected_numbers:
                continue


            file_path = os.path.join(base_dir, f_name)
            mk_data = np.load(file_path)

            mk_dict[number] = mk_data
            mk_files[number] = f_name
        valid_nums = sorted(set(mk_dict))
        print(f"[INFO] Loaded {len(valid_nums)} valid samples from {base_dir}")

        im_list = [mk_dict[num] for num in valid_nums]
        im_files_list = [mk_files[num] for num in valid_nums]



    # total_list = [x for _, x in sorted(zip(nums_list, total_list))]
    # total_files = [x for _, x in sorted(zip(nums_list, total_files))]

    

    # nums_list.sort()

    return (im_list, valid_nums, im_files_list)


###########################
# better way to write
# when different folders have diff numbers of files
#
############################

def list_numbers_and_files(base_dir, prefix=None):
    all_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".npy")])
    num_to_file = {}
    for f in all_files:
        if prefix is not None and not f.startswith(prefix):
            continue
        num = int(f.split('_')[2].split('.')[0])
        num_to_file[num] = f
    return num_to_file

def make_split_numbers(nums, train_size=8000, seed=42, train=True):
    nums = np.array(sorted(nums))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(nums))
    if train:
        sel = perm[:train_size]
    else:
        sel = perm[train_size:]
    return nums[sel].tolist()

def load_images_by_numbers(base_dir, num_to_file, numbers, norm=True, add_poisson=True):
    im_list, files_list = [], []
    for num in numbers:
        f = num_to_file[num]
        im_data = np.load(os.path.join(base_dir, f))

        im_tot = (im_data[0] + im_data[1] + im_data[2] + im_data[3] + im_data[4])

        if add_poisson:
            im_tot = poisson.rvs(im_tot * 10)

        if norm:
            im_tot = im_tot / (np.max(im_tot) + 1e-9)

        im_list.append(im_tot)
        files_list.append(f)

    return im_list, numbers, files_list

def load_masks_by_numbers(base_dir, num_to_file, numbers):
    mk_list, files_list = [], []
    for num in numbers:
        f = num_to_file[num]
        mk = np.load(os.path.join(base_dir, f))
        mk_list.append(mk)
        files_list.append(f)
    return mk_list, numbers, files_list
