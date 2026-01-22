import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D,\
    BatchNormalization, Concatenate

import time
import keras
from keras import backend as K


import numpy as np
import matplotlib.pyplot as plt

import datareader_LAT_det_Loc













start_time = time.time()

#############################################
# paths to ims
#############################################

path_to_base = '/d6/CAC/sbhattacharyya/Downloads/ps_data_Roberto/'

path_to_0d3_0d5 = path_to_base + 'test_im_iem_psr_bll_fsrq_pwn0d3_0d5_patch768/' # bin0
path_to_0d5_1 = path_to_base + 'test_im_iem_psr_bll_fsrq_pwn0d5_1_patch768/' # bin1
path_to_1_2 = path_to_base + 'test_im_iem_psr_bll_fsrq_pwn1_2_patch768/' # bin2
path_to_2_7 = path_to_base + 'test_im_iem_psr_bll_fsrq_pwn2_7_patch768/' # bin3
path_to_7_20 = path_to_base + 'test_im_iem_psr_bll_fsrq_pwn7_20_patch768_rad8/' # bin4


## mask

path_to_b3_mask = path_to_base + 'test_mk_iem_psr_bll_fsrq_pwn2_7_patch768_rad8/'
#############################
# check reading files
#############################

im_b0_map = datareader_LAT_det_Loc.list_numbers_and_files(path_to_0d3_0d5, prefix='test_image_')
im_b1_map = datareader_LAT_det_Loc.list_numbers_and_files(path_to_0d5_1, prefix='test_image_')
im_b2_map = datareader_LAT_det_Loc.list_numbers_and_files(path_to_1_2, prefix='test_image_')
im_b3_map = datareader_LAT_det_Loc.list_numbers_and_files(path_to_2_7, prefix='test_image_')
im_b4_map = datareader_LAT_det_Loc.list_numbers_and_files(path_to_7_20, prefix='test_image_')

mk_b3_map = datareader_LAT_det_Loc.list_numbers_and_files(path_to_b3_mask, prefix='test_masks_')

common_nums = sorted(set(im_b0_map)& set(im_b1_map) & set(im_b2_map) 
                     & set(im_b3_map) & set(im_b4_map) & set(mk_b3_map))

train_nums = datareader_LAT_det_Loc.make_split_numbers(common_nums, train_size=7000, 
                                                   seed=42, train=True)




(im_b0_list, valid_b0_nums, 
 im_b0_files) = datareader_LAT_det_Loc.load_images_by_numbers(path_to_0d3_0d5, im_b0_map, train_nums, 
                                                          norm=True, add_poisson=True)

(im_b1_list, valid_b1_nums, 
 im_b1_files) = datareader_LAT_det_Loc.load_images_by_numbers(path_to_0d5_1, im_b1_map, train_nums, 
                                                          norm=True, add_poisson=True)


(im_b2_list, valid_b2_nums, 
 im_b2_files) = datareader_LAT_det_Loc.load_images_by_numbers(path_to_1_2, im_b2_map, train_nums, 
                                                          norm=True, add_poisson=True)


(im_b3_list, valid_b3_nums, 
 im_b3_files) = datareader_LAT_det_Loc.load_images_by_numbers(path_to_2_7, im_b3_map, train_nums, 
                                                          norm=True, add_poisson=True)



(im_b4_list, valid_b4_nums, 
 im_b4_files) = datareader_LAT_det_Loc.load_images_by_numbers(path_to_7_20, im_b4_map, train_nums, 
                                                          norm=True, add_poisson=True)






#### same for masks

(mk_b3_list, mk_valid_b3_nums, 
 mk_b3_files) = datareader_LAT_det_Loc.load_masks_by_numbers(path_to_b3_mask, mk_b3_map, train_nums)



print ("checking for consistency")

print ('check shapes across: ', im_b4_list[0].shape, im_b3_list[0].shape, 
       im_b2_list[0].shape, im_b1_list[0].shape, im_b0_list[0].shape, '\n', mk_b3_list[0].shape)



print ('check names across: ', im_b4_files[50], im_b3_files[50], 
       im_b2_files[50], im_b1_files[50], im_b0_files[50], mk_b3_files[50])


print ('check names across: ', im_b4_files[150], im_b3_files[150], 
       im_b2_files[150], im_b1_files[150], im_b0_files[150], mk_b3_files[150])







######################################### 
# process the data as tensorflow datasets
##########################################


### 

def three_to_4d(inp_list):
  x = np.asarray(inp_list) # (N, H, W, C)
  if x.ndim==3: # should be 4 already
    x = x[..., None]
  return x.astype(np.float32)


X_b4 = three_to_4d(im_b4_list)
X_b3 = three_to_4d(im_b3_list)
X_b2 = three_to_4d(im_b2_list)
X_b1 = three_to_4d(im_b1_list)
X_b0 = three_to_4d(im_b0_list) 

print ('check shapes after: ', X_b4.shape, X_b3.shape)


Y = np.asarray(mk_b3_list)
if Y.ndim==3:
  Y = Y[..., None]
Y = Y.astype(np.float32)



##################
# create a val split and add test
###################
N = X_b4.shape[0] # num samples

val_frac  = 0.15
test_frac = 0.10   # small test split

rng = np.random.default_rng(42)
idx = rng.permutation(N)

n_test = int(round(test_frac * N))
n_val  = int(round(val_frac  * N))

test_idx = idx[:n_test]
val_idx  = idx[n_test:n_test + n_val]
tr_idx   = idx[n_test + n_val:]


X_b4_train, X_b4_val, X_b4_test = X_b4[tr_idx], X_b4[val_idx], X_b4[test_idx]
X_b3_train, X_b3_val, X_b3_test = X_b3[tr_idx], X_b3[val_idx], X_b3[test_idx]
X_b2_train, X_b2_val, X_b2_test = X_b2[tr_idx], X_b2[val_idx], X_b2[test_idx]
X_b1_train, X_b1_val, X_b1_test = X_b1[tr_idx], X_b1[val_idx], X_b1[test_idx]
X_b0_train, X_b0_val, X_b0_test = X_b0[tr_idx], X_b0[val_idx], X_b1[test_idx]

Y_b3_train, Y_b3_val, Y_b3_test = Y[tr_idx], Y[val_idx], Y[test_idx] 

autotune = tf.data.AUTOTUNE

def make_multi_input_ds(X_b4, X_b3, X_b2, X_b1, X_b0, Y,
                        batch_size=32, shuffle=True, augment=True, seed=42):

    ds = tf.data.Dataset.from_tensor_slices((
        {
            "inp7_20": X_b4, "inp2_7": X_b3,
            "inp1_2": X_b2, "inp0d5_1": X_b1, "inp0d3_0d5": X_b0,}, 
            Y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(Y), seed=seed, reshuffle_each_iteration=True)

    if augment:
        # augmentation should be reproducible too!!
        rng = tf.random.Generator.from_seed(seed)

        def _maybe_flip(t, flip_lr, flip_ud):
            # flip_lr / flip_ud work for tensors
            t = tf.cond(flip_lr, lambda: tf.image.flip_left_right(t), lambda: t)
            t = tf.cond(flip_ud, lambda: tf.image.flip_up_down(t),  lambda: t)
            return t

        def aug_fn(x, y):
            # One random decision per sample
            r = rng.uniform(shape=[2], minval=0, maxval=1, dtype=tf.float32)
            flip_lr = r[0] < 0.5
            flip_ud = r[1] < 0.5

            x = {k: _maybe_flip(v, flip_lr, flip_ud) for k, v in x.items()}
            y = _maybe_flip(y, flip_lr, flip_ud)
            return x, y

        ds = ds.map(aug_fn, num_parallel_calls=autotune)

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(autotune)
    return ds




############# use this to create to datasets

train_ds = make_multi_input_ds(X_b4_train, X_b3_train, X_b2_train, X_b1_train, X_b0_train, 
                               Y_b3_train, batch_size=32, shuffle=True, augment=True, seed=42)


val_ds = make_multi_input_ds(X_b4_val, X_b3_val, X_b2_val, X_b1_val, X_b0_val, 
                             Y_b3_val, batch_size=32, shuffle=False, augment=False, seed=42)



test_ds = make_multi_input_ds(X_b4_test, X_b3_test, X_b2_test, X_b1_test, X_b0_test, 
                              Y_b3_test, batch_size=32, shuffle=False, augment=False)

mean_time = time.time()

print('time taken to read all data: ', (mean_time - start_time)/60.)


##############################
# plot random ims-mask pair and check !!
###############################
# to plot we need to unbatch the dataset and plot! shit!!
# 
 
def _to_2d(img):
    """(H,W,C)->(H,W), (H,W)->(H,W)."""
    img = tf.convert_to_tensor(img)
    if img.shape.rank == 3 and img.shape[-1] == 1:
        return img[..., 0]
    return img

def _mask_to_show(y):
    """mask shape is (H,W,2), plot only 0th channel."""
    y = tf.convert_to_tensor(y)
    y = y[..., 0]
    return y


def plot_multi_input_samples(ds, n=3, shuffle_buf=512):
    # ds yields (x_dict, y); we want individual samples, not batches
    ds_single = ds.unbatch().shuffle(shuffle_buf, reshuffle_each_iteration=True).take(n)

    keys = ["inp7_20", "inp2_7", "inp1_2", "inp0d5_1", "inp0d3_0d5"]
    titles = ["7-20", "2-7", "1-2", "0.5-1", "0.3-0.5", "mask"]

    fig, axs = plt.subplots(n, 6, figsize=(18, 4*n))
    if n == 1:
        axs = axs[None, :]  # make it 2D

    for row, (x, y) in enumerate(ds_single):
        imgs = [x[k] for k in keys]
        imgs = [_to_2d(im).numpy() for im in imgs]
        msk  = _mask_to_show(y).numpy()

        for col in range(5):
            axs[row, col].imshow(imgs[col])
            axs[row, col].set_title(titles[col])
            axs[row, col].axis("off")

        axs[row, 5].imshow(msk)
        axs[row, 5].set_title(titles[5])
        axs[row, 5].axis("off")

    plt.tight_layout()
    plt.savefig(path_to_base + './random_im_mk_pairs.png', dpi=200)
    #plt.show()

# Call it:
plot_multi_input_samples(train_ds, n=3)

######################################################
# Build U-Net
######################################################

input7_20 = Input(shape=(256, 256, 1), name="inp7_20")
input2_7 = Input(shape=(128, 128, 1), name="inp2_7")
input1_2 = Input(shape=(64, 64, 1), name="inp1_2")
input0d5_1 = Input(shape=(32, 32, 1), name="inp0d5_1")
input0d3_0d5 = Input(shape=(32, 32, 1), name="inp0d3_0d5")


def multi_in_unet_mod2_7(input7_20, input2_7, input1_2, input0d5_1, input0d3_0d5):
  A0 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input7_20)
  A0_1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(A0)
  A0_2 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(A0_1)
  A0_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(A0_2)
  A0_M = Model(inputs=input7_20, outputs=A0_3) # 128


  A = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input2_7)
  A_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(A)  # m1, layer 2
  A_2 = Model(inputs=input2_7, outputs=A_1) # 128
  
  B = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input1_2)
  B = UpSampling2D(size=(2, 2), )(B) # 128
  B_1 = Model(inputs=input1_2, outputs=B)
  
  
  
  C = UpSampling2D(size=(2, 2), )(input0d5_1)
  C_1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(C)
  C_1 = UpSampling2D(size=(2, 2), )(C_1) # 128
  C_2 = Model(inputs=input0d5_1, outputs=C_1)
  
  D = UpSampling2D(size=(2, 2), )(input0d3_0d5)
  D_1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(D)
  D_1 = UpSampling2D(size=(2, 2), )(D_1)# 128
  D_2 = Model(inputs=input0d3_0d5, outputs=D_1)
  
  # concatenate
  ab_conc = Concatenate(axis=-1)([A0_M.output, A_2.output, B_1.output, C_2.output, D_2.output])
  # after concatenate let's add some more operation
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(ab_conc)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 64
  
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool2)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 32
  
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 16
  
  # upsampling path
  up_trconv1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(0.001))(pool4) # 32
  up_trconv1 = Concatenate()([up_trconv1, conv4])
  up_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_trconv1)
  up_conv1 = Conv2D(128, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_conv1)
  
  up_trconv2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(up_conv1)
  up_trconv2 = Concatenate()([up_trconv2, conv3]) # 64
  up_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_trconv2)
  # up_conv2 = Concatenate()([up_conv2, B]) # 
  up_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_conv2)
  
  up_trconv3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(up_conv2) # 128
  up_trconv3 = Concatenate()([up_trconv3, conv2])
  up_conv3 = Conv2D(32, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_trconv3)
  up_conv3 = Conv2D(32, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_conv3)

  # up_trconv4 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(up_conv3)
  # up_trconv4 = Concatenate()([up_trconv4, A0_1])
  # up_conv4 = Conv2D(16, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_trconv4)
  # up_conv4 = Conv2D(16, (3, 3), activation='relu', padding='same',  kernel_regularizer=l2(0.001))(up_conv4)
  
  output = Conv2D(2, (1, 1), activation='softmax')(up_conv3)
  
  model = Model(inputs=[A0_M.input, A_2.input, B_1.input, C_2.input, D_2.input], outputs=[output])
  
  return model


###################################
#
# Call Model
#
###################################


multi_in_unet_mod2_7 = multi_in_unet_mod2_7(input7_20, input2_7, input1_2, input0d5_1, input0d3_0d5)
print (multi_in_unet_mod2_7.summary())
# trainable params: ~ 2,436,386

################################
# Add dice_coeff score as metric 
################################

def dice_coeff(y_true, y_pred, smooth=1e-4):
    intersection = K.sum(y_true * y_pred, axis=(0, 1))
    union = K.sum(y_true, axis=(0, 1)) + K.sum(y_pred, axis=(0, 1))
    dice = (2. * intersection + smooth)/(union + smooth)
    return dice


#################################
# try weighted binary_cross_entropy
#################################

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


#################################
# Compile Model
#################################

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, 
                                                 patience=5, min_lr=1e-6, verbose=1)

# check loss values and if doesn't decrease include early stopping, else for now we use 40/60 epochs and don't stop the training 

save_checkpoint3 = ModelCheckpoint(path_to_base + 'checkpoints/unet_mask_try_iem_bll_psr_fsrq_pwn_poisson_rvs_5inputs_p768_fcji0_40_1_60_250epochs_normed_rad8_check_mk2_7GeV.h5', 
                                   verbose=1, save_best_only=True, monitor='val_loss')

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-7)

#multi_in_unet_mod4.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['acc', dice_coeff])

multi_in_unet_mod2_7.compile(loss=weighted_binary_cross_entropy({0:0.40, 1:0.60}, from_logits = False), 
                             optimizer=Adam(learning_rate=1e-3), metrics=['acc', dice_coeff])

# ##########################
# # Fit the Model
# ##########################

histr4 = multi_in_unet_mod2_7.fit(train_ds, 
                                  validation_data= val_ds, 
                                  batch_size=32, epochs=250, 
                                  callbacks=[save_checkpoint3, reduce_lr, es])


train_time = time.time()
print('time (in minutes) to train the network: ', (train_time-mean_time)/60.)

###################################
# Plot the Training Curves
###################################


fig = plt.figure(figsize=(12, 5))

train_loss_p = histr4.history['loss']
val_loss_p = histr4.history['val_loss']

train_dice = histr4.history['dice_coeff']
val_dice = histr4.history['val_dice_coeff']


# train_acc_p = histr4.history['acc']
# val_acc_p = histr4.history['val_acc']

fig.add_subplot(121)

plt.plot(range(len(train_loss_p)), train_loss_p, linestyle='-', color='r', label='Train Loss')
plt.plot(range(len(val_loss_p)), val_loss_p, linestyle='-.', color='b', label='Val Loss')

plt.xlabel('Num Epochs', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)

fig.add_subplot(122)

plt.plot(range(len(train_dice)), train_dice, linestyle='-', color='r', label='Train dice')
plt.plot(range(len(val_dice)), val_dice, linestyle='-.', color='b', label='Val dice')

plt.xlabel('Num Epochs', fontsize=13)
plt.ylabel('Dice', fontsize=13)
plt.legend(fontsize=13)


plt.tight_layout()
plt.savefig(path_to_base + 'train_plots/multi-unet_iem_bll_psr_fsrq_pwn_rvs5inp_loss_Dice_epochs250_weight0_40_1_60_normed_rad8_Front_mk2_7.png', 
            dpi=200)

# #########################################
# # do a generic prediction on the test set 
# # just for check
# #########################################


#################
# load the best model
#################

model_path = path_to_base + 'checkpoints/unet_mask_try_iem_bll_psr_fsrq_pwn_poisson_rvs_5inputs_p768_fcji0_40_1_60_250epochs_normed_rad8_check_mk2_7GeV.h5'
multi_input_unet_model = load_model(model_path, 
                                    custom_objects={'weighted_cross_entropy_fn': weighted_binary_cross_entropy({0:0.40, 1:0.60}, from_logits = False), 
                                                    'dice_coeff': dice_coeff})


total_params = multi_input_unet_model.count_params()


print ('saved model loaded!', multi_input_unet_model)
print(f"!!!! Total parameters !!!! : {total_params:,}")

def _mask_to_show_pred(y):
    """
    Accepts (H,W,2) or (B,H,W,2) and returns (H,W)
    """
    y = tf.convert_to_tensor(y)

    # remove batch if present
    if y.shape.rank == 4:
        y = y[0]

    # select channel 0
    y = y[..., 0]

    return y
### prepare for predictions on test ds and plot

def plot_test_predictions(model, test_ds, n=3, shuffle_buf=512, 
                          save_path=None):
    keys   = ["inp7_20", "inp2_7", "inp1_2", "inp0d5_1", "inp0d3_0d5"]
    titles = ["7-20", "2-7", "1-2", "0.5-1", "0.3-0.5", "pred", "true"]

    ds_single = test_ds.unbatch().shuffle(shuffle_buf, reshuffle_each_iteration=True).take(n)

    fig, axs = plt.subplots(n, 7, figsize=(21, 4*n))
    if n == 1:
        axs = axs[None, :]

    for row, (x, y_true) in enumerate(ds_single):
        # Add batch dim for inference
        x_batched = {k: tf.expand_dims(x[k], axis=0) for k in keys}

        y_pred = model(x_batched, training=False)

        imgs = [_to_2d(x[k]).numpy() for k in keys]
        pred = _mask_to_show_pred(y_pred, ).numpy()
        true = _mask_to_show(y_true, ).numpy()

        # 5 inps..
        for col in range(5):
            axs[row, col].imshow(imgs[col])
            axs[row, col].set_title(titles[col])
            axs[row, col].axis("off")

        # pred + true
        axs[row, 5].imshow(pred)
        axs[row, 5].set_title("pred")
        axs[row, 5].axis("off")

        axs[row, 6].imshow(true)
        axs[row, 6].set_title("true")
        axs[row, 6].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


# after loading best weights:
plot_test_predictions(model=multi_input_unet_model, 
                      test_ds=test_ds, n=3, 
                      save_path=path_to_base + "./test_random_preds_masks2-7GeV.png")