'''
binary classification of true vs fake sources on the same simulated Fermi-LAT
patches # after detect; characterize; same patches used for the flux regression:

  20x20 pixels, 3 energy bins above 1 GeV (7-20 + 20-1000 summed),
  10 time bins kept separate; fake won't vary much over the years

input cube is (W, H, E, T) — Conv3D treats (x, y, energy) as spatial dims and
time as channels. real sources (class 0-3) are label 0, fake (class 4) label 1.
'''


#########
#
#########
import os, time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

import classification_dataloader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)

import tensorflow as tf

from tensorflow.keras.layers import (Input, Conv3D, MaxPool3D,
                                     GlobalAveragePooling3D, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import binary_accuracy

print("Imports done")

start_time = time.time()

# -----------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------
HOMEPATH  = "/d11/CAC/sbhattacharyya/Downloads/"
BASE_PATH = os.path.join(HOMEPATH,
    "Fermi-Flux1000/corrected_S20_LoG_Th0d1_ol0d2")

CSVFILE   = "train_cat90_91_910_rad8_s10_predSNR_check_tbin1.csv"
SAVE_PATH = os.path.join(HOMEPATH, "Fermi-Flux1000/")

PLOT_DIR = os.path.join(SAVE_PATH, "train_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

TAG = "S20_3bins_10tbins_TrueVsFake"

# SNR cut applies to real sources only — fakes have no meaningful predicted SNR.
SNR_MIN          = 0.05 # tested between 0.0 to 0.1 # a metric calculated by Boris from previous work
USE_CLASS_WEIGHT = False   # previous runs were weighted; aug ratio does the balancing
# after discussing with dima and sascha we do not do the balancing by classweights

# -----------------------------------------------------------------------
# load data
# -----------------------------------------------------------------------
print("Reading real sources ...")
ims_real, labs_real, lats_real, lons_real, probs_real, names_real = \
    classification_dataloader.read_sources(
        csvfile=CSVFILE,
        cls_label=[0, 1, 2, 3],
        binary_label=0,
        base_path=BASE_PATH,
        snr_min=SNR_MIN,
        use_poisson=True,
    )

print("Reading fake sources ...")
ims_fake, labs_fake, lats_fake, lons_fake, probs_fake, names_fake = \
    classification_dataloader.read_sources(
        csvfile=CSVFILE,
        cls_label=[4],
        binary_label=1,
        base_path=BASE_PATH,
        snr_min=None,
        use_poisson=True,
    )

print("real:", ims_real.shape, " fake:", ims_fake.shape)

INPUT_SHAPE = ims_real.shape[1:]          # (20, 20, 3, 10)
print("input shape:", INPUT_SHAPE)

# -----------------------------------------------------------------------
# hand-crafted train / test split  (85 / 15), applied per group so the
# test set keeps the same real:fake ratio as the full sample
# -----------------------------------------------------------------------
FRAC = 0.85

def _split(arr):
    n = int(len(arr) * FRAC)
    return arr[:n], arr[n:]

ims_real_tr,  ims_real_te  = _split(ims_real)
labs_real_tr, labs_real_te = _split(labs_real)
lats_real_tr, lats_real_te = _split(lats_real)
lons_real_tr, lons_real_te = _split(lons_real)

ims_fake_tr,  ims_fake_te  = _split(ims_fake)
labs_fake_tr, labs_fake_te = _split(labs_fake)
lats_fake_tr, lats_fake_te = _split(lats_fake)
lons_fake_tr, lons_fake_te = _split(lons_fake)

# -----------------------------------------------------------------------
# augmentation
#   real →  4x  (original + h-flip + v-flip + rot90)
#   fake →  5x  (+ rot270)
# the uneven factors partly compensate the class imbalance, as before.
# -----------------------------------------------------------------------
ims_real_aug = classification_dataloader.aug_images(
    ims_real_tr, rotate_90=True, rotate_lr_90=False)
ims_fake_aug = classification_dataloader.aug_images(
    ims_fake_tr, rotate_90=True, rotate_lr_90=True)

N_AUG_REAL, N_AUG_FAKE = 4, 5

labs_real_aug = np.concatenate([labs_real_tr] * N_AUG_REAL, axis=0)
labs_fake_aug = np.concatenate([labs_fake_tr] * N_AUG_FAKE, axis=0)

# Group IDs follow aug_images' block ordering, so every transformed version
# of one original source receives the same ID. The offset keeps real and fake
# IDs from colliding.
real_groups = np.tile(np.arange(len(ims_real_tr)), N_AUG_REAL)
fake_offset = len(ims_real_tr)
fake_groups = fake_offset + np.tile(np.arange(len(ims_fake_tr)), N_AUG_FAKE)

print("aug shapes  real/fake:", ims_real_aug.shape, ims_fake_aug.shape)

# -----------------------------------------------------------------------
# concatenate all train and test segments
# -----------------------------------------------------------------------
all_ims_tr  = np.concatenate([ims_real_aug,  ims_fake_aug],  axis=0)
all_labs_tr = np.concatenate([labs_real_aug, labs_fake_aug], axis=0)

all_ims_te  = np.concatenate([ims_real_te,  ims_fake_te],  axis=0)
all_labs_te = np.concatenate([labs_real_te, labs_fake_te], axis=0)
all_lats_te = np.concatenate([lats_real_te, lats_fake_te], axis=0)
all_lons_te = np.concatenate([lons_real_te, lons_fake_te], axis=0)

all_groups_tr = np.concatenate([real_groups, fake_groups], axis=0)

print("train pool:", all_ims_tr.shape, " test:", all_ims_te.shape)
print("train class balance  real/fake:",
      int((all_labs_tr == 0).sum()), int((all_labs_tr == 1).sum()))

# -----------------------------------------------------------------------
# train / val. split
#
# split by original-source ID rather than by augmented image. prevent data leak
# — otherwise a rotated copy of a training source lands in validation and the
# val. loss reads far too optimistic.
# -----------------------------------------------------------------------
group_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20)
train_idx, valid_idx = next(
    group_split.split(all_ims_tr, all_labs_tr, groups=all_groups_tr)
)

train_ims, train_labs = shuffle(
    all_ims_tr[train_idx], all_labs_tr[train_idx], random_state=20)
valid_ims, valid_labs = shuffle(
    all_ims_tr[valid_idx], all_labs_tr[valid_idx], random_state=20)

print("train:", train_ims.shape, train_labs.shape)
print("valid:", valid_ims.shape, valid_labs.shape)

# -----------------------------------------------------------------------
# optional class weighting (not used, kept from previous tests)
# -----------------------------------------------------------------------
class_weights = None
flat_labs = train_labs.reshape(-1)
weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(flat_labs), y=flat_labs)
print("balanced class weights would be:", dict(enumerate(weights)))
if USE_CLASS_WEIGHT:
    class_weights = dict(enumerate(weights))

# -----------------------------------------------------------------------
# model  —  3D CNN
#
# pooling is (2, 2, 1): the energy axis is only 3 deep, so pooling it away
# after the first block would leave nothing to convolve over.
# -----------------------------------------------------------------------
def build_model():
    inp = Input(shape=INPUT_SHAPE)
    x = Conv3D(16, kernel_size=(3, 3, 2), activation='relu', padding='same')(inp)
    x = Conv3D(32, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)

    x = Conv3D(32, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = Conv3D(64, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)

    x = Conv3D(64, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name="class3d")


# -----------------------------------------------------------------------
# hyperparams
# -----------------------------------------------------------------------
EPOCHS        = 300
BATCH_SIZE    = 32
LEARNING_RATE = 5e-5

es     = EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta=1e-6)
red_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1,
                           patience=8, min_lr=1e-8)

weight_file = os.path.join(HOMEPATH, f"best_weights_class_{TAG}.weights.h5")
mcp = ModelCheckpoint(filepath=weight_file, save_best_only=True,
                      save_weights_only=True, monitor='val_loss', mode='min')

model = build_model()
print("number of params in cnn model:", model.count_params())
model.summary()

model.compile(optimizer=RMSprop(LEARNING_RATE),
              loss="binary_crossentropy",
              metrics=[binary_accuracy, tf.keras.metrics.AUC(name='auc')])

print("\n### Training ###")
history = model.fit(train_ims, train_labs,
                    validation_data=(valid_ims, valid_labs),
                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                    class_weight=class_weights,
                    callbacks=[es, mcp, red_lr])

print(f"training time: {(time.time() - start_time)/60.:.1f} min")

# -----------------------------------------------------------------------
# loss curves
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'],     color='r', linestyle='-',  label='Train')
axes[0].plot(history.history['val_loss'], color='b', linestyle='-.', label='Val')
axes[0].set_ylabel('Binary cross-entropy')

axes[1].plot(history.history['binary_accuracy'],     color='r', linestyle='-',  label='Train')
axes[1].plot(history.history['val_binary_accuracy'], color='b', linestyle='-.', label='Val')
axes[1].set_ylabel('Binary accuracy')

for ax in axes:
    ax.set_xlabel('Epoch', fontsize=13)
    ax.legend(fontsize=12)


plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f'train_class_{TAG}.png'), dpi=200)

# -----------------------------------------------------------------------
# load best weights and evaluate
# -----------------------------------------------------------------------
model.load_weights(weight_file)

val_metrics = model.evaluate(valid_ims, valid_labs, verbose=0)
print("validation  loss / acc / auc:", val_metrics)

pred_te = model.predict(all_ims_te, batch_size=BATCH_SIZE).reshape(-1)
y_true  = all_labs_te.reshape(-1)

# -----------------------------------------------------------------------
# ROC curve — max TPR/min FPR to get threshold
# -----------------------------------------------------------------------
CLASS_TYPES = ['TRUE', 'FAKE']


def plot_roc_curve(probas, y_true):
    '''roc curve + Youden-J optimal threshold. returns (fig, threshold).'''
    ns_probas = np.zeros_like(y_true, dtype=float)     # no-skill / blind prediction
    ns_auc    = roc_auc_score(y_true, ns_probas)
    model_auc = roc_auc_score(y_true, probas)
    print("check score (Area of ROC): no skill v Model:", ns_auc, model_auc)

    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probas)
    model_fpr, model_tpr, ths = roc_curve(y_true, probas)
    optimal_idx = np.argmax(model_tpr - model_fpr)
    optimal_threshold = ths[optimal_idx]
    print("Threshold value is:", optimal_threshold)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', color='red', alpha=0.7,
             label='Random Pred. [Area: %2.2f]' % ns_auc)
    plt.plot(model_fpr, model_tpr, linestyle='-', color='navy', alpha=0.7,
             label='Model Pred. [Area: %2.2f]' % model_auc)
    plt.scatter(model_fpr[optimal_idx], model_tpr[optimal_idx],
                color='navy', zorder=5, s=40)
    plt.title('ROC-AUC: Optimal Threshold: %2.2f' % optimal_threshold, fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(fontsize=13)
    ax.text(0.79, 0.3, 'Note: 0==TP, 1==FP', transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.tick_params(labelcolor='black', labelsize=14, width=3)
    plt.tight_layout()
    return fig, optimal_threshold


def conf_matrix(probas, y_true, threshold):
    '''confusion matrices at the optimal threshold and at 0.5, plus the report.'''
    y_pred_th  = (probas > threshold).astype(int)
    y_pred_05  = np.round(probas).astype(int)

    fig, axs = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    for ax, y_pred, title in zip(
            axs, [y_pred_th, y_pred_05],
            ['Conf. Matrix w Th: %2.2f' % threshold, 'Conf. Matrix w Th: 0.50']):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"{title}\n{cm_norm}")
        sns.heatmap(cm_norm, annot=True, fmt="0.2f", ax=ax,
                    xticklabels=CLASS_TYPES, yticklabels=CLASS_TYPES,
                    cbar=(title.endswith('0.50')))
        ax.set_xlabel('Pred', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(title, fontsize=12)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred_th, target_names=CLASS_TYPES))

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'confmat_class_{TAG}.png'), dpi=250)
    return fig


rocfig, class_th = plot_roc_curve(pred_te, y_true)
rocfig.savefig(os.path.join(PLOT_DIR, f'roc_class_{TAG}.png'), dpi=250)
print("true optimal threshold:", class_th)

conf_matrix(pred_te, y_true, class_th)

# -----------------------------------------------------------------------
# put the results in a file
# -----------------------------------------------------------------------
df_out = pd.DataFrame({
    'true_lon':    all_lons_te,
    'true_lat':    all_lats_te,
    'class_prob':  pred_te,
    'class':       np.round(pred_te).astype(int),
    'class_th':    (pred_te > class_th).astype(int),
    'true_labels': y_true.astype(int),
})

out_dir = os.path.join(BASE_PATH, f'class_{TAG}')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, f'Class3D_{TAG}_test_Pred.csv')
df_out.to_csv(out_file, sep=',', index=False)
print("predictions saved to:", out_file)

print(f"total time: {(time.time() - start_time)/60.:.1f} min")
