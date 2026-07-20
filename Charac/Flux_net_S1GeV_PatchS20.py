'''
updated from previous runs with 
20x20 pixels images (before 10x10)

and counts map with energy bins above 1 GeV (7-20 + 20-1000 summed)
'''


#########
#
#########
import os, time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.stats import pearsonr

import Flux_dataloader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

tfd  = tfp.distributions
tfpl = tfp.layers

print("Imports done")

start_time = time.time()

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
HOMEPATH  = "/d11/CAC/sbhattacharyya/Downloads/"
BASE_PATH = os.path.join(HOMEPATH,
    "Fermi-Flux1000/corrected_S20_LoG_Th0d1_ol0d2")

CSVFILE   = "train_cat90_91_910_rad8_s10_predSNR_check_tbin1.csv"
SAVE_PATH = os.path.join(HOMEPATH, "Fermi-Flux1000/")

SCALE_HIGH_FLUX = True
SCALE_FACTOR    = 1.5

os.makedirs(os.path.join(HOMEPATH, "train_plots"), exist_ok=True)

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
print("Reading data ...")
images, labels, lats, lons, fluxes, names = Flux_dataloader.read_train_sources(
    csvfile=CSVFILE,
    cls_label=[0, 1, 2, 3],
    base_path=BASE_PATH,
    use_poisson=True,
)

print("images shape:", images.shape)
print("fluxes shape:", fluxes.shape)

# -----------------------------------------------------------------------
# Flux standardisation:  log10 -> StandardScaler
# -----------------------------------------------------------------------
flux_log = np.log10(fluxes)
std_obj  = StandardScaler()
flux_std = std_obj.fit_transform(flux_log)

print("flux_std min/max:", flux_std.min(), flux_std.max())

# -----------------------------------------------------------------------
# Split into mid / high / low flux ranges
# -----------------------------------------------------------------------
HIGH_TH, LOW_TH = 2.034, -2.034

sq        = flux_std.squeeze()
mid_mask  = (sq >= LOW_TH) & (sq <= HIGH_TH)
high_mask = sq > HIGH_TH
low_mask  = sq < LOW_TH

print(f"mid: {mid_mask.sum()}  high: {high_mask.sum()}  low: {low_mask.sum()}")

ims_mid,  flux_mid,  labs_mid,  lats_mid,  lons_mid  = (
    images[mid_mask],  flux_std[mid_mask],  labels[mid_mask],
    lats[mid_mask],    lons[mid_mask])

ims_high, flux_high, labs_high, lats_high, lons_high = (
    images[high_mask], flux_std[high_mask], labels[high_mask],
    lats[high_mask],   lons[high_mask])

ims_low,  flux_low,  labs_low,  lats_low,  lons_low  = (
    images[low_mask],  flux_std[low_mask],  labels[low_mask],
    lats[low_mask],    lons[low_mask])

# -----------------------------------------------------------------------
# Hand-crafted train / test split  (85 / 15)
# -----------------------------------------------------------------------
FRAC = 0.85

def _split(arr):
    n = int(len(arr) * FRAC)
    return arr[:n], arr[n:]

ims_mid_tr,  ims_mid_te   = _split(ims_mid)
flux_mid_tr, flux_mid_te  = _split(flux_mid)
labs_mid_tr, labs_mid_te  = _split(labs_mid)
lats_mid_tr, lats_mid_te  = _split(lats_mid)
lons_mid_tr, lons_mid_te  = _split(lons_mid)

ims_high_tr,  ims_high_te  = _split(ims_high)
flux_high_tr, flux_high_te = _split(flux_high)
labs_high_tr, labs_high_te = _split(labs_high)
lats_high_tr, lats_high_te = _split(lats_high)
lons_high_tr, lons_high_te = _split(lons_high)

ims_low_tr,  ims_low_te  = _split(ims_low)
flux_low_tr, flux_low_te = _split(flux_low)
labs_low_tr, labs_low_te = _split(labs_low)
lats_low_tr, lats_low_te = _split(lats_low)
lons_low_tr, lons_low_te = _split(lons_low)

# -----------------------------------------------------------------------
# Optional scaling of high-flux training images and labels
# -----------------------------------------------------------------------
if SCALE_HIGH_FLUX:
    ims_high_tr  = ims_high_tr  * SCALE_FACTOR
    flux_high_tr = flux_high_tr * SCALE_FACTOR
    print(f"high-flux train images and labels scaled by {SCALE_FACTOR}")

scale_tag = f'MultHFlux{str(SCALE_FACTOR).replace(".", "d")}' if SCALE_HIGH_FLUX else 'NoScale'

# -----------------------------------------------------------------------
# Augmentation
#   mid  →  3x  (original + h-flip + v-flip)
#   high →  5x  (+ rot90 + rot270)
#   low  →  5x
# -----------------------------------------------------------------------
ims_mid_aug  = Flux_dataloader.aug_images(ims_mid_tr,  rotate_90=False, rotate_lr_90=False)
ims_high_aug = Flux_dataloader.aug_images(ims_high_tr, rotate_90=True,  rotate_lr_90=True)
ims_low_aug  = Flux_dataloader.aug_images(ims_low_tr,  rotate_90=True,  rotate_lr_90=True)

flux_mid_aug  = np.concatenate([flux_mid_tr]  * 3, axis=0)
flux_high_aug = np.concatenate([flux_high_tr] * 5, axis=0)
flux_low_aug  = np.concatenate([flux_low_tr]  * 5, axis=0)

labs_mid_aug  = np.concatenate([labs_mid_tr]  * 3, axis=0)
labs_high_aug = np.concatenate([labs_high_tr] * 5, axis=0)
labs_low_aug  = np.concatenate([labs_low_tr]  * 5, axis=0)

lats_mid_aug  = np.concatenate([lats_mid_tr]  * 3, axis=0)
lats_high_aug = np.concatenate([lats_high_tr] * 5, axis=0)
lats_low_aug  = np.concatenate([lats_low_tr]  * 5, axis=0)

lons_mid_aug  = np.concatenate([lons_mid_tr]  * 3, axis=0)
lons_high_aug = np.concatenate([lons_high_tr] * 5, axis=0)
lons_low_aug  = np.concatenate([lons_low_tr]  * 5, axis=0)

# Group IDs follow aug_images' block ordering, so every transformed version
# of one original source receives the same ID. Offsets keep IDs unique
# between the mid-, high-, and low-flux subsets.
mid_groups = np.tile(np.arange(len(ims_mid_tr)), 3)
high_offset = len(ims_mid_tr)
high_groups = high_offset + np.tile(np.arange(len(ims_high_tr)), 5)
low_offset = high_offset + len(ims_high_tr)
low_groups = low_offset + np.tile(np.arange(len(ims_low_tr)), 5)

print("aug shapes  mid/high/low:", ims_mid_aug.shape, ims_high_aug.shape, ims_low_aug.shape)

# -----------------------------------------------------------------------
# Concatenate all train and test segments
# -----------------------------------------------------------------------
all_ims_tr  = np.concatenate([ims_mid_aug,  ims_high_aug,  ims_low_aug],  axis=0)
all_flux_tr = np.concatenate([flux_mid_aug, flux_high_aug, flux_low_aug], axis=0)

all_ims_te  = np.concatenate([ims_mid_te,  ims_high_te,  ims_low_te],  axis=0)
all_flux_te = np.concatenate([flux_mid_te, flux_high_te, flux_low_te], axis=0)
all_labs_te = np.concatenate([labs_mid_te, labs_high_te, labs_low_te], axis=0)
all_lats_te = np.concatenate([lats_mid_te, lats_high_te, lats_low_te], axis=0)
all_lons_te = np.concatenate([lons_mid_te, lons_high_te, lons_low_te], axis=0)

all_groups_tr = np.concatenate([mid_groups, high_groups, low_groups], axis=0)

# -----------------------------------------------------------------------
# train / val. split
#
# split by original-source ID rather than by augmented image. prevent data leak
# -----------------------------------------------------------------------
group_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20)
train_idx, valid_idx = next(
    group_split.split(all_ims_tr, all_flux_tr, groups=all_groups_tr)
)

train_ims, train_flux = shuffle(
    all_ims_tr[train_idx], all_flux_tr[train_idx], random_state=20)
valid_ims, valid_flux = shuffle(
    all_ims_tr[valid_idx], all_flux_tr[valid_idx], random_state=20)

print("train:", train_ims.shape, train_flux.shape)
print("valid:", valid_ims.shape, valid_flux.shape)

# -----------------------------------------------------------------------
# model  —  plain Dense(2) head; no TFP layer in the graph
#
# model outputs raw parameters [loc, raw_scale] as a plain tensor.
# tfd.Normal is constructed explicitly in the loss and during inference,
# avoiding the KerasTensor-inside-lambda error that DistributionLambda
# and IndependentNormal both cause with the Functional API.
# -----------------------------------------------------------------------
def build_model():
    inp = Input(shape=(20, 20, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(2, activation='linear')(x)       # [..., 0] = loc, [..., 1] = raw_scale
    return Model(inputs=inp, outputs=x)


def _make_dist(params):
    """Build tfd.Normal from raw model output (works on real tensors or numpy)."""
    loc   = params[..., :1]
    scale = 1e-4 + tf.math.softplus(params[..., 1:])
    return tfd.Normal(loc=loc, scale=scale)


def nll(y_true, params):
    """Gaussian NLL loss — constructs the distribution from raw parameters."""
    return -_make_dist(params).log_prob(y_true)


# -----------------------------------------------------------------------
# hyperparams
# -----------------------------------------------------------------------
EPOCHS        = 300
BATCH_SIZE    = 64
LEARNING_RATE = 5e-5
NUM_MODELS    = 3

es     = EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta=1e-7)
red_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1,
                           patience=8, min_lr=1e-8)

# -----------------------------------------------------------------------
# ensemble training
# -----------------------------------------------------------------------
def train_ensemble(x_train, y_train, x_val, y_val, num, batch_size, epochs):
    histories, models = [], []
    for i in range(num):
        model = build_model()
        model.compile(optimizer=Adam(LEARNING_RATE), loss=nll,
                      metrics=['mean_absolute_error'])
        weight_file = os.path.join(
            HOMEPATH,
            f"best_weights_flux1000_3bins_S20_TFProba_Ensemble{i}_{scale_tag}.weights.h5")
        mcp = ModelCheckpoint(filepath=weight_file, save_best_only=True,
                              save_weights_only=True, monitor='val_loss', mode='min')
        h = model.fit(x_train, y_train,
                      batch_size=batch_size, epochs=epochs,
                      validation_data=(x_val, y_val),
                      callbacks=[es, red_lr, mcp])
        histories.append(h)
        models.append(model)
        print(f"model {i} done")
    return histories, models


print("\n### Training ensemble ###")
histories, models = train_ensemble(
    train_ims, train_flux, valid_ims, valid_flux,
    num=NUM_MODELS, batch_size=BATCH_SIZE, epochs=EPOCHS)

# -----------------------------------------------------------------------
# Training curves
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['magenta', 'lime', 'orange']

for k, col in enumerate(colors):
    axes[0].plot(histories[k].history['loss'],                     color=col, linestyle='-')
    axes[0].plot(histories[k].history['val_loss'],                 color=col, linestyle='-.')
    axes[1].plot(histories[k].history['mean_absolute_error'],      color=col, linestyle='-')
    axes[1].plot(histories[k].history['val_mean_absolute_error'],  color=col, linestyle='-.')

for ax, title in zip(axes, ['NLL Loss', 'MAE']):
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_yscale('log')

color_handles = [Patch(color=c, label=f'M{k+1}') for k, c in enumerate(colors)]
style_handles = [plt.Line2D([], [], linestyle=s, color='k', label=l)
                 for s, l in [('-', 'Train'), ('-.', 'Val')]]
axes[0].legend(handles=color_handles + style_handles, fontsize=8)

plt.suptitle('Ensemble training — 3-bin 20x20 flux regression (TFProba, DistributionLambda)')
plt.tight_layout()
plt.savefig(os.path.join(HOMEPATH, 'Fermi-Flux1000/train_plots',
            f'train_flux_3bins_S20_TFProba_Ensemble_{scale_tag}.png'), dpi=200)

# -----------------------------------------------------------------------
# Load best weights and evaluate on test set
# -----------------------------------------------------------------------
loaded_models = []
for i in range(NUM_MODELS):
    m = build_model()
    m.load_weights(os.path.join(
        HOMEPATH,
        f"best_weights_flux1000_3bins_S20_TFProba_Ensemble{i}_{scale_tag}.weights.h5"))
    loaded_models.append(m)

pred_flux_file = f'Flux1000_3bins_S20_TFProba_Ensemble{NUM_MODELS}_{scale_tag}_test.csv'

# -----------------------------------------------------------------------
# Uncertainty decomposition:
#   aleatoric  = mean of per-model predicted σ   (data noise, irreducible)
#   epistemic  = std  of per-model predicted μ   (model disagreement)
#   total σ    = sqrt(mean(σ_i²) + var(μ_i))    (mixture-of-Gaussians)
# All quantities below are in standardised flux space; inverse-transform
# the mean to log10(flux) for saving.
# -----------------------------------------------------------------------
records = []
for lat_t, lon_t, lab_t, flx_t, im_t in zip(all_lats_te, all_lons_te,
                                              all_labs_te, all_flux_te,
                                              all_ims_te):
    im_in     = im_t.reshape(1, 20, 20, 3)
    true_flux = std_obj.inverse_transform(flx_t.reshape(-1, 1))[0][0]

    means, sigmas = [], []
    for m in loaded_models:
        params = m(im_in, training=False)
        dist   = _make_dist(params)
        means.append(float(dist.mean().numpy().squeeze()))
        sigmas.append(float(dist.stddev().numpy().squeeze()))

    mu_bar    = np.mean(means)                              # ensemble mean (std space)
    sigma_ale = np.mean(sigmas)                             # aleatoric: avg predicted σ
    sigma_epi = np.std(means)                               # epistemic: spread of μ_i
    sigma_tot = np.sqrt(np.mean(np.array(sigmas)**2)
                        + np.var(means))                    # total (mixture decomposition)

    pred_flux_mean = std_obj.inverse_transform([[mu_bar]])[0][0]   # log10(flux)

    records.append({
        'flux1000_pred_log10':   pred_flux_mean,
        'flux1000_true_log10':   true_flux,
        'sigma_aleatoric_std':   sigma_ale,    # in standardised units
        'sigma_epistemic_std':   sigma_epi,
        'sigma_total_std':       sigma_tot,
        'test_lat':              lat_t,
        'test_lon':              lon_t,
        'label':                 int(lab_t.item()),
    })

df_out  = pd.DataFrame(records)
out_dir = os.path.join(SAVE_PATH, f'corrected_S20_LoG_Th0d1_ol0d2/true_S20_3bins_{scale_tag}_TFProba/')
os.makedirs(out_dir, exist_ok=True)
df_out.to_csv(os.path.join(out_dir, pred_flux_file), index=False)
print("predictions saved to:", os.path.join(out_dir, pred_flux_file))



print(f"total time: {(time.time() - start_time)/60.:.1f} min")

