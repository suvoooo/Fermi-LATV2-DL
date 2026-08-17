'''
position refinement + localisation uncertainty on the 20x20 S20 patches,
trained *without* pred_SNR. dropped early on after discussing with Sascha.
BP's calc. of SNR was too simplistic

patch centre is pred_x, pred_y; we are giving the network frac_x, frac_y 
which is how far the detection sits from the center; (basically 0) 
network wants to predict dx, dy; which is the true source offset from center.
in very short, given the patch centers are based on predicted locations, we are training 
the model to predict how far this center is from the true center
'''

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import distance_dataloader as ddl

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPool2D, Flatten, Dense,
                                     Concatenate, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("Imports done")
start_time = time.time()

# -----------------------------------------------------------------------
# all paths
# -----------------------------------------------------------------------
HOMEPATH  = "/d11/CAC/sbhattacharyya/Downloads/"
BASE_PATH = os.path.join(HOMEPATH, "Fermi-Flux1000/corrected_S20_LoG_Th0d1_ol0d2")

CSVFILE  = "train_cat90_91_910_rad8_s10_predSNR_check_tbin1.csv"

# evaluation or stat_csv are the predicted coordinates file from unet+log
STAT_CSV = os.path.join(HOMEPATH, "Fermi-Flux1000/outputs-csv",
    "uneLoG_evaluation_cat90_91_910R_allSources.csv")

SAVE_PATH = os.path.join(HOMEPATH, "Fermi-Flux1000/")
PLOT_DIR  = os.path.join(SAVE_PATH, "train_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# switches
# -----------------------------------------------------------------------
CLS_LABEL     = [0, 1, 2, 3]   # real sources only; a fake has no true position
DIST_MAX      = 0.25           # true-positive definition, degrees
SNR_MIN       = None           # row *selection* on pred_SNR. leave at None: a cut
                               # that cannot be reproduced on 4FGL reintroduces
                               # exactly the dependence this script removes
USE_PRED_SNR  = False          # the point of this variant
USE_FRAC      = True           # (frac_x, frac_y) — see the header
USE_POISSON   = True
CROP          = True           # 19x19 so the reference pixel is the exact centre
USE_SHIFTS    = False          # extra whole-pixel translations, targets moved with them
N_MODELS      = 12

# built from the switches so the outputs can never be filed under a tag that
# does not describe the run that produced them
TAG = ("S20_2d7frame_PosNLL"
       + ("" if USE_PRED_SNR else "_noSNR")
       + ("_frac" if USE_FRAC else ""))
print("TAG:", TAG)

# -----------------------------------------------------------------------
# load
# -----------------------------------------------------------------------
print("Reading sources ...")
images, aux, targets, meta, CENTRE = ddl.read_sources(
    patch_csv=CSVFILE, stat_csv=STAT_CSV, base_path=BASE_PATH,
    cls_label=CLS_LABEL, dist_max=DIST_MAX, snr_min=SNR_MIN,
    use_poisson=USE_POISSON, crop=CROP, use_pred_snr=USE_PRED_SNR,
    use_frac=USE_FRAC)

SIZE = images.shape[1]
INPUT_SHAPE = images.shape[1:]
N_AUX = aux.shape[1]
AUX_NAMES = meta.attrs['aux_names']
print("input shape:", INPUT_SHAPE, " centre index:", CENTRE, " n_aux:", N_AUX)

assert N_AUX == 3 + 2 * USE_FRAC + USE_PRED_SNR, \
    f"aux is {N_AUX} wide, switches say {3 + 2 * USE_FRAC + USE_PRED_SNR}"

# frac_x / frac_y are positions in the patch frame, so the augmentation has to
# transform them alongside the target rather than tiling them unchanged
AUX_VEC_COLS = ((AUX_NAMES.index('frac_x'), AUX_NAMES.index('frac_y'))
                if USE_FRAC else None)

DEG_PER_PIX = ddl.calibrate_deg_per_pixel(meta)
ddl.check_transforms(SIZE, CENTRE)

# -----------------------------------------------------------------------
# train / test split, at source level and before augmentation
# no copy of a training source can reach the test set
# -----------------------------------------------------------------------
FRAC = 0.85
n_tr = int(len(images) * FRAC)

ims_tr,  ims_te  = images[:n_tr],  images[n_tr:]
aux_tr,  aux_te  = aux[:n_tr],     aux[n_tr:]
tgt_tr,  tgt_te  = targets[:n_tr], targets[n_tr:]
meta_tr, meta_te = meta.iloc[:n_tr].reset_index(drop=True), \
                   meta.iloc[n_tr:].reset_index(drop=True)

print("train pool:", ims_tr.shape, " test:", ims_te.shape)

# -----------------------------------------------------------------------
# augmentation — 5x, and every transform moves the target with the image
# -----------------------------------------------------------------------
ims_aug, tgt_aug, aux_aug, groups = ddl.aug_images(
    ims_tr, tgt_tr, aux_tr, rotate_90=True, rotate_lr_90=True,
    aux_vec_cols=AUX_VEC_COLS)

if USE_SHIFTS:
    ims_sh, tgt_sh, aux_sh, grp_sh = ddl.shift_images(
        ims_aug, tgt_aug, aux_aug, groups, CENTRE, aux_vec_cols=AUX_VEC_COLS)
    ims_aug = np.concatenate([ims_aug, ims_sh], axis=0)
    tgt_aug = np.concatenate([tgt_aug, tgt_sh], axis=0)
    aux_aug = np.concatenate([aux_aug, aux_sh], axis=0)
    groups  = np.concatenate([groups,  grp_sh], axis=0)

print("after augmentation:", ims_aug.shape, tgt_aug.shape)

# -----------------------------------------------------------------------
# train / val split by original-source id
#
# split on grps, 
# 
# -----------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20)
train_idx, valid_idx = next(gss.split(ims_aug, tgt_aug, groups=groups))

train_ims, train_tgt, train_aux = shuffle(
    ims_aug[train_idx], tgt_aug[train_idx], aux_aug[train_idx], random_state=20)
valid_ims, valid_tgt, valid_aux = shuffle(
    ims_aug[valid_idx], tgt_aug[valid_idx], aux_aug[valid_idx], random_state=20)

print("train:", train_ims.shape, " valid:", valid_ims.shape)
assert not (set(groups[train_idx]) & set(groups[valid_idx])), "group leak"

# -----------------------------------------------------------------------
# standardise the auxiliary scalars — fit on train only, once
#
# -----------------------------------------------------------------------
scaler = StandardScaler().fit(train_aux)
train_aux_s = scaler.transform(train_aux).astype(np.float32)
valid_aux_s = scaler.transform(valid_aux).astype(np.float32)
test_aux_s  = scaler.transform(aux_te).astype(np.float32)
np.savez(os.path.join(SAVE_PATH, f"aux_scaler_{TAG}.npz"),
         mean=scaler.mean_, scale=scaler.scale_, names=np.array(AUX_NAMES))
print("aux features:", AUX_NAMES)

# -----------------------------------------------------------------------
# loss
#
# -----------------------------------------------------------------------
LOGVAR_MIN, LOGVAR_MAX = -10.0, 4.0


def _squash_log_var(raw):
    return LOGVAR_MIN + (LOGVAR_MAX - LOGVAR_MIN) * tf.sigmoid(raw)


def negative_log_likelihood(y_true, y_pred):
    mu, log_var = y_pred[:, :2], y_pred[:, 2:]
    sq = tf.square(y_true - mu)
    nll = 0.5 * (log_var + sq * tf.exp(-log_var))
    return tf.reduce_mean(tf.reduce_sum(nll, axis=-1))


def mu_rmse(y_true, y_pred):
    '''rms of the mean prediction alone, in pixels — NLL on its own can improve
    by widening sigma, so watch the two together.'''
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred[:, :2])))


# -----------------------------------------------------------------------
# model
# Flatten, not GlobalAveragePooling. global pooling is translation invariant,
# -----------------------------------------------------------------------
def build_model(log_var_bias):
    img_in = Input(shape=INPUT_SHAPE, name='patch')
    x = Conv2D(32, 3, activation='relu', padding='same')(img_in)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPool2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPool2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    aux_in = Input(shape=(N_AUX,), name='aux')
    a = Dense(32, activation='relu')(aux_in)
    a = Dense(16, activation='relu')(a)

    h = Concatenate()([x, a])
    h = Dense(64, activation='relu')(h)
    h = Dense(32, activation='relu')(h)

    mu = Dense(16, activation='relu')(h)
    mu = Dense(2, name='mu')(mu)

    ###### lv
    lv = Dense(16, activation='relu')(h)
    lv = Dense(2, kernel_initializer='zeros',
               bias_initializer=Constant(log_var_bias))(lv)
    lv = Lambda(_squash_log_var, name='log_var')(lv)

    return Model([img_in, aux_in], Concatenate()([mu, lv]), name='pos_nll')


# sigma the data actually shows, mapped through the squash so the head starts there
_lv0 = float(np.log(train_tgt.var(axis=0).mean()))
_frac = np.clip((_lv0 - LOGVAR_MIN) / (LOGVAR_MAX - LOGVAR_MIN), 1e-3, 1 - 1e-3)
LOG_VAR_BIAS = float(np.log(_frac / (1 - _frac)))
print(f"target spread: sigma0 = {np.sqrt(np.exp(_lv0)):.3f} px "
      f"({np.sqrt(np.exp(_lv0)) * DEG_PER_PIX:.4f} deg), log_var bias {LOG_VAR_BIAS:.3f}")

# -----------------------------------------------------------------------
# hyperparams
# -----------------------------------------------------------------------
EPOCHS        = 300
BATCH_SIZE    = 32
LEARNING_RATE = 9e-5

model_0 = build_model(LOG_VAR_BIAS)
model_0.summary()
print("number of params in cnn model:", model_0.count_params())

histories, weight_files = [], []

for i in range(N_MODELS):
    print(f"\n### Training ensemble member {i + 1}/{N_MODELS} ###")
    tf.keras.utils.set_random_seed(20 + i)

    model = build_model(LOG_VAR_BIAS)
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss=negative_log_likelihood, metrics=[mu_rmse])

    wfile = os.path.join(SAVE_PATH, f"best_weights_dist_{TAG}_M{i}.weights.h5")
    weight_files.append(wfile)

    cbs = [
        EarlyStopping(monitor='val_loss', patience=25, verbose=1, min_delta=1e-6),
        ModelCheckpoint(filepath=wfile, save_best_only=True, save_weights_only=True,
                        monitor='val_loss', mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1,
                          patience=10, min_lr=1e-8),
    ]

    h = model.fit([train_ims, train_aux_s], train_tgt,
                  validation_data=([valid_ims, valid_aux_s], valid_tgt),
                  batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=cbs)
    histories.append(h)

print(f"training time: {(time.time() - start_time) / 60.:.1f} min")

# -----------------------------------------------------------------------
# loss curves
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = plt.cm.viridis(np.linspace(0, 0.85, N_MODELS))

for i, (h, c) in enumerate(zip(histories, colors)):
    axes[0].plot(h.history['loss'],     color=c, linestyle='-',  label=f'M{i} train')
    axes[0].plot(h.history['val_loss'], color=c, linestyle='-.', label=f'M{i} val')
    axes[1].plot(h.history['mu_rmse'],     color=c, linestyle='-')
    axes[1].plot(h.history['val_mu_rmse'], color=c, linestyle='-.')

axes[0].set_ylabel('Gaussian NLL')
axes[1].set_ylabel('RMSE of mean [px]')
for ax in axes:
    ax.set_xlabel('Epoch', fontsize=13)
axes[0].legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f'train_dist_{TAG}.png'), dpi=200)

# -----------------------------------------------------------------------
# ensemble prediction
#
# predictive variance of a gaussian mixture is mean(var_i) + var(mu_i): the
# aleatoric part each member reports, plus the epistemic spread between members.
# -----------------------------------------------------------------------
def ensemble_predict(ims, aux_s):
    preds = []
    for i, wfile in enumerate(weight_files):
        m = build_model(LOG_VAR_BIAS)
        m.load_weights(wfile)
        preds.append(m.predict([ims, aux_s], batch_size=BATCH_SIZE, verbose=0))
    preds = np.stack(preds)                       # (M, N, 4)
    mus, lvs = preds[:, :, :2], preds[:, :, 2:]
    mu = mus.mean(axis=0)
    var = np.exp(lvs).mean(axis=0) + mus.var(axis=0)
    return mu, var, np.exp(lvs).mean(axis=0), mus.var(axis=0)


print("\n### Predicting on the held-out test set ###")
mu_te, var_te, var_ale, var_epi = ensemble_predict(ims_te, test_aux_s)
sig_te = np.sqrt(var_te)

resid = tgt_te - mu_te                                   # px
sig_rad = np.sqrt(0.5 * (var_te[:, 0] + var_te[:, 1]))   # px, per-axis average
err_before = np.hypot(tgt_te[:, 0], tgt_te[:, 1]) * DEG_PER_PIX
err_after  = np.hypot(resid[:, 0], resid[:, 1]) * DEG_PER_PIX

print(f"offset from patch centre : median {np.median(err_before):.4f} deg")
print(f"after refinement         : median {np.median(err_after):.4f} deg")
print(f"median predicted sigma   : {np.median(sig_rad) * DEG_PER_PIX:.4f} deg")

# -----------------------------------------------------------------------
# calibration
#
# z = (truth - mu) / sigma has to be a unit gaussian. std(z) < 1 means the
# uncertainties are inflated, > 1 means overconfident.
# -----------------------------------------------------------------------
z = resid / sig_te
print(f"z-score  mean {z.mean():.3f}  std {z.std():.3f}  "
      f"(x: {z[:, 0].std():.3f}, y: {z[:, 1].std():.3f})")

# the parity split that motivated frac_x / frac_y. with them as inputs the two
# rows should now agree; if 'odd' still carries a ~+0.5 px residual the aux
# columns are not reaching the network.
if USE_FRAC:
    par = meta_te['pred_x'].to_numpy().astype(int) % 2 == 1
    for lbl, m in (('even', ~par), ('odd', par)):
        if m.sum():
            print(f"  pred_x {lbl:<4s} (n={int(m.sum()):5d})  "
                  f"target mean {tgt_te[m, 0].mean():+.3f}  "
                  f"residual {resid[m, 0].mean():+.3f} px")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

grid = np.linspace(-4, 4, 200)
axes[0].hist(z.reshape(-1), bins=60, range=(-4, 4), density=True,
             color='steelblue', alpha=0.75, label='pooled x and y')
axes[0].plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi),
             'k--', lw=2, label='N(0, 1)')
axes[0].set_xlabel(r'$z = (y_{\rm true} - \mu) / \sigma$', fontsize=12)
axes[0].set_ylabel('density')
axes[0].set_title(f'std(z) = {z.std():.3f}', fontsize=12)
axes[0].legend(fontsize=10)

ks = np.linspace(0.1, 3.0, 40)
emp = [(np.abs(z) < k).mean() for k in ks]
exp = [tf.math.erf(k / np.sqrt(2)).numpy() for k in ks]
axes[1].plot(exp, emp, '-', color='navy', lw=2)
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5)
axes[1].set_xlabel('expected coverage', fontsize=12)
axes[1].set_ylabel('observed coverage', fontsize=12)
axes[1].set_title('coverage of the predicted intervals', fontsize=12)


# reliability: does a larger predicted sigma actually mean a larger error
def bin_edges(values, log=False):
    '''quantile/spread bins sized to the sample, so a small test set still
    produces a readable curve instead of an empty one.'''
    n = int(np.clip(len(values) // 25, 2, 8))
    if log:
        return np.geomspace(np.nanpercentile(values, 2),
                            np.nanpercentile(values, 98), n + 1)
    return np.quantile(values[np.isfinite(values)], np.linspace(0, 1, n + 1))


MIN_PER_BIN = 5

q = bin_edges(sig_rad)
cent, obs = [], []
for lo, hi in zip(q[:-1], q[1:]):
    m = (sig_rad >= lo) & (sig_rad < hi)
    if m.sum() < MIN_PER_BIN:
        continue
    cent.append(np.median(sig_rad[m]))
    obs.append(np.sqrt(np.mean(resid[m] ** 2)))

if len(cent) >= 2:
    axes[2].plot(np.array(cent) * DEG_PER_PIX, np.array(obs) * DEG_PER_PIX,
                 'o-', color='crimson')
    lims = [min(cent) * DEG_PER_PIX * 0.8, max(cent) * DEG_PER_PIX * 1.2]
    axes[2].plot(lims, lims, 'k--', lw=1.5)
else:
    axes[2].text(0.5, 0.5, f'too few test sources\n({len(sig_rad)} available)',
                 ha='center', va='center', transform=axes[2].transAxes, fontsize=11)
axes[2].set_xlabel(r'predicted $\sigma$ [deg]', fontsize=12)
axes[2].set_ylabel('observed RMS residual [deg]', fontsize=12)
axes[2].set_title('reliability', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f'calibration_dist_{TAG}.png'), dpi=250)

# -----------------------------------------------------------------------
# the physics check: sigma against source brightness
#
# brightness is passed via counts
# -----------------------------------------------------------------------
flux = meta_te['test_flux_1000'].to_numpy()
snr = meta_te['test_SNR'].to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, xv, xlabel, logx in [
        (axes[0], flux, r'true flux $>$ 1 GeV [ph cm$^{-2}$ s$^{-1}$]', True),
        (axes[1], snr,  'test SNR', False)]:
    good = np.isfinite(xv) & (xv > 0 if logx else np.isfinite(xv))
    edges = bin_edges(xv[good], log=logx)
    mids, med, p68 = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = good & (xv >= lo) & (xv < hi)
        if m.sum() < MIN_PER_BIN:
            continue
        mids.append(np.sqrt(lo * hi) if logx else 0.5 * (lo + hi))
        med.append(np.median(sig_rad[m]) * DEG_PER_PIX)
        p68.append(np.percentile(np.hypot(resid[m, 0], resid[m, 1]), 68) * DEG_PER_PIX)
    ax.plot(mids, med, 'o-', color='navy', label=r'predicted $\sigma$')
    ax.plot(mids, p68, 's--', color='darkorange', label='observed 68% radius')
    if logx:
        ax.set_xscale('log')
    if len(med) and min(med) > 0:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(r'$\sigma$ [deg]', fontsize=12)
    ax.legend(fontsize=10)

axes[2].hist(err_before, bins=40, histtype='step', lw=2, color='grey',
             label='offset from patch centre')
axes[2].hist(err_after, bins=40, histtype='step', lw=2, color='crimson',
             label='after refinement')
axes[2].set_xlabel('positional error [deg]', fontsize=12)
axes[2].set_ylabel('sources')
axes[2].legend(fontsize=10)

plt.suptitle('localisation uncertainty vs source brightness — no pred_SNR', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f'sigma_vs_brightness_{TAG}.png'), dpi=250)

# -----------------------------------------------------------------------
# results file
# -----------------------------------------------------------------------
div = ddl.EBIN_DIV[ddl.REF_EBIN]

df_out = pd.DataFrame({
    'filename':      meta_te['filename'],
    'class':         meta_te['class'],
    'lon_ps':        meta_te['lon_ps'],
    'lat_ps':        meta_te['lat_ps'],
    'test_lon':      meta_te['test_lon'],
    'test_lat':      meta_te['test_lat'],
    'pred_x':        meta_te['pred_x'],
    'pred_y':        meta_te['pred_y'],
    'frac_x':        meta_te['frac_x'],
    'frac_y':        meta_te['frac_y'],
    'true_dx':       tgt_te[:, 0],
    'true_dy':       tgt_te[:, 1],
    'model_dx':      mu_te[:, 0],
    'model_dy':      mu_te[:, 1],
    'sigma_x_px':    sig_te[:, 0],
    'sigma_y_px':    sig_te[:, 1],
    'sigma_x_deg':   sig_te[:, 0] * DEG_PER_PIX,
    'sigma_y_deg':   sig_te[:, 1] * DEG_PER_PIX,
    'sigma_rad_deg': sig_rad * DEG_PER_PIX,
    'var_aleatoric': var_ale.mean(axis=1),
    'var_epistemic': var_epi.mean(axis=1),
    # refined absolute position, back on the native detection grid
    'refined_x':     (meta_te['centre_x'].to_numpy() + mu_te[:, 0]) * div,
    'refined_y':     (meta_te['centre_y'].to_numpy() + mu_te[:, 1]) * div,
    'err_before_deg': err_before,
    'err_after_deg':  err_after,
    'distance_degree': meta_te['distance_degree'],
    'test_flux_1000': meta_te['test_flux_1000'],
    'test_SNR':      meta_te['test_SNR'],
    'pred_SNR':      meta_te['pred_SNR'],     # kept for comparison, not an input
    'probs':         meta_te['probs'],
})

out_dir = os.path.join(BASE_PATH, f'dist_{TAG}')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, f'PosNLL_{TAG}_Ensemble{N_MODELS}_test_Pred.csv')
df_out.to_csv(out_file, sep=',', index=False)
print("predictions saved to:", out_file)
print(f"deg per {ddl.REF_EBIN} pixel used for all degree columns: {DEG_PER_PIX:.5f}")

print(f"total time: {(time.time() - start_time) / 60.:.1f} min")
