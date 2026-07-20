import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

TBINS = [f"tbin{i}" for i in range(1, 10)]
EBINS = ['bin1_2', 'bin2_7', 'bin7_20']
REF_TBIN = 'tbin1'


def _tbin_folder(base_path, ebin, tbin):
    return os.path.join(base_path, ebin, f"cat90_91_910_rad8_s20_predSNR_check_{tbin}")


def _norm_max(arr):
    return arr / (np.max(arr) + 1e-9)


def _load_cumulative_channel(base_path, filename, ebin, use_poisson):
    """sum raw patch across all tbins for one energy bin, then normalise."""
    cumsum = None
    for tbin in TBINS:
        path = os.path.join(_tbin_folder(base_path, ebin, tbin), filename)
        patch = np.load(path)
        if patch.ndim == 3:
            patch = patch[:, :, 0]
        cumsum = patch.copy() if cumsum is None else cumsum + patch
    if use_poisson:
        cumsum = poisson.rvs(np.maximum(cumsum.astype(float), 0))
    return _norm_max(cumsum)


def read_train_sources(csvfile, cls_label: list, base_path: str, use_poisson=True):
    """
    load (20, 20, 3) cumul. patches for the specified source classes.

    patches are summed over tbin0-tbin9 per energy bin, then the three bins
    (1-2, 2-7, 7-20+20-1000 GeV) are stacked as channels.

    args:
        csvfile:    csv fname, read from the REF_TBIN folder of bin1_2
        cls_label:  list of class ids to include, e.g. [0, 1, 2, 3]
        base_path:  root path, e.g.
                    '.../check_patches_classification/all_sources/corrected_S20_LoG_Th0d1_ol0d2'
        use_poisson: apply poisson resampling to the cumulative sum before normalising

    returns:
        images  np.ndarray (N, 20, 20, 3)
        labels  np.ndarray (N, 1)
        lats    np.ndarray (N,)
        lons    np.ndarray (N,)
        fluxes  np.ndarray (N, 1)  — test_flux_1000
        names   list[str]
    """
    ref_folder = _tbin_folder(base_path, EBINS[0], REF_TBIN)
    df = pd.read_csv(os.path.join(ref_folder, csvfile), sep=',')

    images, labels, lats, lons, fluxes, names = [], [], [], [], [], []

    for _, row in df.iterrows():
        if int(row['class']) not in cls_label:
            continue

        fname = row['filename']
        ch1 = _load_cumulative_channel(base_path, fname, 'bin1_2',  use_poisson)
        ch2 = _load_cumulative_channel(base_path, fname, 'bin2_7',  use_poisson)
        ch3 = _load_cumulative_channel(base_path, fname, 'bin7_20', use_poisson)

        images.append(np.stack([ch1, ch2, ch3], axis=-1))  # (20, 20, 3)
        labels.append(int(row['class']))
        lats.append(float(row['test_lat']))
        lons.append(float(row['test_lon']))
        fluxes.append(float(row['test_flux_1000']))
        names.append(fname)

    return (np.array(images),
            np.array(labels).reshape(-1, 1),
            np.array(lats),
            np.array(lons),
            np.array(fluxes).reshape(-1, 1),
            names)


def aug_images(images, rotate_90=False, rotate_lr_90=False):
    """
    augment a batch of (N, 20, 20, 3) images.

    rotate_90=False, rotate_lr_90=False  →  3x  (original + h-flip + v-flip)
    rotate_90=True,  rotate_lr_90=True   →  5x  (+ rot90 + rot270)

    args:
        images: np.ndarray (N, 20, 20, 3)
    returns:
        np.ndarray (N*k, 20, 20, 3)
    """
    parts = [
        images,
        images[:, :, ::-1, :],              # horz. flp
        images[:, ::-1, :, :],              # vert. flp
    ]
    if rotate_90:
        parts.append(np.rot90(images, k=1, axes=(1, 2)))   # 90°
    if rotate_lr_90:
        parts.append(np.rot90(images, k=3, axes=(1, 2)))   # 270°
    return np.concatenate(parts, axis=0)
