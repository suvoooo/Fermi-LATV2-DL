import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

# 10 time bins because we had yearly data
TBINS = [f"tbin{i}" for i in range(1, 11)]
EBINS = ['bin1_2', 'bin2_7', 'bin7_20']
REF_TBIN = 'tbin1'


def _tbin_folder(base_path, ebin, tbin):
    return os.path.join(base_path, ebin, f"cat90_91_910_rad8_s20_predSNR_check_{tbin}")


def _norm_max(arr):
    return arr / (np.max(arr) + 1e-9)


def _load_cube(base_path, filename, use_poisson):
    """
    load one source as a (H, W, n_ebins, n_tbins) cube.

    each (ebin, tbin) patch is poisson-resampled and then max-normalised
    independently — same order as the flux loader, so the two stay comparable.
    time bins are kept separate here (not summed as in the flux case): a real
    source persists across them, a background fluctuation does not.
    """
    cube = None
    for ie, ebin in enumerate(EBINS):
        for it, tbin in enumerate(TBINS):
            path = os.path.join(_tbin_folder(base_path, ebin, tbin), filename)
            patch = np.load(path)
            if patch.ndim == 3:
                patch = patch[:, :, 0]
            if cube is None:                       # infer patch size from the data
                cube = np.empty(patch.shape + (len(EBINS), len(TBINS)), dtype=np.float32)
            if use_poisson:
                patch = poisson.rvs(np.maximum(patch.astype(float), 0))
            cube[:, :, ie, it] = _norm_max(patch)
    return cube


def read_sources(csvfile, cls_label: list, binary_label: int, base_path: str,
                 snr_min=None, use_poisson=True):
    """
    load (H, W, E, T) time-resolved patches for the specified source classes.

    the three energy bins (1-2, 2-7, 7-20+20-1000 GeV) form the third axis and
    the nine time bins the fourth, so a Conv3D sees (H, W, ebin) as spatial dims
    with time as channels — the layout the old classifier used.

    args:
        csvfile:      csv fname, read from the REF_TBIN folder of EBINS[0]
        cls_label:    list of class ids to include; [0, 1, 2, 3] = real sources,
                      [4] = fake
        binary_label: label written for every row kept — 0 for real, 1 for fake
        base_path:    root path, e.g.
                      '.../Fermi-Flux1000/corrected_S20_LoG_Th0d1_ol0d2'
        snr_min:      keep only rows with test_SNR > snr_min; None = no cut
        use_poisson:  poisson-resample each patch before normalising

    returns:
        images  np.ndarray (N, H, W, E, T)  float32
        labels  np.ndarray (N, 1)
        lats    np.ndarray (N,)
        lons    np.ndarray (N,)
        probs   np.ndarray (N,)
        names   list[str]
    """
    ref_folder = _tbin_folder(base_path, EBINS[0], REF_TBIN)
    df = pd.read_csv(os.path.join(ref_folder, csvfile), sep=',')
    # the S10 csv wrote ' probs' with a leading space, the S20 one writes 'probs'
    df.columns = df.columns.str.strip()

    if snr_min is not None:
        n_before = len(df)
        df = df.loc[df['test_SNR'] > snr_min]
        print(f"  SNR > {snr_min}: {n_before} -> {len(df)} rows")

    images, lats, lons, probs, names = [], [], [], [], []

    for _, row in df.iterrows():
        if int(row['class']) not in cls_label:
            continue

        fname = row['filename']
        images.append(_load_cube(base_path, fname, use_poisson))
        lats.append(float(row['test_lat']))
        lons.append(float(row['test_lon']))
        probs.append(float(row['probs']))
        names.append(fname)

    labels = np.full((len(images), 1), binary_label, dtype=np.float32)

    return (np.array(images, dtype=np.float32),
            labels,
            np.array(lats),
            np.array(lons),
            np.array(probs),
            names)


def aug_images(images, rotate_90=False, rotate_lr_90=False):
    """
    augment a batch of (N, H, W, E, T) cubes.

    flips and rotations act on the two spatial axes only; energy and time ride
    along untouched.

    rotate_90=False, rotate_lr_90=False  ->  3x  (original + h-flip + v-flip)
    rotate_90=True,  rotate_lr_90=True   ->  5x  (+ rot90 + rot270)

    blocks are stacked whole (all sources, then all h-flips, ...), so tiling a
    label array k times keeps it aligned with the images.

    args:
        images: np.ndarray (N, H, W, E, T)
    returns:
        np.ndarray (N*k, H, W, E, T)
    """
    parts = [
        images,
        images[:, :, ::-1],                 # horz. flp
        images[:, ::-1],                    # vert. flp
    ]
    if rotate_90:
        parts.append(np.rot90(images, k=1, axes=(1, 2)))   # 90°
    if rotate_lr_90:
        parts.append(np.rot90(images, k=3, axes=(1, 2)))   # 270°
    return np.concatenate(parts, axis=0)
