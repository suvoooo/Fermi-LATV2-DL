'''
loader for the position-refinement / localisation-uncertainty pipeline.

two csv files are needed and they must agree row-for-row:

  patch csv  — contains patch (20x20) names, original predicted location from LoG;
               this was created when we cut the smaller patches from the big one based on the pred locs. 
  stat  csv  — the uneLoG evaluation output that the patches were cut from. contains the predicted and true locs.

the target is the offset of the true source from the patch centre, in REF_EBIN
pixels. 
'''

import os
import numpy as np
import pandas as pd
from scipy.stats import poisson

TBINS = [f"tbin{i}" for i in range(1, 11)]
EBINS = ['bin1_2', 'bin2_7', 'bin7_20'] # 7-20 and >20 are summed as usual

# the frame the coords are regressed in. every ebin sits on its own grid,
# so "one pixel" means something different in each of them.
REF_EBIN = 'bin2_7'
REF_TBIN = 'tbin1'

# mirrors bin_en() in stat_to_patch_for_class_gen_allS_LoG_Size20.py
EBIN_DIV = {'bin7_20': 1, 'bin2_7': 2, 'bin1_2': 4,
            'bin0d5_1': 8, 'bin0d3_0d5': 8}

BOX_INF = 10        # patches are cut as [centre-10, centre+10)


def _tbin_folder(base_path, ebin, tbin):
    return os.path.join(base_path, ebin, f"cat90_91_910_rad8_s20_predSNR_check_{tbin}")


def _patch_centre(pred_coord, div):
    '''grid coordinate the 20x20 box was centred on. written the long way on
    purpose so it matches stat_to_patch's truncation exactly.'''
    return int(int(pred_coord) / div)


def read_stat_rows(stat_csv):
    '''
    replay stat_to_patch's iteration order so the returned frame lines up
    row-for-row with the patch csv it produced: patches in first-seen order,
    stat_code 2.0 dropped, original row order within each patch.
    '''
    df = pd.read_csv(stat_csv, sep=',')
    df.columns = df.columns.str.strip()
    parts = [df[(df["patch_number"] == patch) & (df["stat_code"] != 2.0)]
             for patch in df["patch_number"].unique()]
    return pd.concat(parts, ignore_index=True)


def _load_cumulative(base_path, filename, ebin, use_poisson):
    '''
    sum one source's patch over all time bins for one energy bin.

    returns the raw counts, *not* normalised — the caller needs the un-normalised
    array to measure brightness, which is exactly what max-normalisation destroys.
    '''
    cumsum = None
    for tbin in TBINS:
        path = os.path.join(_tbin_folder(base_path, ebin, tbin), filename)
        patch = np.load(path)
        if patch.ndim == 3:
            patch = patch[:, :, 0]
        patch = patch.astype(np.float64)
        cumsum = patch.copy() if cumsum is None else cumsum + patch
    if use_poisson:
        cumsum = poisson.rvs(np.maximum(cumsum, 0)).astype(np.float64)
    return cumsum


def _norm_max(arr):
    return arr / (np.max(arr) + 1e-9)


def read_sources(patch_csv, stat_csv, base_path, cls_label, dist_max,
                 snr_min=None, use_poisson=True, crop=True, use_pred_snr=True,
                 use_frac=False, verbose=True):
    '''
    load true-positive sources with their position-offset targets.

    returns:
        images   (N, S, S, 3) float32, max-normalised per channel, S = 19 or 20
        aux      (N, K) float32 — log10 total counts, log10 peak counts,
                 [pred_SNR], [frac_x, frac_y], |lat_ps|, in that order. the exact
                 list is meta.attrs['aux_names']. all available at inference time.
        targets  (N, 2) float32 — (dx, dy) offset from the reference pixel,
                 in REF_EBIN pixels, float precision
        meta     DataFrame, one row per kept source
        centre   int, index of the reference pixel along each spatial axis
    '''
    ref_folder = _tbin_folder(base_path, REF_EBIN, REF_TBIN)
    patch_df = pd.read_csv(os.path.join(ref_folder, patch_csv), sep=',')
    patch_df.columns = patch_df.columns.str.strip()

    stat_df = read_stat_rows(stat_csv)

    if len(patch_df) != len(stat_df):
        raise ValueError(
            f"patch csv has {len(patch_df)} rows but the replayed stat csv has "
            f"{len(stat_df)} — the two files do not correspond. check that "
            f"{os.path.basename(stat_csv)} is the file the patches were cut from.")

    # ------------------------------------------------------------------
    # alignment check before joining cols. 
    # ------------------------------------------------------------------
    for col in ('test_lat', 'test_lon'):
        drift = np.abs(patch_df[col].to_numpy() - stat_df[col].to_numpy())
        if np.nanmax(drift) > 1e-6:
            raise ValueError(f"patch/stat csv rows disagree on {col} "
                             f"(max drift {np.nanmax(drift):.3g}) — order mismatch")
    if not np.array_equal(patch_df['class'].to_numpy(), stat_df['class'].to_numpy()):
        raise ValueError("patch/stat csv rows disagree on class — order mismatch")

    # ------------------------------------------------------------------
    # targets: offset of the true source from the pixel the box was cut around
    # ------------------------------------------------------------------
    div = EBIN_DIV[REF_EBIN]
    cx = np.array([_patch_centre(p, div) for p in stat_df['pred_x']], dtype=np.float64)
    cy = np.array([_patch_centre(p, div) for p in stat_df['pred_y']], dtype=np.float64)

    dx = stat_df['test_x'].to_numpy(dtype=np.float64) / div - cx
    dy = stat_df['test_y'].to_numpy(dtype=np.float64) / div - cy

    
    frac_x = stat_df['pred_x'].to_numpy(dtype=np.float64).astype(int) / div - cx
    frac_y = stat_df['pred_y'].to_numpy(dtype=np.float64).astype(int) / div - cy

    # ------------------------------------------------------------------
    # true-positive select.
    # ------------------------------------------------------------------
    centre = BOX_INF - 1 if crop else BOX_INF
    reach = centre if crop else BOX_INF - 1

    with np.errstate(invalid='ignore'):
        keep = patch_df['class'].isin(cls_label).to_numpy()
        keep &= (stat_df['distance_degree'].to_numpy() <= dist_max)
        keep &= np.isfinite(dx) & np.isfinite(dy)
        keep &= (np.abs(dx) <= reach) & (np.abs(dy) <= reach)
        if snr_min is not None:
            keep &= (patch_df['pred_SNR'].to_numpy() > snr_min)

    if verbose:
        print(f"  selection: {len(keep)} -> {int(keep.sum())} rows "
              f"(class {cls_label}, distance_degree <= {dist_max}"
              f"{'' if snr_min is None else f', pred_SNR > {snr_min}'})")

    rebuilt_x = (dx[keep] + BOX_INF).astype(int)
    rebuilt_y = (dy[keep] + BOX_INF).astype(int)
    bad = ((rebuilt_x != patch_df['test_x'].to_numpy()[keep].astype(int)) |
           (rebuilt_y != patch_df['test_y'].to_numpy()[keep].astype(int)))
    if verbose:
        print(f"  target rebuild check: {bad.sum()} / {len(bad)} selected rows "
              f"disagree with the truncated patch-csv columns")
    if bad.mean() > 0.01:
        raise ValueError(
            f"{100 * bad.mean():.1f}% of rebuilt targets disagree with the patch "
            f"csv — bin_en arithmetic or REF_EBIN ({REF_EBIN}) is wrong")

    sel_patch = patch_df.loc[keep].reset_index(drop=True)
    sel_stat = stat_df.loc[keep].reset_index(drop=True)
    dx, dy = dx[keep], dy[keep]
    cx, cy = cx[keep], cy[keep]
    frac_x, frac_y = frac_x[keep], frac_y[keep]

    # ------------------------------------------------------------------
    # images + brightness scalars
    # ------------------------------------------------------------------
    images, totals, peaks = [], [], []
    for fname in sel_patch['filename']:
        chans, tot, pk = [], 0.0, 0.0
        for ebin in EBINS:
            raw = _load_cumulative(base_path, fname, ebin, use_poisson)
            tot += float(raw.sum())
            pk = max(pk, float(raw.max()))
            chans.append(_norm_max(raw))
        stack = np.stack(chans, axis=-1)
        if crop:
            stack = stack[1:2 * BOX_INF, 1:2 * BOX_INF]
        images.append(stack)
        totals.append(tot)
        peaks.append(pk)

    images = np.array(images, dtype=np.float32)

    aux_cols = [np.log10(1.0 + np.array(totals)),
                np.log10(1.0 + np.array(peaks))]
    aux_names = ['log10 total counts', 'log10 peak counts']
    if use_pred_snr:
        aux_cols.append(sel_patch['pred_SNR'].to_numpy(dtype=np.float64))
        aux_names.append('pred_SNR')
    if use_frac:
        aux_cols += [frac_x, frac_y]
        aux_names += ['frac_x', 'frac_y']
    aux_cols.append(np.abs(sel_patch['lat_ps'].to_numpy(dtype=np.float64)))
    aux_names.append('|lat_ps|')

    aux = np.stack(aux_cols, axis=-1).astype(np.float32)

    targets = np.stack([dx, dy], axis=-1).astype(np.float32)

    meta = pd.DataFrame({
        'filename': sel_patch['filename'].to_numpy(),
        'class': sel_patch['class'].to_numpy().astype(int),
        'lon_ps': sel_patch['lon_ps'].to_numpy(),
        'lat_ps': sel_patch['lat_ps'].to_numpy(),
        'test_lon': sel_patch['test_lon'].to_numpy(),
        'test_lat': sel_patch['test_lat'].to_numpy(),
        'pred_x': sel_stat['pred_x'].to_numpy(),
        'pred_y': sel_stat['pred_y'].to_numpy(),
        'centre_x': cx,
        'centre_y': cy,
        'frac_x': frac_x,
        'frac_y': frac_y,
        'true_dx': dx,
        'true_dy': dy,
        'distance_degree': sel_stat['distance_degree'].to_numpy(),
        'distance_pixel': sel_stat['distance_pixel'].to_numpy(),
        'test_flux_1000': sel_patch['test_flux_1000'].to_numpy(),
        'test_SNR': sel_patch['test_SNR'].to_numpy(),
        'pred_SNR': sel_patch['pred_SNR'].to_numpy(),
        'probs': sel_patch['probs'].to_numpy(),
    })

    # the names travel with the data so the training script can write them into
    # the scaler file and inference can rebuild the vector by name, not by count
    meta.attrs['aux_names'] = aux_names

    if verbose:
        print(f"  images {images.shape}  aux {aux.shape}  targets {targets.shape}")
        print(f"  aux features: {', '.join(aux_names)}")
        if use_frac:
            print(f"  frac_x values: {np.unique(np.round(frac_x, 3))}")
        print(f"  offset spread px: dx {dx.std():.3f}  dy {dy.std():.3f}  "
              f"max |d| {np.abs(np.hypot(dx, dy)).max():.2f}")

    return images, aux, targets, meta, centre


def calibrate_deg_per_pixel(meta, verbose=True):
    '''
    degrees per REF_EBIN pixel, from distance_degree / distance_pixel.
    added later in the calcs...
    '''
    div = EBIN_DIV[REF_EBIN]
    dpx = meta['distance_pixel'].to_numpy()
    deg = meta['distance_degree'].to_numpy()

    ok = (dpx > 0.5) & np.isfinite(dpx) & np.isfinite(deg)
    deg_per_ref_px = div * np.median(deg[ok] / dpx[ok])

    test_x = (meta['centre_x'].to_numpy() + meta['true_dx'].to_numpy()) * div
    test_y = (meta['centre_y'].to_numpy() + meta['true_dy'].to_numpy()) * div
    rebuilt = np.hypot(test_x - meta['pred_x'].to_numpy(),
                       test_y - meta['pred_y'].to_numpy())
    drift = np.median(np.abs(rebuilt - dpx))

    if verbose:
        print(f"  deg per {REF_EBIN} pixel: {deg_per_ref_px:.5f}")
        print(f"  distance_pixel rebuilt from coordinates, median drift "
              f"{drift:.4f} native px")
        if drift > 0.05:
            print("  WARNING: distance_pixel does not match the reconstruction, so "
                  "it is probably not on the native grid. every degree column "
                  "scales with this number — check it before quoting any sigma.")
    return deg_per_ref_px


# ---------------------------------------------------------------------------
# augment data
#
# every transform acts on the image *and* the target. images are indexed
# [row, col] = [y, x], and after the 19x19 crop the reference pixel is the exact
# centre, so the offsets transform as clean sign swaps with no half-pixel shift.
# ---------------------------------------------------------------------------
def _t_identity(d):  return d
def _t_hflip(d):     return np.stack([-d[:, 0],  d[:, 1]], axis=-1)
def _t_vflip(d):     return np.stack([ d[:, 0], -d[:, 1]], axis=-1)
def _t_rot90(d):     return np.stack([ d[:, 1], -d[:, 0]], axis=-1)
def _t_rot270(d):    return np.stack([-d[:, 1],  d[:, 0]], axis=-1)


def _i_identity(im): return im
def _i_hflip(im):    return im[:, :, ::-1, :]
def _i_vflip(im):    return im[:, ::-1, :, :]
def _i_rot90(im):    return np.rot90(im, k=1, axes=(1, 2))
def _i_rot270(im):   return np.rot90(im, k=3, axes=(1, 2))


_TRANSFORMS = [
    ('identity', _i_identity, _t_identity),
    ('hflip',    _i_hflip,    _t_hflip),
    ('vflip',    _i_vflip,    _t_vflip),
    ('rot90',    _i_rot90,    _t_rot90),
    ('rot270',   _i_rot270,   _t_rot270),
]


def check_transforms(size, centre, verbose=True):
    '''
    numeric self-test of the image/target transform pairs.
    added later just for a sanity check!
    '''
    for dx, dy in [(3, 1), (-2, 4), (5, -5), (0, 2), (-4, 0)]:
        im = np.zeros((1, size, size, 1), dtype=np.float32)
        im[0, centre + dy, centre + dx, 0] = 1.0
        d = np.array([[dx, dy]], dtype=np.float32)

        for name, im_fn, t_fn in _TRANSFORMS:
            moved = im_fn(im)
            want = t_fn(d)[0]
            row, col = np.unravel_index(np.argmax(moved[0, :, :, 0]), (size, size))
            got = np.array([col - centre, row - centre], dtype=np.float32)
            if not np.array_equal(got, want):
                raise AssertionError(
                    f"transform '{name}' moves ({dx},{dy}) to {tuple(got)} "
                    f"but the target says {tuple(want)}")
    if verbose:
        print(f"  transform self-test passed for {size}x{size}, centre {centre}")


def aug_images(images, targets, aux, rotate_90=False, rotate_lr_90=False,
               aux_vec_cols=None):
    '''
    augment images, targets and aux together.

    rotate_90=False, rotate_lr_90=False  ->  3x  (original + h-flip + v-flip)
    rotate_90=True,  rotate_lr_90=True   ->  5x  (+ rot90 + rot270)

    returns:
        images, targets, aux, groups
    '''
    active = list(_TRANSFORMS[:3])
    if rotate_90:
        active.append(_TRANSFORMS[3])
    if rotate_lr_90:
        active.append(_TRANSFORMS[4])

    ims = np.concatenate([im_fn(images) for _, im_fn, _ in active], axis=0)
    tgt = np.concatenate([t_fn(targets) for _, _, t_fn in active], axis=0)

    aux_blocks = []
    for _, _, t_fn in active:
        block = aux.copy()
        if aux_vec_cols is not None:
            ix, iy = aux_vec_cols
            block[:, [ix, iy]] = t_fn(aux[:, [ix, iy]])
        aux_blocks.append(block)
    axs = np.concatenate(aux_blocks, axis=0)

    groups = np.tile(np.arange(len(images)), len(active))
    return ims, tgt.astype(np.float32), axs, groups


def shift_images(images, targets, aux, groups, centre, shifts=((1, 0), (0, 1)),
                 aux_vec_cols=None):
    '''
    translate the patch content by whole pixels, zero-filling the vacated edge,
    and move the target with it. similar to Fio's pipeline for ASID-FE

    returns:
        images, targets, aux, groups
    '''
    out_ims, out_tgt, out_aux = [], [], []
    for sx, sy in shifts:
        moved = np.zeros_like(images)
        ys_src = slice(max(0, -sy), images.shape[1] - max(0, sy))
        ys_dst = slice(max(0, sy), images.shape[1] - max(0, -sy))
        xs_src = slice(max(0, -sx), images.shape[2] - max(0, sx))
        xs_dst = slice(max(0, sx), images.shape[2] - max(0, -sx))
        moved[:, ys_dst, xs_dst] = images[:, ys_src, xs_src]
        out_ims.append(moved)
        out_tgt.append(targets + np.array([sx, sy], dtype=np.float32))

        # the whole patch content moved, so the detection moved with it
        block = aux.copy()
        if aux_vec_cols is not None:
            ix, iy = aux_vec_cols
            block[:, ix] += sx
            block[:, iy] += sy
        out_aux.append(block)

    ims = np.concatenate(out_ims, axis=0)
    tgt = np.concatenate(out_tgt, axis=0).astype(np.float32)
    axs = np.concatenate(out_aux, axis=0)
    grp = np.concatenate([groups] * len(shifts), axis=0)

    # drop anything the shift pushed outside the patch
    keep = (np.abs(tgt[:, 0]) <= centre) & (np.abs(tgt[:, 1]) <= centre)
    return ims[keep], tgt[keep], axs[keep], grp[keep]
