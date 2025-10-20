# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# ---------- tool ----------
def load_zero_mask(path):
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    return np.any(arr == 0, axis=2)        # (H,W) boolean mask

def scan_all(dir_a, dir_b):
    """
    Dictionary:
      zero_a_ratio : {fname: zero elements of normal-light / total pixels}
      zero_b_ratio : {fname: zero elements of low-light / total pixels}
      miss_ratio   : {fname: missing pixels / total zero elements in low-light}
    """
    zero_a, zero_b, miss = {}, {}, {}
    fnames = [f for f in os.listdir(dir_a)
              if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}]
    assert set(fnames) == set(os.listdir(dir_b)), "File names are inconsistent!"

    for fname in tqdm(fnames, desc="Processing"):
        mask_a = load_zero_mask(os.path.join(dir_a, fname))
        mask_b = load_zero_mask(os.path.join(dir_b, fname))
        total = mask_a.size          # H*W

        # Ratio of zero elements
        za = mask_a.sum() / total
        zb = mask_b.sum() / total

        # Missing information pixels
        missing_mask = mask_b & (~mask_a)
        missing_count = missing_mask.sum()
        zero_b_count = mask_b.sum()

        # Avoid division by zero: if low-light image has no zero elements, define missing ratio as 0
        zm = (missing_count / zero_b_count) if zero_b_count else 0.0

        zero_a[fname] = za
        zero_b[fname] = zb
        miss[fname]   = zm
    return zero_a, zero_b, miss

# ---------- Plotting ----------
def plot_three(zero_a, zero_b, miss, save_path=None):

    # ---- Increase global font size ----
    plt.rcParams.update({'font.size': 22})

    # Sort by respective ratios in ascending order
    items_a = sorted(zero_a.items(), key=lambda x: x[1])
    items_b = sorted(zero_b.items(), key=lambda x: x[1])
    items_m = sorted(miss.items(),   key=lambda x: x[1])

    def extract(items):
        y = [v for _, v in items]
        x = np.arange(len(y))
        return x, y

    x_a, y_a = extract(items_a)
    x_b, y_b = extract(items_b)
    x_m, y_m = extract(items_m)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_a, y_a, s=15, c='red',  label='High-light image')
    plt.scatter(x_b, y_b, s=15, c='blue',   label='Low-light image')
    plt.scatter(x_m, y_m, s=15, c='green', label='First class')

    # ---------- New: Dashed line ----------
    # Find the last index where miss_ratio == 0
    zero_indices = [i for i, v in enumerate(y_m) if v == 0.0]
    if zero_indices:
        split_x = zero_indices[-1] + 0.5   # +0.5 places it to the right of the last zero
        plt.axvline(x=split_x, color='gray', linestyle='--', linewidth=1.5)

    # --------------------------------

    plt.xlabel('Image number (sorted by respective ratio)')
    plt.ylabel('Zero-element pixel ratio')
    plt.title('SMID testing set')
    # plt.legend()#loc='center'

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ---------- Main entry ----------
if __name__ == '__main__':
    dir_a = r'D:\Database\SMID\Test\high'   # Normal-light
    dir_b = r'D:\Database\SMID\Test\low'    # Low-light
    zero_a, zero_b, miss = scan_all(dir_a, dir_b)
    plot_three(zero_a, zero_b, miss, save_path='SMID-test.png')
