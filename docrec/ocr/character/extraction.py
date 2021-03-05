import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt


def extract_characters(
    image, offset=(0, 0), max_width=None
):
    ''' Extract characters from a text box.'''

    # Char box width range
    h, w = image.shape
    if max_width is None:
        max_width = w

    # Filtering
    labels = measure.label(image)
    props = measure.regionprops(labels)
    props = [region for region in props if region.image.shape[1] < max_width]

    # Merge characters
    offset_x, offset_y = offset
    min_overlap = 0.5
    chars = []
    left = set(range(len(props)))
    while left:
        i = min(left)
        left.remove(i)
        merged = [i]
        region_i = props[i]
        yi, xi, _, _ = region_i.bbox
        hi, wi = region_i.image.shape
        for j in left:
            region_j = props[j]
            yj, xj, _, _ = region_j.bbox
            hj, wj = region_j.image.shape
            intersection =  min(xi + wi - 1, xj + wj - 1) - max(xi, xj)
            overlap = float(intersection) / min(wi, wj)
            if overlap >= min_overlap:
                merged.append(j)
                xm = min(xi, xj)
                ym = min(yi, yj)
                wm = max(xi + wi, xj + wj) - xm
                hm = max(yi + hi, yj + hj) - ym
                xi, yi, wi, hi = xm, ym, wm, hm
        left -= set(merged)
        patch = np.zeros((hi, wi), dtype=np.uint8)
        for k in merged:
            region_k = props[k]
            yk, xk, _, _ = region_k.bbox
            hk, wk = region_k.image.shape
            x = xk - xi
            y = yk - yi
            patch[y : y + hk, x : x + wk] = cv2.bitwise_or(
                patch[y : y + hk, x : x + wk],
                255 * region_k.image.astype(np.uint8)
            )
        chars.append(((offset_x + xi, offset_y + yi, wi, hi), patch))
    return chars