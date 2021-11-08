import cv2
from skimage import measure


def extract_text(image, min_height, max_height, max_separation, max_extent):
    ''' Extract text regions from image. '''

    # Thresholded image
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Text candidates
    dx = int(2 * max_separation)
    dy = int(dx / 4)
    dx_dy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dx, dy))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, dx_dy)
    labels = measure.label(dilated)
    props = measure.regionprops(labels)

    # Filtering
    labels_to_remove = set([])
    for region in props:
        yt_min, _, yt_max, _ = region.bbox
        ht, wt = region.image.shape

        if (region.label in labels_to_remove) or (ht > max_height):# or \
#            (region.extent > max_extent):
            labels_to_remove.update(
                set(labels[yt_min : yt_max, : ].flatten())
            )
        elif (ht < min_height) or (wt < 2):
            labels_to_remove.update([region.label])

    text = []
    for region in props:
        if region.label not in labels_to_remove:
            y_min, x_min, y_max, x_max = region.bbox
            h, w = region.image.shape
            box = (x_min, y_min, w, h)
            patch = thresh[y_min : y_max, x_min : x_max]
            text.append((box, patch))
    return text