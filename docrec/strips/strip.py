import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

from ..ndarray.utils import first_nonzero, last_nonzero
from ..ocr.text.extraction import extract_text
from ..ocr.character.extraction import extract_characters


class Strip(object):
    ''' Strip image.'''


    def __init__(self, image, position, mask=None):

        h, w = image.shape[: 2]
        if mask is None:
            mask = 255 * np.ones((h, w), dtype=np.uint8)

        self.h = h
        self.w = w
        self.image = cv2.bitwise_and(image, image, mask=mask)
        self.position = position
        self.mask = mask

        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))


    def copy(self):
        ''' Copy object. '''

        return copy.deepcopy(self)


    def shift(self, disp):
        ''' shift strip vertically. '''

        M = np.float32([[1, 0, 0], [0, 1, disp]])
        self.image = cv2.warpAffine(self.image, M, (self.w, self.h))
        self.mask = cv2.warpAffine(self.mask, M, (self.w, self.h))
        self.offsets_l = np.apply_along_axis(first_nonzero, 1, self.mask) # left border (hor.) offsets
        self.offsets_r = np.apply_along_axis(last_nonzero, 1, self.mask)   # right border (hor.) offsets
        self.approx_width = int(np.mean(self.offsets_r - self.offsets_l + 1))
        return self


    def filled_image(self):
        ''' Return image with masked-out areas in white. '''

        return cv2.bitwise_or(
            self.image, cv2.cvtColor(
                cv2.bitwise_not(self.mask), cv2.COLOR_GRAY2RGB
            )
        )


    def is_blank(self, blank_tresh=127):
        ''' Check if is a blank strip. '''
        
        blurred = cv2.GaussianBlur(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY), (5, 5), 0
        )
        return (blurred < blank_tresh).sum() == 0

    # def is_blank(self, blank_tresh=127):
    #     ''' Check if is a blank strip. '''

    #     _, thresh = cv2.threshold(
    #         cv2.cvtColor(self..filled_image(), cv2.COLOR_RGB2GRAY), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    #     )
    #     return thresh.sum() == 0


    def stack(self, other, disp=0, filled=False):
        ''' Stack horizontally with other strip. '''

        y1_min, y1_max = 0, self.h - 1
        y2_min, y2_max = 0, other.h - 1
        y_inter_min = max(0, disp)
        y_inter_max = min(y1_max, y2_max + disp) + 1
        h_inter = y_inter_max - y_inter_min

        # borders coordinates
        r1 = self.offsets_r[y_inter_min : y_inter_max]
        if disp >= 0:
            l2 = other.offsets_l[: h_inter]
        else:
            l2 = other.offsets_l[-disp : -disp + h_inter]

        # horizontal offset
        offset = self.w - np.min(l2 + self.w - r1) + 1

        # union
        y_union_min = min(0, disp)
        y_union_max = max(y1_max, y2_max + disp) + 1
        h_union = y_union_max - y_union_min

        min_h, max_h = min(self.h, other.h), max(self.h, other.h)

        # new image / mask
        temp_image = np.zeros((h_union, offset + other.w, 3), dtype=np.uint8)
        temp_mask = np.zeros((h_union, offset + other.w), dtype=np.uint8)
        if disp >= 0:
            temp_image[: self.h, : self.w] = self.image
            temp_image[disp : disp + other.h, offset :] += other.image
            temp_mask[: self.h, : self.w] = self.mask
            temp_mask[disp : disp + other.h, offset :] += other.mask
        else:
            temp_image[-disp : -disp + self.h, : self.w] = self.image
            temp_image[: other.h, offset :] += other.image
            temp_mask[-disp : -disp + self.h, : self.w] = self.mask
            temp_mask[: other.h, offset :] += other.mask

        self.h, self.w = temp_mask.shape
        self.image = temp_image
        self.mask = temp_mask
        self.offsets_l =np.apply_along_axis(first_nonzero, 1, self.mask)
        self.offsets_r =np.apply_along_axis(last_nonzero, 1, self.mask)
        if filled:
            self.image = self.filled_image()
        return self


    # -----------------------------------------------
    # These methods are used by the character shape-based algorithm
    # ----------------------------------------------- 
    def extract_text(self, min_height, max_height, max_separation):
        '''Extract text information contained in the strip. '''

        self.text = extract_text(
            cv2.cvtColor(self.filled_image(), cv2.COLOR_RGB2GRAY),
            min_height, max_height, max_separation, 0.95
        )


    def extract_characters(self, d, min_height, max_height, max_extent):
        ''' Extract characters information. '''

        # Extent range
        left = []
        right = []
        inner = []
        if self.text:
            # Borders coordinates
            lb = self.offsets_l
            rb = self.offsets_r
            lb += d
            rb -= d
            # Character extraction
            chars = []
            for box, patch in self.text:
                x, y, w, h = box
                chars += extract_characters(
                    patch, offset=(x, y),
                    max_width=self.approx_width / 2
                )

            # Categorization
            for char in chars:
                box, patch = char
                x, y, w, h = box

                # Left ?
                if np.any(x <= lb[y : y + h]):
                    left.append(char)
                # Right ?
                elif np.any(x + w - 1 >= rb[y : y + h]):
                    right.append(char)
                # Inner !
                else:
                    inner.append(char)

        filtered_inner = []
        filtered_inner = inner

        # Filtering non-null characters (OpenCV findContour requirements)
        not_null = lambda char: cv2.findContours(
            char[1].copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )[1] != []
        self.inner = [char for char in filtered_inner if not_null(char)]
        self.left = [char for char in left if not_null(char)]
        self.right = [char for char in right if not_null(char)]
