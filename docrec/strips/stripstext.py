import cv2
import matplotlib.patches as patches

from .strips import Strips


class StripsText(Strips):
    ''' Strips operations manager including segmentation of text lines. '''

    def __init__(self, dpi=300, min_height_factor=1.8, max_height_factor=5.5, max_separation_factor=1.2, **kwargs):
        ''' StripsText constructor. '''

        # Parent constructor
        super(StripsText, self).__init__(**kwargs)

        # Resolution in dpi
        self.dpi = dpi
        # Resolution in pixels / mm
        self.R = dpi / 25.44
        
        self.min_height = min_height_factor * self.R
        self.max_height = max_height_factor * self.R
        self.max_separation = max_separation_factor * self.R

        filter_blanks = False
        if 'filter_blanks' in kwargs:
            filter_blanks = kwargs['filter_blanks']

        if len(self.strips) > 0:
            self._extract_text()
            if filter_blanks:
                self.strips = [strip for strip in self.strips if len(strip.text) > 0]


    def _extract_text(self):
        ''' Extract text for each strip. '''

        for strip in self.strips:
            strip.extract_text(
                self.min_height, self.max_height, self.max_separation
            )


    def plot(self, show_boxes=True, **kwargs):
        ''' Show background image and text boxes. '''

        fig, ax, offsets = super(StripsText, self).plot(**kwargs)
        if show_boxes:
            for strip, offset in zip(self.strips, offsets):
                if strip.text:
                    boxes, _ = zip(*strip.text)
                    for box in boxes:
                        ax.add_patch(
                            patches.Rectangle(
                                (box[0] + offset, box[1]), box[2], box[3],
                                facecolor="none", edgecolor='red'
                            )
                        )

        return fig, ax, offsets
