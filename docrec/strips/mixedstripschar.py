import os
import re
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from skimage import transform

from .strips import Strips
from .stripschar import StripsChar
from .mixedstrips import MixedStrips

class MixedStripsChar(MixedStrips, StripsChar):
    ''' Strips operations manager.'''

    def __init__(self, strips_list, shuffle=True):
        ''' MixedStrips constructor. '''
       
        MixedStrips.__init__(self, strips_list, shuffle)
        #self.sizes = sizes
        #self.init_perm = init_perm
        #print(self.sizes)
        StripsChar.__init__(self, strips_list=self.strips)
        self._compute_matchings()
        self._compute_all_characters()
        #print(self.sizes, len(self.all), len(self.inner))
        
