import numpy as np
import random

from .strips import Strips


class MixedStrips(Strips):

    def __init__(self, strips_list, shuffle=True):
        ''' MixedStrips constructor. '''

        strips = []
        sizes = []
        for strips_ in strips_list:
            strips += strips_.strips
            sizes.append(len(strips_.strips))

        init_perm = list(range(len(strips)))
        if shuffle:
            random.shuffle(init_perm)
            strips = [strips[i] for i in init_perm]

        Strips.__init__(self, strips_list=strips, filter_blanks=False)
        self.sizes = sizes
        self.init_perm = init_perm
