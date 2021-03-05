import cv2
import numpy as np
from libjigsaw import solve_from_strips, solve_from_matrix
import sys
sys.path.append('../../')
from strips.strips import Strips
import matplotlib.pyplot as plt
import cPickle as pickle
#image1 = np.random.randint(0, 256, (100, 10))
#image2 = np.random.randint(0, 256, (200, 20))
margins = pickle.load(open('../../../margins_marques_2013.pkl', 'r'))
doc = 'D001'
strips = Strips(path='../../../dataset/marques_2013/%s' % doc)
left = margins[doc]['left']
right = margins[doc]['right']
strips.trim(left, right)
print(len(strips.images))

convert = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images = map(convert, strips.images)
stacked = np.hstack(images)
result = solve_from_strips(stacked, len(images))

#matrix = np.load('../../../.cache/SIBGRAPI_1/proposed/matrix/D001.npy')

#matrix
#result = solve_from_matrix(matrix[0])
print result

#print len(strips.images)

#plt.imshow(image)
#plt.show()
