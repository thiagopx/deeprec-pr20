import sys
import os
import cv2
import json
import shutil
from itertools import product
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
from docrec.strips.strips import Strips
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import pandas as pd
import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=3)
colors = sns.color_palette('deep')

#bboxes = json.load(open('datasets/D3/mechanical/bboxes.json'))
#annotation = json.load(open('datasets/D3/mechanical/annotation.json'))
map_doc_category = {
    'D001': 'Legal',
    'D002': 'Business',
    'D003': 'Business',
    'D004': 'Business',
    'D005': 'Legal',
    'D006': 'Business',
    'D007': 'Business',
    'D008': 'Legal',
    'D009': 'Legal',
    'D010': 'Legal',
    'D011': 'Legal',
    'D012': 'Legal',
    'D013': 'Business',
    'D014': 'Legal',
    'D015': 'Business',
    'D016': 'Business',
    'D017': 'Business',
    'D018': 'Legal',
    'D019': 'Business',
    'D020': 'Legal'
}

total = 0
perfect = 0
under_80 = 0
under_70 = 0
fname = 'ablation/results/D2_0.2_1000_32x32_10.json'
results = json.load(open(fname, 'r'))['data']['1']
records = []
for run in results:
    total += 1
    doc = run['docs'][0].split('/')[-1]
    # print(doc)

    accuracy = run['accuracy']
        
    if accuracy == 1.0:
        perfect += 1
    if accuracy < 0.8:
        under_80 += 1
    if accuracy < 0.7:
        under_70 += 1

print('{}/{} are perfect reconstructions'.format(perfect, total))
print('{}/{} are under 80\% of accuracy'.format(under_80, total))
print('{}/{} are under 70\% of accuracy'.format(under_70, total))

category = {}
for run in results:
    total += 1
    doc = run['docs'][0].split('/')[-1]
    # print(doc)

    category = map_doc_category[doc]
    accuracy = run['accuracy']
    print(category, accuracy)
    #if category == 'letter':# and accuracy <= 0.8:
    #    print(doc, accuracy)