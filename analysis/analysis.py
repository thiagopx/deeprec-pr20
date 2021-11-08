import os
import json
import shutil
from texttable import Texttable
import numpy as np
import cv2
import glob
from itertools import product

from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips

# experiments = glob.glob('results/*.json')

archs = ['sn'] #, 'mn']
threshs = ['sauvola', 'otsu']
datasets = ['isri-ocr']#, 'cdip']
annotation = json.load(open('datasets/D3/mechanical/annotation.json'))
categories = set([info['category'] for info in annotation.values()])
print(categories)
print('================== Single experiment ==================')
for dataset, arch, thresh in product(datasets, archs, threshs):
    print('training dataset={} thresholding method={}'.format(dataset, thresh))
    results = json.load(open('results/proposed-single_{}-{}-{}.json'.format(dataset, arch, thresh)))
    if dataset == 'isri-ocr':
        table = Texttable()
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(['t', 'f', 'f', 'f'])
        table.set_cols_align(['l', 'r', 'r', 'r'])
        table.add_rows([['category', 'avg.', 'min.', 'max.']])
        accuracies = {category: [] for category in categories}
        for doc in results:
            doc_id = os.path.basename(doc)
            category = annotation[doc_id]['category']
            accuracies[category].append(results[doc]['accuracy'])

        for category in accuracies:
            values = np.array(accuracies[category])
            table.add_row([category, values.mean(), values.min(), values.max()])
        #print(accuracies)
        print(table.draw())
        print()


# print('global')
# print('{:28} max {:5}min {:5}avg'.format('', '', '', ''))
# for experiment in experiments:
#     results = json.load(open(experiment))
#     acc = np.array([results[doc]['accuracy'] for doc in results])
#     print('{:24}: {:8.3f} {:8.3f} {:8.3f}'.format(os.path.basename(experiment), acc.max(), acc.min(), acc.mean()))

# for dataset in ['D1', 'D2']:
#     print()
#     print(dataset)
#     print('{:28} max {:5}min {:5}avg'.format('', '', '', ''))
#     for experiment in experiments:
#         results = json.load(open(experiment))
#         acc = np.array([results[doc]['accuracy'] for doc in results if doc.split('/')[1] == dataset])
#         print('{:24}: {:8.3f} {:8.3f} {:8.3f}'.format(os.path.basename(experiment), acc.max(), acc.min(), acc.mean()))


'''
import sys
sys.exit()
print()
print('Neutral samples effect')
path = 'preliminar-results/{}/results-rec-proposed-sn.json'.format('3-1000-bin')
path_neu = 'preliminar-results/{}/results-rec-proposed-sn.json'.format('3-1000-neu-bin')
results = json.load(open(path))
results_neu = json.load(open(path_neu))
docs = sorted([doc for doc in results])
print('         bin     bin-neu')
for doc in docs:
    print('{}: '.format(os.path.basename(doc)), end='')
    print('{:7.2f} {:7.2f}'.format(results[doc]['accuracy'], results_neu[doc]['accuracy']))
    for case, results_doc in zip(['bin', 'bin-neu'], [results[doc], results_neu[doc]]):
        solution = results_doc['solution']
        order = [results_doc['init_perm'][x] for x in solution]
        displacements = [results_doc['displacements'][i][j] for i, j in zip(solution[: -1], solution[1 :])]
        strips = MixedStrips([Strips(path=doc, filter_blanks=True)], shuffle=False)
        image = strips.image(order=order, displacements=displacements, filled=True, verbose=True)
        cv2.imwrite('preliminar-analysis/{}/{}.jpg'.format(case, os.path.basename(doc)), image)



# dataset 1 (category info)
categories = json.load(open('categories_D1.json', 'r'))
doc_category_map = {}
for category, docs in categories.items():
    for doc in docs:
        doc_category_map[doc] = category.upper()
print('#documents in D1 (per categories): to = {}, lg = {} fg = {}'.format(
    len(categories['to']), len(categories['lg']), len(categories['fg'])
))

# data to be analyzed
results = json.load(open('results/results_proposed.json', 'r'))
for doc in results:
    _, dataset, _, doc_ = doc.split('/')
    solution = results[doc]['solution']
    comp = results[doc]['compatibilities']
    if dataset == 'D2':
        print(comp)
        continue
    acc = accuracy(solution)
    print('{}-{} {:.2f}% {}'.format(dataset, doc_, 100 * acc, doc_category_map[doc_] if dataset == 'D1' else ''))
    if acc < 0.8:
        if dataset == 'D1':
            src1 = '{}/{}.jpg'.format(doc, doc_)
        else:
            src1 = 'datasets/D2/integral/{}.TIF'.format(doc_)
        shutil.copy(src1, 'ignore/{}-{}'.format(dataset, os.path.basename(src1)))

        if os.path.exists('ignore/{}-{}'.format(dataset, doc_)):
            shutil.rmtree('ignore/{}-{}'.format(dataset, doc_))
        shutil.copytree(doc, 'ignore/{}-{}'.format(dataset, doc_))
'''
