import os
import json
import shutil
from texttable import Texttable
import numpy as np
import cv2
import glob
from itertools import product

import pandas as pd
from docrec.metrics.solution import neighbor_comparison
from docrec.strips.strips import Strips
from docrec.strips.mixedstrips import MixedStrips

annotation = json.load(open('datasets/D3/mechanical/annotation.json'))
categories = set([info['category'] for info in annotation.values()])
datasets = ['D1', 'D2', 'cdip']
template_fname = 'ablation/1-neutral/{}-{}.json'
records = []
count_roll = {}
for dataset, rneu in product(datasets, [0.0, 0.025, 0.05]):
    fname = template_fname.format(dataset, rneu)
    results = json.load(open(fname, 'r'))['data']['1'] # k = 1 (number of documents)
    for run in results:
        doc = run['docs'][0].split('/')[-1]
        category = annotation[doc]['category'] if dataset == 'cdip' else ''
        accuracy = run['accuracy']
        # roll checking
        k = 0
        solution = [run['init_perm'][s] for s in run['solution']]
        while k < len(solution) - 1:
            if (solution[k] == len(solution) - 1) and (solution[k + 1] == 0):
                try:
                    count_roll[(dataset, rneu)] += 1
                except KeyError:
                    count_roll[(dataset, rneu)] = 1
                break
            k += 1

        records.append([doc, dataset, category, 100 * rneu, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('doc', 'dataset', 'category', 'rneu', 'accuracy'))

# rolls (spice the last strip with the first one)
print('count roll')
table = Texttable(max_width=0)
table.set_deco(Texttable.HEADER)
table.set_cols_dtype(['t', 'i', 'i', 'i'])
table.set_cols_align(['l', 'r', 'r', 'r'])
table.add_rows([['dataset', '0.0', '2.5', '5.0']])
for dataset in datasets:
    row = [dataset] + [count_roll[(dataset, rneu)] for rneu in [0.0, 0.025, 0.05]]
    table.add_row(row)
print(table.draw())

for dataset in datasets:
    print(dataset)
    # overall
    print('overall')
    df_ = df[df['dataset'] == dataset]
    df_glob = df_[['doc', 'category']].drop_duplicates()
    df_glob['0.0'] = df_[df_['rneu'] == 0.0]['accuracy'].values
    df_glob['2.5'] = df_[df_['rneu'] == 2.5]['accuracy'].values
    df_glob['5.0'] = df_[df_['rneu'] == 5.0]['accuracy'].values
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_glob)

    # ordered by max diff (5.0 - 0.0)
    print('ordered')
    df_glob['diff'] = df_glob['5.0'] - df_glob['0.0']
    df_sorted = df_glob.sort_values(by='diff', ascending=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_sorted)

    # winner
    print('winner')
    N = len(df_glob)
    table = Texttable(max_width=0)
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['i', 'i', 'i'])
    table.set_cols_align(['r', 'r', 'r'])
    table.add_rows([['wins', 'draw', 'looses']])
    wins = (df_glob['diff'] > 0).sum()
    draws = (df_glob['diff'] == 0).sum()
    looses = (df_glob['diff'] < 0).sum()
    table.add_row([wins, draws, looses])
    print(table.draw())
    
    print('descending')
    print(df_glob[(df_glob['0.0'] > df_glob['2.5']) & (df_glob['2.5'] >= df_glob['5.0'])])
    
    print('ascending')
    print(df_glob[(df_glob['0.0'] < df_glob['2.5']) & (df_glob['2.5'] <= df_glob['5.0'])])
#df_result['']

# for rneu in [0.0, 0.025, 0.05]]

# for category, df_cat in df[df['dataset'] == 'cdip'].groupby(['category']):
#     rneu_map = {}
#     for rneu, df_rneu in df_cat.groupby(['rneu']):
#         rneu_map[rneu] = df_rneu['accuracy'].values.tolist()
#         docs = df_rneu['doc'].values.tolist()
#     for doc, v1, v3 in zip(docs, rneu_map[0.0], rneu_map[2.5], rneu_map[5.0]):
#         row = [doc, category, v1, v2, v3]
#         table.add_row(row)
# print(table.draw())

# # new names for datasets
# #datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
# #df['dataset'].replace(datasets_map, inplace=True)

# #df = df.sort_values(by=['dataset', 'doc'])
# #X = df[df['dataset'] == 'cdip'].groupby(['category', 'rneu']).mean()
# #print(df[df['dataset'] == 'cdip'].groupby(['category', 'rneu']).mean())
# #for reg in df[df['dataset'] == 'cdip'].groupby(['category', 'rneu']).mean():

# # category x rneu
# table = Texttable(max_width=0)
# table.set_deco(Texttable.HEADER)
# table.set_cols_dtype(['t', 'f', 'f', 'f'])
# table.set_cols_align(['l', 'r', 'r', 'r'])
# table.add_rows([['cat.', '0.0', '2.5', '5.0']])
# for category, df_cat in df[df['dataset'] == 'cdip'].groupby(['category']):
#     rneu_map = {}
#     for rneu, df_rneu in df_cat.groupby(['rneu']):
#         rneu_map[rneu] = df_rneu['accuracy'].mean()
#     row = [category, rneu_map[0.0], rneu_map[2.5], rneu_map[5.0]]
#     table.add_row(row)
# print(table.draw())

# print()

# doc x category x rneu
# table = Texttable(max_width=0)
# table.set_deco(Texttable.HEADER)
# table.set_cols_dtype(['t', 't', 'f', 'f', 'f'])
# table.set_cols_align(['l', 'l', 'r', 'r', 'r'])
# table.add_rows([['doc.', 'cat.', '0.0', '2.5', '5.0']])
# for category, df_cat in df[df['dataset'] == 'cdip'].groupby(['category']):
#     rneu_map = {}
#     for rneu, df_rneu in df_cat.groupby(['rneu']):
#         rneu_map[rneu] = df_rneu['accuracy'].values.tolist()
#         docs = df_rneu['doc'].values.tolist()
#     for doc, v1, v2, v3 in zip(docs, rneu_map[0.0], rneu_map[2.5], rneu_map[5.0]):
#         row = [doc, category, v1, v2, v3]
#         table.add_row(row)
# print(table.draw())

# #     print()


# for reg in df.groupby(['doc', 'dataset']):

# print(categories, annotation)
# print('================== Single experiment ==================')
# for dataset, arch in product(datasets, archs):
#     print('training dataset={} thresholding method={}'.format(dataset, thresh))
#     results = json.load(open('results/proposed-single_{}-{}-{}.json'.format(dataset, arch, thresh)))
#     if dataset == 'isri-ocr':
#         table = Texttable()
#         table.set_deco(Texttable.HEADER)
#         table.set_cols_dtype(['t', 'f', 'f', 'f'])
#         table.set_cols_align(['l', 'r', 'r', 'r'])
#         table.add_rows([['category', 'avg.', 'min.', 'max.']])
#         accuracies = {category: [] for category in categories}
#         for doc in results:
#             doc_id = os.path.basename(doc)
#             category = annotation[doc_id]['category']
#             accuracies[category].append(results[doc]['accuracy'])

#         for category in accuracies:
#             values = np.array(accuracies[category])
#             table.add_row([category, values.mean(), values.min(), values.max()])
#         #print(accuracies)
#         print(table.draw())
#         print()


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