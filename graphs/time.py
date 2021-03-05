import sys
import json
from itertools import product
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib import rc
rc('text', usetex=True)
# rc('font', family='sans-serif')

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=7)
# sns.set_style({'font.family': ['sans-serif'], 'sans-serif': ['Arial']})

datasets = ['D1', 'D2']#, cdip']
# datasets_map = {'D1': 'S-Marques', 'D2': 'S-Isri-OCR', 'cdip': 'S-cdip'}


# proposed
# matrix_data = json.load(open('comparison/proposed/D2_matrix.json', 'r'))
# N = sum(matrix_data['sizes'])
# total_inferences = N * (N - 1)
# mean_inference_time = matrix_data['comp_time'] / total_inferences # pair evaluation mean inf. time

records = []
for dataset in datasets:
    fname_mat = 'exp1_proposed/results/{}_matrix.json'.format(dataset)
    fname_res = 'exp1_proposed/results/{}.json'.format(dataset)
    matrix_data = json.load(open(fname_mat, 'r'))
    N = sum(matrix_data['sizes'])
    total_inferences = N * (N - 1)
    mean_inference_time = matrix_data['comp_time'] / total_inferences # pair evaluation mean time
    results = json.load(open(fname_res, 'r'))['data']
    for k in range(1, 6):
        for run in results[str(k)]:
            N = sum(run['sizes'])
            total_inferences = N * (N - 1)
            comp_time = total_inferences * mean_inference_time
            total_time = comp_time + run['opt_time']
            records.append([k, 'proposed', total_time])

# marques
for dataset in datasets:
    fname_mat = 'exp3_comparison/marques/results/{}_matrix.json'.format(dataset)
    fname_res = 'exp3_comparison/marques/results/{}.json'.format(dataset)
    matrix_data = json.load(open(fname_mat, 'r'))
    N = sum(matrix_data['sizes'])
    total_inferences = N * (N - 1)
    mean_inference_time = matrix_data['comp_time'] / total_inferences # pair evaluation mean time
    results = json.load(open(fname_res, 'r'))['data']
    for k in range(1, 6):
        for run in results[str(k)]:
            N = sum(run['sizes'])
            total_inferences = N * (N - 1)
            comp_time = total_inferences * mean_inference_time
            total_time = comp_time + run['opt_time']
            records.append([k, 'marques', total_time])

# paixao
for dataset in datasets:
    fname = 'exp3_comparison/paixao/results/{}.json'.format(dataset)
    results = json.load(open(fname, 'r'))['data']
    for k in range(1, 6):
        for run in results[str(k)]:
            total_time = run['comp_time'] + run['opt_time']
            records.append([k, 'paixao', total_time])

# liang
for dataset in datasets:
    fname = 'exp3_comparison/liang/results/{}_threads=240_use-mask=True.json'.format(dataset)
    results = json.load(open(fname, 'r'))['data']
    # k_max = 5 if dataset == 'D1' else 3
    k_max = 3
    for k in range(1, k_max + 1):
        for run in results[str(k)]:
            if run['solution'] != None:
                total_time = run['opt_time']
                records.append([k, 'liang', total_time])

df = pd.DataFrame.from_records(records, columns=['k', 'method', 'time'])

methods = ['proposed', 'paixao', 'marques', 'liang']
fp = sns.FacetGrid(
    hue='method', hue_order=methods, data=df,
    height=17, aspect=2, legend_out=False, sharey=False
)
fp = fp.map(sns.lineplot, 'k', 'time', marker='s', ci=None, markersize=40)
fp.add_legend(title='method', fontsize=100, labelspacing=0.2) # bug in title_fontsize
fp.set(yticks=(0, 2000, 4000, 6000), xticks=list(range(1, 6)), ylim=(-200, 6200))
fp.ax.set_xlabel('$k$', fontdict={'fontsize': 110})
fp.ax.set_ylabel('time (s)', fontdict={'fontsize': 110})
fp.set_yticklabels((0, 2000, 4000, 6000), fontdict={'fontsize': 110})
fp.set_xticklabels(list(range(1, 6)), fontdict={'fontsize': 110})

# figure 1 - all methods
# legend adjustment
methods_map = {'proposed': '\\textbf{Proposed}', 'paixao': 'Paixão', 'marques': 'Marques', 'liang': 'Liang'}
leg = fp.ax.get_legend()

for text in leg.get_texts():
    text.set_text(methods_map[text.get_text()])
plt.setp(leg.get_title(), fontsize=110)  # legend title size

bb = leg.get_bbox_to_anchor().inverse_transformed(fp.ax.transAxes)
bb.y0 += 0.035
bb.y1 += 0.035
bb.x0 -= 0.015
bb.x1 -= 0.015
leg.set_bbox_to_anchor(bb, transform=fp.ax.transAxes)

path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]
plt.savefig('{}/time1.pdf'.format(path), bbox_inches='tight')

# figure 2 - only proposed and marques
df = df[df['method'].isin(['marques', 'proposed'])]
fp = sns.FacetGrid(
    hue='method', hue_order=methods, data=df,
    height=17, aspect=1, legend_out=False, sharey=False
)
fp = fp.map(sns.lineplot, 'k', 'time', marker='s', ci=None, markersize=40)
fp.set(yticks=(0, 100, 200, 300), xticks=list(range(1, 6)), ylim=(-10, 360))
fp.ax.set_xlabel('$k$', fontdict={'fontsize': 110})
fp.ax.set_ylabel('time (s)', fontdict={'fontsize': 110})
fp.set_yticklabels((0, 100, 200, 300), fontdict={'fontsize': 110})
fp.set_xticklabels(list(range(1, 6)), fontdict={'fontsize': 110})
plt.savefig('{}/time2.pdf'.format(path), bbox_inches='tight')
