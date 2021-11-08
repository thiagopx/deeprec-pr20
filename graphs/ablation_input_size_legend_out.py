import sys
import json
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib import rc
rc('text', usetex=True)

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=7)
# sns.set_style({'font.family': ['sans-serif'], 'sans-serif': ['Arial']})

datasets = ['D1', 'D2', 'cdip']
template_fname = 'exp2_ablation/results/{}_0.2_1000_{}x{}_10.json'
records = []
sizes = [32, 64]
for dataset, H, W in product(datasets, sizes, sizes):
    fname = template_fname.format(dataset, H, W)
    results = json.load(open(fname, 'r'))['data']
    max_value = len(results)
    for k in range(1, max_value + 1):
        for run in results[str(k)]:
            accuracy = run['accuracy']
            records.append([k, dataset, H, W, '{} $\\times$ {}'.format(H, W), 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('k', 'dataset', 'H', 'W', 'size', 'accuracy'))
df.sort_values(by='size', inplace=True)

path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]

max_value = 5
fp = sns.FacetGrid(col='dataset', hue='size', data=df, height=17, col_order=datasets, aspect=1, legend_out=False)
fp = fp.map(sns.lineplot, 'k', 'accuracy', marker='s', ci=None, markersize=40)
fp.set(yticks=(80, 85, 90, 95, 100), xticks=list(range(1, max_value + 1)), ylim=(80, 101))
fp.set_xticklabels(list(range(1, max_value + 1)), fontdict={'fontsize': 120})
fp.set_yticklabels([80, 85, 90, 95, 100], fontdict={'fontsize': 120})

datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
for ax in fp.axes.ravel():
    dataset = ax.get_title().replace('dataset = ', '')
    ax.set_title(datasets_map[dataset], fontdict={'fontsize': 120})
    ax.set_xlabel('$k$', fontsize=120)

fp.axes.flat[0].set_ylabel('accuracy (\%)', fontsize=120)

last_ax = fp.axes.flat[-1]
leg = last_ax.legend(title='sample size', loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=90, title_fontsize=100, labelspacing=0.1)

plt.savefig('{}/ablation_input_size_legend_out.pdf'.format(path), bbox_inches='tight')
