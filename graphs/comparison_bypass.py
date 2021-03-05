import sys
import json
from itertools import product
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib import rc
rc('text', usetex=True)

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=6.75)
# sns.set_style({'font.family': ['sans-serif'], 'sans-serif': ['Arial']})

datasets = ['D1', 'D2', 'cdip']
records = []
for dataset in datasets:
    fname = 'ablation/results/{}_0.2_1000_32x32_10.json'.format(dataset)
    results = json.load(open(fname, 'r'))['data']
    max_value = len(results)
    for k in range(1, max_value + 1):
        for run in results[str(k)]:
            accuracy = run['accuracy']
            records.append(['Vanilla SqueezeNet', k, dataset, 100 * accuracy])

    fname = 'ablation/results/{}_0.2_1000_32x32_10_sn-bypass.json'.format(dataset)
    results = json.load(open(fname, 'r'))['data']
    max_value = len(results)
    for k in range(1, max_value + 1):
        for run in results[str(k)]:
            accuracy = run['accuracy']
            records.append(['SqueezeNet + SB', k, dataset, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('architecture', '$k$', 'dataset', 'Accuracy (\\%)'))

# new names for datasets
datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
df['dataset'].replace(datasets_map, inplace=True)

# df.sort_values(by='size', inplace=True)
path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]

max_value = len(df['$k$'].unique())

fp = sns.FacetGrid(
    col='dataset',
    hue='architecture',
    data=df,
    col_order=[datasets_map[dset] for dset in ['D1', 'D2', 'cdip']],
    height=12, aspect=1.25#, legend=False
    )
# font = font_manager.FontProperties(family='sans-serif', size=20)
fp = fp.map(sns.lineplot, '$k$', 'Accuracy (\\%)', marker='s', ci=None, markersize=25) #.add_legend(
    # title='Architecture', prop={'size': 50}, labelspacing=0.25))
fp.set(yticks=(25, 50, 75, 100), xticks=list(range(1, max_value + 1)), ylim=(25, 101))

# trick to change the fonts
fp.set_xticklabels(list(range(1, max_value + 1)), fontdict={'fontsize': 70})
fp.set_yticklabels([25, 50, 75, 100], fontdict={'fontsize': 70})
for ax in fp.axes.ravel():
    ax.set_title(ax.get_title().replace('dataset = ', ''))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='architecture')
fp.fig.tight_layout(w_pad=0.52)
plt.savefig('{}/comparison_bypass.pdf'.format(path), bbox_inches='tight')