import sys
import json
from itertools import product
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
rc('text', usetex=True)
# rc('font',**{'family':'serif','serif':['Times']})
import pandas as pd

from docrec.metrics.solution import neighbor_comparison

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=7)
colors = sns.color_palette('deep')

datasets = ['D1', 'D2', 'cdip']
template_fname = 'exp2_ablation/results/{}_0.2_1000_32x32_{}.json'
records = []
for dataset, vshift in product(datasets, [0, 5, 10, 20]):
    fname = template_fname.format(dataset, vshift)
    results = json.load(open(fname, 'r'))['data']
    max_value = len(results)
    values = [1] + list(range(5, max_value + 1, 5))
    for k in list(range(1, max_value + 1)):
        for run in results[str(k)]:
            accuracy = run['accuracy']
            records.append([k, dataset, vshift, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('k', 'dataset', 'vshift', 'accuracy'))

path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]

max_value = 5
fp = sns.FacetGrid(col='dataset', hue='vshift', data=df, height=17, aspect=1.0, legend_out=False)
fp = fp.map(sns.lineplot, 'k', 'accuracy', marker='s', ci=None, markersize=40)
fp.set(yticks=(60, 70, 80, 90, 100), xticks=list(range(1, max_value + 1)), ylim=(60, 101))
fp.set_xticklabels(list(range(1, max_value + 1)), fontdict={'fontsize': 120})
fp.set_yticklabels([60, 70, 80, 90, 100], fontdict={'fontsize': 120})

datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
for ax in fp.axes.ravel():
    dataset = ax.get_title().replace('dataset = ', '')
    ax.set_title(datasets_map[dataset], fontdict={'fontsize': 120})
    ax.set_xlabel('$k$', fontsize=120)

fp.axes.flat[0].set_ylabel('accuracy (\%)', fontsize=120)

last_ax = fp.axes.flat[-1]
leg = last_ax.legend(title='$v_{shift}$', loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=90, title_fontsize=100, labelspacing=0.1)

plt.savefig('{}/ablation_vshift_legend_out.pdf'.format(path), bbox_inches='tight')
