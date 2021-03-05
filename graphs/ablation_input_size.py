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
fp.add_legend(title='sample size', prop={'size': 90}, labelspacing=0.1)
fp.set(yticks=(80, 85, 90, 95, 100), xticks=list(range(1, max_value + 1)), ylim=(80, 101))
fp.set_xticklabels(list(range(1, max_value + 1)), fontdict={'fontsize': 120})
fp.set_yticklabels([80, 85, 90, 95, 100], fontdict={'fontsize': 120})

datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
for ax in fp.axes.ravel():
    dataset = ax.get_title().replace('dataset = ', '')
    ax.set_title(datasets_map[dataset], fontdict={'fontsize': 120})
    ax.set_xlabel('$k$', fontsize=120)

fp.axes.flat[0].set_ylabel('accuracy (\%)', fontsize=120)
leg = fp.axes.flat[0].get_legend()
plt.setp(leg.get_title(), fontsize=100)  # legend title size

# move legend to the last axis
fp.axes.flat[1].legend_ = leg
leg.parent = fp.axes.flat[1]

# move a little bit up/right
bb = leg.get_bbox_to_anchor().transformed(fp.axes.flat[2].transAxes.inverted())
bb.y0 += 0.125
bb.y1 += 0.125
bb.x0 += 0.02
bb.x1 += 0.02
leg.set_bbox_to_anchor(bb, transform=fp.axes.flat[2].transAxes)

plt.savefig('{}/ablation_input_size.pdf'.format(path), bbox_inches='tight')