import sys
import json
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import pandas as pd

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=7)
colors = sns.color_palette('deep')

datasets = ['D1', 'D2', 'cdip']
template_fname = 'results/exp1_proposed/{}.json'
ks = {}
records = []
for dataset in datasets:
    fname = template_fname.format(dataset)
    results = json.load(open(fname, 'r'))['data']
    ks[dataset] = sorted([int(k) for k in results])
    for k in ks[dataset]:
        for run in results[str(k)]:
            accuracy = run['accuracy']
            records.append([k, dataset, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('k', 'dataset', 'accuracy'))

# new names for datasets
datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-Cdip}'}
df['dataset'].replace(datasets_map, inplace=True)
path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]

# legend class
fp = sns.FacetGrid(hue='dataset', height=17, aspect=3.5, data=df, legend_out=False)
fp = fp.map(sns.lineplot, 'k', 'accuracy', marker='s', ci=95, markersize=50)
fp.add_legend(title='dataset', prop={'size': 120}, labelspacing=0.1)
fp.set(yticks=(90, 95, 100), xticks=ks['cdip'], xlim=(0, max(ks['cdip']) + 1), ylim=(89, 101))
xlabels = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fp.ax.set_xlabel('$k$', fontsize=130)
fp.ax.set_ylabel('accuracy (\%)', fontsize=130)
fp.set_xticklabels([k if k in xlabels else None for k in ks['cdip']], fontdict={'fontsize': 130})
fp.set_yticklabels((90, 95, 100), fontdict={'fontsize': 130})

leg = fp.ax.get_legend()
plt.setp(leg.get_title(), fontsize=130)  # legend title size

plt.savefig('{}/accuracy_proposed.pdf'.format(path), bbox_inches='tight')