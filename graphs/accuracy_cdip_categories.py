import sys
import json
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import pandas as pd

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=7)
colors = sns.color_palette('deep')

annotation = json.load(open('datasets/D3/mechanical/annotation.json'))
category_map = {doc: annotation[doc]['category'] for doc in annotation}

fname = 'results/exp2_ablation/cdip_0.2_1000_32x32_10.json'
results = json.load(open(fname, 'r'))['data']['1']
records = []
for run in results:
    doc = run['docs'][0].split('/')[-1]
    accuracy = run['accuracy']
    category = category_map[doc]
    records.append([doc, category, 100 * accuracy])

df = pd.DataFrame.from_records(records, columns=('doc', 'category', 'accuracy'))
df['category'].replace({'news_article': 'news article'}, inplace=True)

path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]

fp = sns.catplot(x='category', y='accuracy', data=df, kind='box', height=17, aspect=3, linewidth=9, width=0.6, fliersize=10)
fp.set_xticklabels(rotation=30, fontdict={'fontsize': 120})
fp.set(yticks=(40, 60, 80, 100), ylim=(40, 101))
fp.set_yticklabels((40, 60, 80, 100), fontdict={'fontsize': 120})
fp.ax.set_xlabel('category', fontsize=140)
fp.ax.set_ylabel('accuracy (\%)', fontsize=140)

plt.savefig('{}/accuracy_cdip_categories.pdf'.format(path), bbox_inches='tight')
