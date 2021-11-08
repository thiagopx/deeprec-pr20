import sys
import json
from itertools import product
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='Arial')
# # plt.rc('font', seri='Times')
from matplotlib import rc
rc('text', usetex=True)

import matplotlib.font_manager as font_manager
import pandas as pd

from docrec.metrics.solution import neighbor_comparison

import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=5)
# sns.set_style({'font.family': ['sans-serif'], 'sans-serif': ['Arial']})
colors = sns.color_palette('deep')

#annotation = json.load(open('datasets/D2/mechanical/annotation.json'))
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

fname = 'exp2_ablation/results/D2_0.2_1000_32x32_10.json'
results = json.load(open(fname, 'r'))['data']['1']
records = []
for run in results:
    doc = run['docs'][0].split('/')[-1]
    print(doc)
    accuracy = run['accuracy']
    category = map_doc_category[doc]
    records.append([doc, category, 100 * accuracy])
df = pd.DataFrame.from_records(records, columns=('doc', 'Category', 'Accuracy (\\%)'))

#map_category = {
#    'news_article': 'news article'
#}
#df['Category'].replace(map_category, inplace=True)
# # new names for datasets
# # datasets_map = {'D1': '\\textsc{S-Marques}', 'D2': '\\textsc{S-Isri-OCR}', 'cdip': '\\textsc{S-cdip}'}
# # df['dataset'].replace(datasets_map, inplace=True)
path = 'graphs'
if len(sys.argv) > 1:
    path = sys.argv[1]
# # max_value = len(df['$k$'].unique())

g = sns.catplot(x='Category', y='Accuracy (\\%)', data=df, kind='box', height=8, aspect=1, linewidth=4, width=0.6, fliersize=10)#kwargs={'width')
g.set_xticklabels(rotation=30, fontdict={'fontsize': 50})
g.set(yticks=(40, 60, 80, 100), ylim=(40, 101))
g.set_yticklabels((40, 60, 80, 100), fontdict={'fontsize': 45})
plt.savefig('{}/accuracy_isri_categories.pdf'.format(path), bbox_inches='tight')
# plt.show()
