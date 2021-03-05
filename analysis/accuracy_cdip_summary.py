import sys
import os
import cv2
import json
import shutil
from itertools import product
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rc
from docrec.strips.strips import Strips
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import pandas as pd
import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=3)
colors = sns.color_palette('deep')

bboxes = json.load(open('datasets/D3/mechanical/bboxes.json'))
annotation = json.load(open('datasets/D3/mechanical/annotation.json'))
map_doc_category = {doc: annotation[doc]['category'] for doc in annotation}

total = 0
perfect = 0
under_80 = 0
under_70 = 0
fname = 'ablation/results/cdip_0.2_1000_32x32_10.json'
results = json.load(open(fname, 'r'))['data']['1']
records = []
for run in results:
    total += 1
    doc = run['docs'][0].split('/')[-1]
    # print(doc)

    category = map_doc_category[doc]
    accuracy = run['accuracy']
        
    if accuracy == 1.0:
        perfect += 1
    if accuracy < 0.8:
        under_80 += 1
    if accuracy < 0.7:
        under_70 += 1

print('{}/{} are perfect reconstructions'.format(perfect, total))
print('{}/{} are under 80\% of accuracy'.format(under_80, total))
print('{}/{} are under 70\% of accuracy'.format(under_70, total))

for run in results:
    total += 1
    doc = run['docs'][0].split('/')[-1]
    # print(doc)

    category = map_doc_category[doc]
    accuracy = run['accuracy']
    if category == 'letter':# and accuracy <= 0.8:
        print(doc, accuracy)
    

    # path = 'analysis/accuracy_cdip/integral/{}'.format(category)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # shutil.copy('datasets/{}'.format(doc2image[doc]), '{}/{}_{}.tif'.format(path, accuracy, doc))
    # # records.append([doc, 100 * map_doc_textarea[doc], 100 * accuracy])
    
    # strips = Strips(path='datasets/D3/mechanical/{}'.format(doc), filter_blanks=True)
    # image = strips.image()[..., :: -1]
    # path = 'analysis/accuracy_cdip/shredded_gt/{}'.format(category)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # cv2.imwrite('{}/{}_{}.jpg'.format(path, accuracy, doc), image)
    
    # solution = [run['init_perm'][s] for s in run['solution']]
    # displacements = [run['displacements'][solution[i]][solution[i + 1]] for i in range(len(solution) - 1)]
    # image = strips.image(solution, displacements, False)[..., :: -1] # RGB to BGR
    # path = 'analysis/accuracy_cdip/shredded_rec/{}'.format(category)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # cv2.imwrite('{}/{}_{}.jpg'.format(path, accuracy, doc), image)

# df = pd.DataFrame.from_records(records, columns=('doc', 'text area (\\%)', 'Accuracy (\\%)'))
# df.sort_values(by='Accuracy (\\%)', inplace=True)    
# print(df[['doc', 'Accuracy (\\%)']])

# for index, row in df.iterrows():
#     doc = row['doc']
#     accuracy = row['Accuracy (\\%)']
#     shutil.copy('datasets/{}'.format(doc2image[doc]), 'analysis/accuracy_cdip/integral/{}_{}.tif'.format(accuracy, doc))
    # strips = Strips(path='datasets/D3/mechanical/{}'.format(doc), filter_blanks=True)
    
    # image = strips.image()[..., :: -1]
    # cv2.imwrite('analysis/accuracy_cdip/shredded/{}_{}.tif'.format(accuracy, doc), image)
    # solution = [run['init_perm'][s] for s in run['solution']]
    #     displacements = [run['displacements'][solution[i]][solution[i + 1]] for i in range(len(solution) - 1)]
    #     strips = Strips(path=doc, filter_blanks=True)
    #     reconstruction = strips.image(solution, displacements, True)[:, :, ::-1] # RGB to BGR

    
    
# path = 'graphs'
# if len(sys.argv) > 1:
#     path = sys.argv[1]

# fig, ax = plt.subplots(figsize=(9, 9))
# sns.scatterplot(
#     x='text area (\\%)', y='Accuracy (\\%)',
#     # palette="ch:r=-.2,d=.3_r",
#     # hue_order=clarity_ranking,
#     sizes=(1, 8), linewidth=0,
#     data=df, ax=ax
# )
# # legend class
# # https://matplotlib.org/_modules/matplotlib/legend.html#Legend
# max_value = len(df['$k$'].unique())
# #fig, ax = plt.subplots(figsize=(3, 9))
# # hue_order=['on', 'off'],
# fp = sns.FacetGrid(col='dataset', hue='neutral_thresh', height=6, aspect=1.0, data=df, legend_out=True)
# rc('font',**{'family':'serif','serif':['Times']})
# font = font_manager.FontProperties(family='serif', size=20)
# fp = (fp.map(sns.lineplot, '$k$', 'Accuracy (\\%)', marker='s', markersize=13).add_legend(title='$\\rho_{neutral}$', prop=font, labelspacing=0.25))
# #plt.xlim(1, max_value)
# #plt.ylim(60, 101)
# #fig.tight_layout()
# #fp.set_ytickslabels(list(range(60, 101, 10)))
# fp.set(yticks=(85, 90, 95, 100), xticks=list(range(1, max_value + 1)), ylim=(85, 101))
# #plt.setp(fp.axes.ravel()[0].get_legend().get_frame(), labelspacing=2) # for legend text
# #plt.setp(fp.axes.ravel()[0].get_legend().get_texts(), fontsize='8') # for legend text
# #plt.setp(fp.axes.ravel()[0].get_legend().get_title(), fontsize='x-small') # for legend text
# #fp.fig.tight_layout(w_pad=1)
# plt.savefig('{}/others/accuracy_by_text.pdf'.format(path), bbox_inches='tight')
# # plt.show()
# #fp.axes.clear()'