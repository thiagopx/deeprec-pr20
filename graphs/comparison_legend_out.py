import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib import rc

rc("text", usetex=True)
# rc('font', family='sans-serif')

import seaborn as sns

sns.set(context="paper", style="darkgrid", font_scale=7)

datasets = ["D1", "D2", "cdip"]
methods = ["proposed", "proposed-nn", "paixao", "liang", "marques"]
records = []
ks = {}
for method in methods:
    for dataset in datasets:
        if method == "liang" and dataset == "cdip":
            continue
        complement_fname = "_threads=240_use-mask=True" if method == "liang" else ""
        base_dir = "exp1_proposed" if method == "proposed" else "exp3_comparison/{}".format(method)
        full_fname = "results/{}/{}{}.json".format(base_dir, dataset, complement_fname)
        results = json.load(open(full_fname, "r"))["data"]
        ks[dataset] = sorted([int(x) for x in results.keys()])
        last = 0
        for k in ks[dataset]:
            if method == "liang" and k > 3:
                break
            for run in results[str(k)]:
                accuracy = run["accuracy"]
                document = run["docs"][0]
                records.append([k, document, dataset, method, 100 * accuracy])
            if k == ks[dataset][-1]:
                last = results[str(k)][0]["accuracy"]

        # pertubation on the last point (confidence interval plotting)
        if method != "liang":
            records.append([k, None, dataset, method, 100 * last + 0.001])
            records.append([k, None, dataset, method, 100 * last - 0.001])

records.append([1, None, "D1", "Liang", 150])  # hack to force put Liang in the legend
df = pd.DataFrame.from_records(records, columns=("k", "document", "dataset", "method", "accuracy"))

# new names for datasets
# df['dataset'].replace(datasets_map, inplace=True)

path = "graphs"
if len(sys.argv) > 1:
    path = sys.argv[1]

# size = {'S-Marques': 14, 'S-Isri-OCR': 8, 'S-cdip': 18}
# methods = ['\\textbf{Proposed}', 'Proposed-NN', 'Paixão', 'Liang', 'Marques']

fp = sns.FacetGrid(
    col="dataset",
    hue="method",
    hue_order=methods,
    data=df,
    height=17,
    aspect=1.5,
    sharex=False,
    legend_out=False,
    gridspec_kws=dict(width_ratios=[3, 2, 5]),
)
# font = font_manager.FontProperties(family='sans-serif', size=22)
# font = font_manager.FontProperties(size=30)
# font = font_manager.FontProperties(family='serif', size=22)
fp = fp.map(sns.lineplot, "k", "accuracy", marker="s", ci=95, markersize=40)
# fp = fp.add_legend(title='method', prop={'size': 100}, labelspacing=0.2, columnspacing=0.5, ncol=3)

datasets_map = {"D1": "\\textsc{S-Marques}", "D2": "\\textsc{S-Isri-OCR}", "cdip": "\\textsc{S-cdip}"}
for ax, max_val in zip(fp.axes[0], [60, 20, 100]):
    values = [1, 2, 3, 4, 5] + list(range(10, max_val + 1, 5))
    ax.set_xticks(values)  # , minor=False)
    values = [1, None, None, None, 5] + [(value if value % 10 == 0 else None) for value in range(10, max_val + 1, 5)]
    ax.set_xticklabels(values, fontdict={"fontsize": 110})
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([0, 25, 50, 75, 100], fontdict={"fontsize": 110})
    dataset = ax.get_title().replace("dataset = ", "")
    ax.set_title(datasets_map[dataset], fontdict={"fontsize": 110})
    ax.set_xlabel("$k$", fontsize=110)

fp.axes.flat[0].set_ylabel("accuracy (\%)", fontsize=110)
fp.set(ylim=(-5, 101))

# change legend labels
methods_map = {
    "proposed": "\\textbf{Deeprec-CL}",
    "proposed-nn": "Deeprec-CL-NN",
    "paixao": "Paixão",
    "marques": "Marques",
    "liang": "Liang",
}
# leg = fp.axes.flat[0].get_legend()

first_ax = fp.axes.flat[0]
leg = first_ax.legend(
    title="method",
    loc="center left",
    bbox_to_anchor=(1.0, -0.5),
    fontsize=100,
    title_fontsize=110,
    labelspacing=0.2,
    columnspacing=0.5,
    ncol=3,
)
for text in leg.get_texts():
    text.set_text(methods_map[text.get_text()])
# plt.setp(leg.get_title(), fontsize=110)  # legend title size

plt.savefig("{}/comparison_legend_out.pdf".format(path), bbox_inches="tight")
