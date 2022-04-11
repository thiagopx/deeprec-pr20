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

palette = sns.color_palette()[:5]

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

# change legend labels
methods_map = {
    "proposed": "\\textbf{Deeprec-CL}",
    "proposed-nn": "Deeprec-CL-NN",
    "paixao": "Paixão",
    "marques": "Marques",
    "liang": "Liang",
}

# S-MARQUES
fig, ax = plt.subplots(figsize=(54, 18))
fp = sns.lineplot(
    x="k",
    y="accuracy",
    hue="method",
    hue_order=methods,
    data=df[df["dataset"] == "D1"],
    # legend_out=False,
    legend=False,
    ci=95,
    markersize=50,
    marker="s",
    # palette=palette,
    ax=ax,
)
max_val = 60
values = [1, 2, 3, 4, 5] + list(range(10, max_val + 1, 5))
fp.set_xticks(values)  # , minor=False)
values = [1, None, None, None, 5] + [(value if value % 10 == 0 else None) for value in range(10, max_val + 1, 5)]
fp.set_xticklabels(values, fontdict={"fontsize": 120})
fp.set_yticks([0, 25, 50, 75, 100])
fp.set_yticklabels([0, 25, 50, 75, 100], fontdict={"fontsize": 120})
fp.set_title("\\textsc{S-Marques}", fontdict={"fontsize": 120})
fp.set_xlabel("$k$", fontsize=120)
fp.set_ylabel("accuracy (\%)", fontsize=120)
fp.set(xlim=(0, 61))
fp.set(ylim=(-1, 101))
handles = ax.lines
plt.savefig("{}/comparison_S-MARQUES.pdf".format(path), bbox_inches="tight")

# S-ISRI-OCR
fig, ax = plt.subplots(figsize=(18, 18))
fp = sns.lineplot(
    x="k",
    y="accuracy",
    hue="method",
    hue_order=methods,
    data=df[df["dataset"] == "D2"],
    # legend_out=False,
    legend=False,
    ci=95,
    markersize=50,
    marker="s",
    ax=ax,
    # palette=palette,
)
max_val = 20
values = [1, 2, 3, 4, 5] + list(range(10, max_val + 1, 5))
ax.set_xticks(values)  # , minor=False)
values = [1, None, None, None, 5] + [(value if value % 10 == 0 else None) for value in range(10, max_val + 1, 5)]
ax.set_xticklabels(values, fontdict={"fontsize": 120})
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(5 * [""])
ax.set_title("\\textsc{S-Isri-OCR}", fontdict={"fontsize": 120})
ax.set_xlabel("$k$", fontsize=120)
ax.set_ylabel("")
fp.set(xlim=(0, 21))
fp.set(ylim=(-1, 101))
plt.savefig("{}/comparison_S-ISRI-OCR.pdf".format(path), bbox_inches="tight")

# S-CDIP
max_val = 100
fig, ax = plt.subplots(figsize=(81, 18))
fp = sns.lineplot(
    x="k",
    y="accuracy",
    hue="method",
    hue_order=["proposed", "proposed-nn", "paixao", "marques"],
    data=df[df["dataset"] == "cdip"],
    # legend_out=False,
    legend=False,
    ci=95,
    markersize=50,
    marker="s",
    ax=ax,
    palette=[palette[0], palette[1], palette[2], palette[4]],
)
max_val = 100
values = [1, 2, 3, 4, 5] + list(range(10, max_val + 1, 5))
ax.set_xticks(values)  # , minor=False)
values = [1, None, None, None, 5] + [(value if value % 10 == 0 else None) for value in range(10, max_val + 1, 5)]
ax.set_xticklabels(values, fontdict={"fontsize": 120})
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels([0, 25, 50, 75, 100], fontdict={"fontsize": 120})
ax.set_title("\\textsc{S-Cdip}", fontdict={"fontsize": 120})
ax.set_xlabel("$k$", fontsize=120)
ax.set_ylabel("accuracy (\%)", fontsize=120)
fp.set(xlim=(0, 101))
fp.set(ylim=(-1, 101))

leg = ax.legend(
    handles=handles,
    labels=["\\textbf{Deeprec-CL}", "Deeprec-CL-NN", "Paixão", "Liang", "Marques"],
    title="method",
    prop={"size": 100},
    labelspacing=0.2,
    columnspacing=0.5,
    ncol=3,
    title_fontsize=100,
    loc="lower right",
    bbox_to_anchor=(1.0, 0.1),
)
plt.savefig("{}/comparison_S-CDIP.pdf".format(path), bbox_inches="tight")

# plt.show()
