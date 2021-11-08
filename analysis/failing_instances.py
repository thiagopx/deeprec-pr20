import json

fname_template = 'comparison/liang/{}_threads=240_use-mask=True.json'
datasets = ['D1', 'D2']
ks = {}
for dataset in datasets:
    if dataset == 'D2':
        continue
    fname = fname_template.format(dataset)
    results = json.load(open(fname, 'r'))['data']
    ks[dataset] = sorted(int(x) for x in results.keys())
    print('dataset={}'.format(dataset))
    for k in ks[dataset]:
        num_failed = 0
        total = 0
        for run in results[str(k)]:
            # doc = run['docs'][0].split('/')[-1]
            accuracy = run['accuracy']
            if run['solution'] == None:
                num_failed += 1
            total += 1
        print('   k={} {}/{}'.format(k, num_failed, total))