from itertools import product
from .config import CONFIG_SAMPLES_DIR

# default setup
NEUTRAL_THRESH_DEFAULT = 0.2
MAX_SAMPLES_DEFAULT = 1000
SAMPLE_SIZE_DEFAULT = (32, 32)
VSHIFT_DEFAULT = 10
INPUT_SIZE_HEIGHT_DEFAULT = 3000
MAX_DOCS_DEFAULT = 5

training_dataset = lambda test_dataset: 'isri-ocr' if test_dataset == 'cdip' else 'cdip'


def generate_samples(dataset, neutral_thresh=NEUTRAL_THRESH_DEFAULT, max_samples=MAX_SAMPLES_DEFAULT, sample_size=SAMPLE_SIZE_DEFAULT):

    dataset_param = '--dataset {}'.format(dataset)
    neutral_thresh_param = '--neutral-thresh {}'.format(neutral_thresh)
    max_samples_param = '--max-samples {}'.format(max_samples)
    sample_size_param = '--sample-size {} {}'.format(sample_size[0], sample_size[1])
    save_dir_param = '--save-dir {}/{}_{}_{}_{}x{}'.format(CONFIG_SAMPLES_DIR, dataset, neutral_thresh, max_samples, sample_size[0], sample_size[1])
    return 'python generate_samples.py {} {} {} {} {}'.format(
        neutral_thresh_param, dataset_param, max_samples_param, sample_size_param, save_dir_param
    )


def remove_samples(dataset, neutral_thresh=NEUTRAL_THRESH_DEFAULT, max_samples=MAX_SAMPLES_DEFAULT, sample_size=SAMPLE_SIZE_DEFAULT):

    save_dir_param = '{}/{}_{}_{}_{}x{}'.format(CONFIG_SAMPLES_DIR, dataset, neutral_thresh, max_samples, sample_size[0], sample_size[1])
    return 'rm -rf {}'.format(save_dir_param)


def train(dataset, neutral_thresh=NEUTRAL_THRESH_DEFAULT, max_samples=MAX_SAMPLES_DEFAULT, sample_size=SAMPLE_SIZE_DEFAULT):

    samples_dir_param = '--samples-dir {}/{}_{}_{}_{}x{}'.format(CONFIG_SAMPLES_DIR, dataset, neutral_thresh, max_samples, sample_size[0], sample_size[1])
    model_id_param = '--model-id {}_{}_{}_{}x{}'.format(dataset, neutral_thresh, max_samples, sample_size[0], sample_size[1])
    return 'python train.py {} {}'.format(model_id_param, samples_dir_param)


def test(dataset, neutral_thresh=NEUTRAL_THRESH_DEFAULT, max_samples=MAX_SAMPLES_DEFAULT, sample_size=SAMPLE_SIZE_DEFAULT, vshift=VSHIFT_DEFAULT):

    dataset_param = '--dataset {}'.format(dataset)
    model_id_param = '--model-id {}_{}_{}_{}x{}'.format(training_dataset(dataset), neutral_thresh, max_samples, sample_size[0], sample_size[1])
    results_id_param = '--results-id {}_{}_{}_{}x{}_{}'.format(dataset, neutral_thresh, max_samples, sample_size[0], sample_size[1], vshift)
    input_size_param = '--input-size {} {}'.format(INPUT_SIZE_HEIGHT_DEFAULT, sample_size[1])
    vshift_param = '--vshift {}'.format(vshift)
    max_docs_param = '--max-ndocs {}'.format(MAX_DOCS_DEFAULT)
    return 'python -m exp2_ablation.test {} {} {} {} {} {}'.format(
        dataset_param, model_id_param, results_id_param, input_size_param, vshift_param, max_docs_param
    )

training_datasets = ['isri-ocr', 'cdip']
test_datasets = ['D1', 'D2', 'cdip']
neutral_thresh_range = [0.1, 0.2, 0.3]
max_samples_range = [500, 1000, 1500]
sample_size_range = [32, 48, 64]
vshift_range = [0, 5, 10, 15, 20]

# default
generated_datasets = []
default_commands = []
commands = []
commands.append('# default ----------')
for test_dataset in test_datasets:
    # generate_command = generate_samples(training_dataset(test_dataset))
    # train_command = train(training_dataset(test_dataset))
    test_command = test(test_dataset)
    # if generate_command not in commands:
    #     commands.append(generate_command)
    #     commands.append(train_command)
    commands.append(test_command)

# others
commands.append('')
commands.append('# others ----------')

for neutral_thresh in neutral_thresh_range:
    for test_dataset in test_datasets:
        generate_command = generate_samples(training_dataset(test_dataset), neutral_thresh=neutral_thresh)
        remove_command = remove_samples(training_dataset(test_dataset), neutral_thresh=neutral_thresh)
        train_command = train(training_dataset(test_dataset), neutral_thresh=neutral_thresh)
        test_command = test(test_dataset, neutral_thresh=neutral_thresh)
        if train_command not in commands:
            comment = '# neutral_thresh={}'.format(neutral_thresh)
            if comment not in commands:
                commands.append('')
                commands.append(comment)
            commands.append(generate_command)
            commands.append(train_command)
            commands.append(remove_command)
        if test_command not in commands:
            commands.append(test_command)

for max_samples in max_samples_range:
   for test_dataset in test_datasets:
       generate_command = generate_samples(training_dataset(test_dataset), max_samples=max_samples)
       remove_command = remove_samples(training_dataset(test_dataset), max_samples=max_samples)
       train_command = train(training_dataset(test_dataset), max_samples=max_samples)
       test_command = test(test_dataset, max_samples=max_samples)
       if generate_command not in commands:
           comment = '# max_samples={}'.format(max_samples)
           if comment not in commands:
               commands.append('')
               commands.append(comment)
           commands.append(generate_command)
           commands.append(train_command)
           commands.append(remove_command)
       if test_command not in commands:
           commands.append(test_command)


for sample_size in product(sample_size_range, sample_size_range):
   for test_dataset in test_datasets:
       generate_command = generate_samples(training_dataset(test_dataset), sample_size=sample_size)
       remove_command = remove_samples(training_dataset(test_dataset), sample_size=sample_size)
       train_command = train(training_dataset(test_dataset), sample_size=sample_size)
       test_command = test(test_dataset, sample_size=sample_size)
       if generate_command not in commands:
           comment = '# sample_size={}x{}'.format(sample_size[0], sample_size[1])
           if comment not in commands:
               commands.append('')
               commands.append(comment)
           commands.append(generate_command)
           commands.append(train_command)
           commands.append(remove_command)
       if test_command not in commands:
           commands.append(test_command)


for vshift in vshift_range:
   for test_dataset in test_datasets:
       test_command = test(test_dataset, vshift=vshift)
       if test_command not in commands:
           comment = '# vshift={}'.format(vshift)
           if comment not in commands:
               commands.append('')
               commands.append(comment)
           commands.append(test_command)

txt = '\n'.join(commands)
open('exp2_ablation/run.sh', 'w').write(txt)