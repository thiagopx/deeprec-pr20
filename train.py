# python train.py --model-id isri-ocr --samples-dir ~/samples/isri-ocr

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import sys
import time

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from docrec.neural.models.squeezenet import SqueezeNet
from docrec.neural.models.squeezenet_simple_bypass import SqueezeNetSB

mpl.use('Agg')


tf.logging.set_verbosity(tf.logging.INFO)

# basic setup
NUM_CLASSES = 2
SEED = 0 # <= change this in case of multiple runs
np.random.seed(SEED)
tf.set_random_seed(SEED)


class Dataset:

    def __init__(self, args, mode='train', shuffle_every_epoch=True):

        assert mode in ['train', 'val']

        lines = open('{}/{}.txt'.format(args.samples_dir, mode)).readlines()
        info = json.load(open('{}/info.json'.format(args.samples_dir), 'r'))
        num_negatives = info['stats']['negatives_{}'.format(mode)]
        num_positives = info['stats']['positives_{}'.format(mode)]
        num_samples_per_class = min(num_positives, num_negatives)

        self.num_samples = 2 * num_samples_per_class
        self.curr_epoch = 1
        self.num_epochs = args.num_epochs
        self.curr_batch = 1
        self.batch_size = args.batch_size
        self.sample_size = info['params']['sample_size']
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        self.shuffle_every_epoch = shuffle_every_epoch

        assert self.num_samples > self.batch_size

        # load data
        count = {'0': 0, '1': 0}
        labels = []
        images = []
        for line in lines:
            filename, label = line.split()
            if count[label] < num_samples_per_class:
                image = np.load(filename)
                images.append(image)
                labels.append(label)
                count[label] += 1

        # data in array format
        self.images = np.array(images).astype(np.float32)
        self.labels = np.array(labels).astype(np.int32)
        if mode == 'train':
            self.labels = self._one_hot(self.labels)


    def _one_hot(self, labels):

        one_hot = np.zeros((labels.shape[0], NUM_CLASSES))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot


    def next_batch(self):

        assert self.curr_epoch <= self.num_epochs

        # shuffle dataset
        if (self.curr_batch == 1) and (self.shuffle_every_epoch):
            self.images, self.labels = shuffle(self.images, self.labels)

        # crop batch
        i1 = (self.curr_batch - 1) * self.batch_size
        i2 = i1 + self.batch_size
        images = self.images[i1 : i2]
        labels = self.labels[i1 : i2]

        # images must be in RGB (channels last) float32 format
        images = np.stack([images, images, images], axis=0).transpose([1, 2, 3, 0]).astype(np.float32)

        # next epoch?
        if self.curr_batch == self.num_batches:
            self.curr_epoch += 1
            self.curr_batch = 1
        else:
            self.curr_batch += 1

        return images, labels


def train(args):
    ''' Training stage. '''

    # network input size
    print('loading training samples :: ', end='')
    sys.stdout.flush()
    dataset = Dataset(args, mode='train')
    height, width = dataset.sample_size
    print('num_samples={} sample_size={}x{}'.format(dataset.num_samples, height, width))

    # general variables and ops
    global_step_var = tf.Variable(1, trainable=False, name='global_step')
    inc_global_step_op = global_step_var.assign_add(1)

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, height, width, 3)) # channels last
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None, NUM_CLASSES))  # one-hot enconding

    # model
    if args.arch == 'sn':
        model = SqueezeNet(images_ph, num_classes=NUM_CLASSES, mode='train', channels_first=False, seed=SEED)
    else:
        model = SqueezeNetSB(images_ph, num_classes=NUM_CLASSES, mode='train', channels_first=False, seed=SEED)

    sess = model.sess
    logits_op = tf.reshape(model.output, [-1, NUM_CLASSES])
        
    # loss function
    loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels_ph, logits=logits_op)

    # learning rate definition
    num_steps_per_epoch = dataset.num_batches  # math.ceil(num_samples / args.batch_size)
    total_steps = args.num_epochs * num_steps_per_epoch
    decay_steps = math.ceil(args.step_size * total_steps)
    learning_rate_op = tf.train.exponential_decay(
        args.learning_rate, global_step_var, decay_steps, 0.1, staircase=True
    )

    # optimizer (adam method)
    optimizer = tf.train.AdamOptimizer(learning_rate_op)
    # training step operation
    train_op = optimizer.minimize(loss_op)

    # summary
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # comment the line beneath to train from scratch
    model.load_weights('docrec/neural/models/imagenet.npy', ignore_layers=['conv10'], BGR=True, ignore_missing=False)
    
    # training loop
    start = time.time()
    losses_group = []
    losses = []
    steps = []
    for epoch in range(1, args.num_epochs + 1):
        for step in range(1, num_steps_per_epoch + 1):
            # global step
            global_step = sess.run(global_step_var)
            # batch data
            images, labels = dataset.next_batch()
            # train
            learning_rate, loss, x = sess.run(
                [learning_rate_op, loss_op, train_op],
                feed_dict={images_ph: images, labels_ph: labels}
            )
            losses_group.append(loss)
            if (step % 10 == 0) or (step == num_steps_per_epoch):
                losses.append(np.mean(losses_group))
                steps.append(global_step)
                elapsed = time.time() - start
                remaining = elapsed * (total_steps - global_step) / global_step
                print('[{}] [{:.2f}%] step={}/{} epoch={} loss={:.3f} :: {:.2f}/{:.2f} seconds lr={:.7f}'.format(
                    args.model_id, 100 * global_step / total_steps, global_step, total_steps, epoch,
                    np.mean(losses_group), elapsed, remaining, learning_rate
                ))
                losses_group = []
            # increment global step
            sess.run(inc_global_step_op)
        # save epoch model
        model.save_weights('traindata/{}/model/{}.npy'.format(args.model_id, epoch))
    plt.plot(steps, losses)
    plt.savefig('traindata/{}/loss.png'.format(args.model_id))
    sess.close()


def validate(args):
    ''' Validate and select the best model. '''

    print('loading validation samples :: ', end='')
    dataset = Dataset(args, mode='val', shuffle_every_epoch=False)
    height, width = dataset.sample_size
    print('num_samples={} sample_size={}x{}'.format(dataset.num_samples, height, width))

    # placeholders
    images_ph = tf.placeholder(tf.float32, name='images_ph', shape=(None, height, width, 3)) # channels last
    labels_ph = tf.placeholder(tf.float32, name='labels_ph', shape=(None,))                  # normal encoding

    # model
    if args.arch == 'sn':
        model = SqueezeNet(images_ph, num_classes=NUM_CLASSES, mode='val', channels_first=False)
    else:
        model = SqueezeNetSB(images_ph, num_classes=NUM_CLASSES, mode='val', channels_first=False)
        
    sess = model.sess
    logits_op = tf.reshape(model.output, [-1, NUM_CLASSES ])
    
    # predictions
    predictions_op = tf.argmax(logits_op, 1)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # validation loop
    num_steps_per_epoch = dataset.num_batches
    best_epoch = 0
    best_accuracy = 0.0
    accuracies = []
    for epoch in range(1, args.num_epochs + 1):
        model.load_weights('traindata/{}/model/{}.npy'.format(args.model_id, epoch))
        total_correct = 0
        total_neg_pos_samples = 0
        for step in range(1, num_steps_per_epoch + 1):
            images, labels = dataset.next_batch()
            batch_size = images.shape[0]
            logits, predictions = sess.run(
                [logits_op, predictions_op],
                feed_dict={images_ph: images, labels_ph: labels}
            )
            num_correct = np.sum(predictions == labels)
            total_correct += num_correct
            if (step % 10 == 0) or (step == num_steps_per_epoch):
                print('step={} accuracy={:.2f}'.format(step, 100 * num_correct / batch_size))
        # epoch average accuracy
        accuracy = 100.0 * total_correct / dataset.num_samples
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
        print('-------------------------------------------------')
        print('epoch={} (best={}) accuracy={:.2f} (best={:.2f})'.format(epoch, best_epoch, accuracy, best_accuracy))
        print('-------------------------------------------------')
    sess.close()
    print('best epoch={} accuracy={:.2f}'.format(best_epoch, best_accuracy))
    plt.clf()
    plt.plot(list(range(1, args.num_epochs + 1)), accuracies)
    plt.savefig('traindata/{}/accuracies.png'.format(args.model_id))
    return best_epoch


def main():

    t0 = time.time()

    parser = argparse.ArgumentParser(description='Training the network.')
    parser.add_argument(
        '-np', '--num-processors', action='store', dest='num_processors', required=False, type=int,
        default=8, help='Number of processors.'
    )
    parser.add_argument(
        '-lr', '--learning-rate', action='store', dest='learning_rate', required=False, type=float,
        default=0.0001, help='Learning rate.'
    )
    parser.add_argument(
        '-bs', '--batch-size', action='store', dest='batch_size', required=False, type=int,
        default=256, help='Batch size.'
    )
    parser.add_argument(
        '-ep', '--num-epochs', action='store', dest='num_epochs', required=False, type=int,
        default=10, help='Number of training epochs.'
    )
    parser.add_argument(
        '-ss', '--step-size', action='store', dest='step_size', required=False, type=float,
        default=0.33, help='Step size for learning with step-down policy.'
    )
    parser.add_argument(
        '-sd', '--samples-dir', action='store', dest='samples_dir', required=False, type=str,
        default='~/datasets/samples', help='Path where samples (samples) are placed.'
    )
    parser.add_argument(
        '-ar', '--arch', action='store', dest='arch', required=False, type=str,
        default='sn', help='Architecture of the CNN'
    )
    parser.add_argument(
        '-mi', '--model-id', action='store', dest='model_id', required=False, type=str,
        default=None, help='Model identifier (tag).'
    )
    args = parser.parse_args()
    
    assert args.arch in ['sn', 'sn-bypass']

    # training stage
    train(args)
    # validation
    best_epoch = validate(args)

    t1 = time.time()

    # dump training info
    info = {
        'time_minutes': (t1 - t0) / 60.0,
        'time_seconds': (t1 - t0),
        'best_model': 'traindata/{}/model/{}.npy'.format(args.model_id, best_epoch),
        'params': args.__dict__
    }
    json.dump(info, open('traindata/{}/info.json'.format(args.model_id), 'w'))
    return t1 - t0


if __name__ == '__main__':

    t = main()
    print('Elapsed time={:.2f} minutes ({} seconds)'.format(t / 60.0, t))
