import os
import argparse
import cv2
import math
import numpy as np
from sklearn.utils import shuffle

# basic setup
NUM_CLASSES = 3
seed = 0 # <= change this in case of multiple runs
np.random.seed(seed)


class Dataset:

    def __init__(self, args, mode='train', shuffle_every_epoch=True):

        assert mode in ['train', 'val']
      
        txt_file = '{}/{}.txt'.format(args.samples_dir, mode)
        lines = open(txt_file).readlines()
        if args.max_samples is not None:
            lines = lines[ : args.max_samples]


        self.num_samples = len(lines)
        self.curr_epoch = 1
        self.num_epochs = args.num_epochs
        self.curr_batch = 1
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        self.shuffle_every_epoch = shuffle_every_epoch

        assert self.num_samples > self.batch_size

        # load data        
        images = []
        labels = []
        for line in lines:
            filename, label = line.split()
            images.append(np.load(filename))
            labels.append(label)
        self.sample_size = images[0].shape

        # data in array format
        self.images = np.array(images)#.astype(np.float32)
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
        if self.curr_batch == 1 and self.shuffle_every_epoch:
            self.images, self.labels = shuffle(self.images, self.labels)
        
        i1 = (self.curr_batch - 1) * self.batch_size
        i2 = i1 + self.batch_size
        images, labels = self.images[i1 : i2], self.labels[i1 : i2]

        # images must be in RGB (channels last) float32 format
        images = np.stack([images, images, images], axis=0).transpose([1, 2, 3, 0]).astype(np.float32)
        
        # next epoch?
        if self.curr_batch == self.num_batches:
            self.curr_epoch += 1
            self.curr_batch = 1
        else:
            self.curr_batch += 1
        
        return images, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the network.')
    parser.add_argument(
        '-ms', '--max-samples', action='store', dest='max_samples', required=False, type=int,
        default=8, help='Max samples.'
    )
    parser.add_argument(
        '-bs', '--batch-size', action='store', dest='batch_size', required=False, type=int,
        default=3, help='Batch size.'
    )
    parser.add_argument(
        '-ep', '--epochs', action='store', dest='num_epochs', required=False, type=int,
        default=2, help='Number of training epochs.'
    )
    parser.add_argument(
        '-sd', '--samples-dir', action='store', dest='samples_dir', required=False, type=str,
        default='~/datasets/samples', help='Path where samples (samples) are placed.'
    )
    args = parser.parse_args()

    base_path = 'illustration/dataset'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    dataset = Dataset(args, 'train', shuffle_every_epoch=True)
    while (True):
        try:
            epoch, batch = dataset.curr_epoch, dataset.curr_batch
            images, labels = dataset.next_batch()
            print('epoch={} batch={} batch_size={} num_batches={}'.format(epoch, batch, images.shape[0], dataset.num_batches))

            images = (255 * images).astype(np.uint8)
            labels = labels.astype(np.int32)

            id_ = 1
            for image, label in zip(images, labels):
                label_str = ' '.join(str(x) for x in label)
                fname = '{}/{}-{}-{}_{}.jpg'.format(base_path, epoch, batch, id_, label_str)
                cv2.imwrite(fname, image)
                id_ += 1
        except AssertionError:
            print('end!')
            break