# python -m illustration.samples --samples-dir ~/samples/isri-ocr
import os
import numpy as np
import cv2
import argparse
import random
import glob


if __name__ == '__main__':

    base_path = 'illustration/samples' 
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    parser = argparse.ArgumentParser(description='Dataset samples generation.')
    parser.add_argument(
        '-sd', '--samples-dir', action='store', dest='samples_dir', required=False, type=str,
        default='~/datasets/samples', help='Path where samples (samples) are placed.'
    )
    parser.add_argument(
        '-gr', '--grid-size', action='store', dest='grid_size', required=False, nargs=2, type=int,
        default=[20, 20], help='Grid size.'
    )
    
    args = parser.parse_args()
    grid_size = tuple(args.grid_size)

    negatives = glob.glob('{}/negatives/train/*.npy'.format(args.samples_dir))
    positives = glob.glob('{}/positives/train/*.npy'.format(args.samples_dir))
    neutrals = glob.glob('{}/neutrals/train/*.npy'.format(args.samples_dir))
    random.shuffle(negatives)
    random.shuffle(positives)
    random.shuffle(neutrals)

    num_images = grid_size[0] * grid_size[1]
    
    load_func = lambda fname: np.pad(np.load(fname), [(1, 1), (1, 1)], 'constant', constant_values=0)
    negatives = [load_func(fname) for fname in negatives[: num_images]]
    positives = [load_func(fname) for fname in positives[: num_images]]
    neutrals = [load_func(fname) for fname in neutrals[: num_images]]
    sample_size = negatives[0].shape
    black = np.zeros((sample_size[0], 5), dtype=np.uint8)

    grid = []
    for i in range(grid_size[0]):
        line = negatives[i * grid_size[1] : (i + 1) * grid_size[1]] + [black]
        line += neutrals[i * grid_size[1] : (i + 1) * grid_size[1]] + [black]
        line += positives[i * grid_size[1] : (i + 1) * grid_size[1]]
        grid.append(line)

    panel = (255 * np.block(grid)).astype(np.uint8)
    cv2.imwrite('{}/panel.jpg'.format(base_path), panel)