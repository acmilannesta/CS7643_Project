
from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm
# import cv2
import shutil
import os
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from utils import make_example, process_image, read_images, process_label, parse_example
# from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--width', '-wt', type=int, default=227, help='Resized image width')
parser.add_argument('--height', '-ht', type=int, default=227, help='Resized image height')
parser.add_argument('--data_dir', '-dd', type=str, default='rsna-pneumonia-detection-challenge.zip',
                    help='Zip file path')
parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size in each tfrecord file')
parser.add_argument('--num_shards', '-ns', type=int, default=200,
                    help='Number of shards (default is 200, must be less than 945)')

args = parser.parse_args()

WIDTH = args.width
HEIGHT = args.height
BATCH_SIZE = args.batch_size
DATA_DIR = args.data_dir
NUM_SHARDS = args.num_shards
SEED = 42
NUM_CLASSES = 2
NUM_CHANNELS = 3
IMG_SIZE = (WIDTH, HEIGHT, NUM_CHANNELS)


def split_into_tfrecords(img_files, labels, tf_shard):
    imgs = list(map(process_image, read_images(img_files, DATA_DIR, (WIDTH, HEIGHT))))
    with tf.io.TFRecordWriter(path=f"tfrecords/shard{tf_shard}_{len(img_files)}.tfrecord") as f:
        for img, label in zip(imgs, labels):
            # label = process_label(label, NUM_CLASSES)
            example = make_example(img, label)
            f.write(example)

def main():

    # sanity check create folder for tfrecords
    if 'tfrecords' not in os.listdir():
        print('Tfrecords folder not available. Make directory!')
        os.mkdir('./tfrecords/')
    else:
        print('Tfrecords exist! Clean all files...')
        shutil.rmtree('./tfrecords')
        os.mkdir('./tfrecords')


    # Divide postive and control samples into shards
    with ZipFile(DATA_DIR, 'r') as archive:
        df_meta = pd.read_csv(archive.open('stage_2_train_labels.csv'))
    df_meta = df_meta.drop_duplicates('patientId')
    # n_folds = np.ceil(len(df_meta) / BATCH_SIZE)
    pos = df_meta[df_meta['Target'] == 1].sample(frac=1, random_state=SEED, ignore_index=True)
    neg = df_meta[df_meta['Target'] == 0].sample(frac=1, random_state=SEED, ignore_index=True)

    t_pos = t_neg = 0
    for i, (pos_split, neg_split) in tqdm(enumerate(zip(np.array_split(pos, len(pos) // (BATCH_SIZE // 2)),
                                                        np.array_split(neg, len(neg) // (BATCH_SIZE // 2)))),
                                          total=NUM_SHARDS):
        if i == NUM_SHARDS: break
        df_tmp = pd.concat((pos_split, neg_split), axis=0)
        img_files, labels = df_tmp['patientId'].tolist(), df_tmp['Target'].tolist()
        t_pos += np.sum(labels)
        t_neg += len(labels) - np.sum(labels)
        split_into_tfrecords(img_files, labels, i)

    print(f'{NUM_SHARDS} tfrecords created!')
    print(f'{t_pos} pneumonia and {t_neg} normal samples.')


if __name__ == '__main__':
    main()

