import os
import numpy as np
import pandas as pd

import tensorflow as tf

from generator.augment import WeakAugment


def set_dataset(args, logger):
    trainset = pd.read_csv(os.path.join(args.src_path, '{}_trainset.csv'.format(args.dataset))).values.tolist()
    valset = pd.read_csv(os.path.join(args.src_path, '{}_valset.csv'.format(args.dataset))).values.tolist()
    testset = pd.read_csv(os.path.join(args.src_path, '{}_testset.csv'.format(args.dataset))).values.tolist()
    return np.array(trainset, dtype='object'), np.array(valset, dtype='object'), np.array(testset, dtype='object')


def dataloader(args, datalist, mode, shuffle=True):
    imglist, labellist = datalist[:,0].tolist(), datalist[:,1].tolist()
    imglist = [os.path.join(args.data_path, i) for i in imglist]
    dataset = tf.data.Dataset.from_tensor_slices((imglist, labellist))
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(datalist))

    def fetch_dataset(path, y):
        x = tf.io.read_file(path)
        return tf.data.Dataset.from_tensors((x, y))

    dataset = dataset.interleave(
        fetch_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def augmentation(img, label, shape):
        if args.augment == 'weak':
            augment = WeakAugment(args, mode)
        else:
            raise ValueError()

        for f in augment.augment_list:
            if 'crop' in f.__name__ or 'pad' in f.__name__:
                img = f(img, shape)
            else:
                img = f(img)

        label = tf.one_hot(label-1, args.classes)
        return img, label

    def preprocess_image(img, label):
        shape = tf.image.extract_jpeg_shape(img)
        img = tf.io.decode_jpeg(img, channels=3)
        img, label = augmentation(img, label, shape)
        if args.embedding == 'softmax':
            tensors = ({'main_input': img}, {'main_output': label})
        elif args.embedding == 'dual':
            tensors = ({'main_input': img, 'arcface_input': label}, 
                       {'main_output': label, 'arcface_output': label})
        elif args.embedding == 'arcface':
            tensors = ({'main_input': img, 'arcface_input': label}, 
                       {'arcface_output': label})
        return tensors

    dataset = dataset.map(
        preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset