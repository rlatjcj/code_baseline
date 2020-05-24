import os
import cv2
import json
import random
import numpy as np
import pandas as pd

import tensorflow as tf

from .augment import Preprocess

def augmentataion(args, x):    
    def _apply_aug(x, mode):
        prep_func_aug = Preprocess(args, mode)
        for f in prep_func_aug.prep_func_list:
            x, _, _ = f(x, None, None)
        return x

    return {'img': _apply_aug(x, 'val'), 'augimg': _apply_aug(x, 'train')}

def dataloader(args, datalist, uratio=5, shuffle=True):
    datalist = [os.path.join(args.data_path, d) for d in datalist.flatten()]
    if shuffle:
        random.shuffle(datalist)

    dataset = tf.data.Dataset.from_tensor_slices(datalist)        
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(len(datalist))
        
    def fetch_dataset(path):
        x = tf.io.read_file(path)
        x = tf.io.decode_jpeg(x, channels=3)
        x = tf.cast(x, tf.float32)
        x /= 255.
        return tf.data.Dataset.from_tensors(x)

    dataset = dataset.interleave(
        lambda x: fetch_dataset(x), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(
        lambda x: augmentataion(args, x), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(args.batch_size*args.unsup_ratio)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset