import os
import cv2
import json
import threading
import numpy as np
import pandas as pd

import tensorflow as tf


class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def dataloader(args, mode, datalist, shuffle=True):
    def _get_pad(_img):
        height, width = _img.shape[:2]
        if height > width:
            residual = (height - width)//2
            pad_width = [[0, 0], [residual, height-(width+residual)], [0, 0]]
            
        elif width > height:
            residual = (width - height)//2
            pad_width = [[residual, width-(height+residual)], [0, 0], [0, 0]]
        
        else:
            pad_width = [[0, 0], [0, 0], [0, 0]]
        
        _img_pad = np.pad(_img, pad_width)
        return _img_pad

    def _set_cropbox(_img):
        crop_ratio = np.random.uniform(low=.8, high=1., size=(2,))
        crop_size = [int(s*r) for s, r in zip(_img.shape[:2], crop_ratio)]+[3]
        # aspect_ratio = np.random.uniform(low=3/4, high=4/3)
        # ratio = [crop_ratio, crop_ratio*aspect_ratio]
        # crop_size = [np.minimum(52, int(52*r)) for r in ratio]+[3]
        return crop_size

    def _loadimg(_data, _mode):
        img = cv2.imread(os.path.join(args.data_path, _data[0]))[...,::-1]
        label = np.array(_data[1:], dtype=np.float32)
        if _mode == 'train':
            prob = np.random.random()
            if args.crop == 0:
                crop_size = [0, 0, 0]
            elif args.crop == 1:
                if prob > .5:
                    img = _get_pad(img)
                crop_size = [0, 0, 0]
            elif args.crop == 2:
                if prob < .3:
                    crop_size = [0, 0, 0]
                elif .3 <= prob < .6:
                    img = _get_pad(img)
                    crop_size = [0, 0, 0]
                else:
                    crop_size = _set_cropbox(img)
                
        else:
            img = _get_pad(img)
            crop_size = [0, 0, 0]

        img = img.astype(np.float32)
        img /= 255.
        return img, label, crop_size

    def _class_gen(_datalist, flip_class=None):
        indices = np.arange(len(_datalist))
        # print(_datalist.shape)
        while True:
            if shuffle:
                indices = np.random.permutation(len(_datalist))

            for idx in indices:
                data = _datalist[idx]
                img, label, crop_bbox = _loadimg(data, mode)
                yield img, label, crop_bbox

    @threadsafe_generator
    def _loader():
        if mode == 'train':
            class_gen = {c: _class_gen(datalist[np.where(datalist[:,c+1] == 1)]) for c in range(args.classes)}
        else:
            indices = np.arange(len(datalist))
        
        while True:
            if mode == 'train':
                if shuffle:
                    class_list = np.random.permutation(args.classes)
                else:
                    class_list = np.arange(args.classes)

                for c in range(len(class_list)):
                    if args.mixup:
                        if np.random.random() > .5:
                            x1, y1, _ = next(class_gen[class_list[c]])
                            x2, y2, _ = next(class_gen[class_list[0 if c == len(class_list)-1 else c+1]])

                            height = min(x1.shape[0], x2.shape[0])
                            width = min(x1.shape[1], x2.shape[1])
                            x1 = x1[max(0, (x1.shape[0]-height)//2):min(x1.shape[0], (x1.shape[0]+height)//2),
                                    max(0, (x1.shape[1]-width)//2):min(x1.shape[1], (x1.shape[1]+width)//2)]
                            x2 = x2[max(0, (x2.shape[0]-height)//2):min(x2.shape[0], (x2.shape[0]+height)//2),
                                    max(0, (x2.shape[1]-width)//2):min(x2.shape[1], (x2.shape[1]+width)//2)]
                            
                            ratio = np.random.beta(a=.2, b=.2)
                            ratio = np.maximum(.5, ratio)
                            ratio = np.clip(ratio, 1e-7, 1-1e-7)
                            x = x1*ratio + x2*(1-ratio)
                            x = np.clip(x, 0., 1.)
                            y = y1*ratio + y2*(1-ratio)
                            crop_bbox = _set_cropbox(x)
                            yield x, y, crop_bbox
                        else:
                            yield next(class_gen[class_list[c]])
                    else:
                        yield next(class_gen[class_list[c]])
            else:
                for idx in indices:
                    data = datalist[idx]
                    img, label, crop_bbox = _loadimg(data, mode)
                    if mode == 'test':
                        yield img, label, [data[0]]
                    else:
                        yield img, label, crop_bbox

    return _loader


def set_dataset(args, logger):
    trainset_whole = []
    for v in range(5):
        if v+1 == args.version:
            continue
        trainset_whole += pd.read_csv(os.path.join(args.src_path, 'trainset_{}.csv'.format(v+1))).values.tolist()
    valset_whole = pd.read_csv(os.path.join(args.src_path, 'trainset_{}.csv'.format(args.version))).values.tolist()
    
    trainset = []
    valset = []
    for dataset in ['trainset', 'valset']:
        exec('{} = []'.format(dataset))
        for t in eval('{}_whole'.format(dataset)):
            label = t[1:]
            if sum(label) == 1:
                eval(dataset).append(t)
            else:
                logger.info('{} don\'t have label!'.format(t[0]))

    logger.info('Train set : {}'.format(len(trainset)))
    logger.info('Validation set : {}'.format(len(valset)))

    if args.mode == 'supervised':
        return np.array(trainset, dtype='object'), np.array(valset, dtype='object')
        
    elif args.mode == 'uda':
        unlabelset = pd.read_csv(os.path.join(args.src_path, 'u_train_in.csv')).values.tolist()
        logger.info('Unlabeled set : {}'.format(len(unlabelset)))
        return np.array(trainset, dtype='object'), np.array(valset, dtype='object'), np.array(unlabelset, dtype='object')


def create_generator(args, dataset, mode, batch_size):
    output_types=(tf.dtypes.float32, tf.dtypes.float32)
    output_shapes = (tf.TensorShape((None, None, None,)), tf.TensorShape((None,)))

    if mode == 'test':
        output_types += (tf.dtypes.string,)
        output_shapes += (tf.TensorShape((None,)),)
    else:
        output_types += (tf.dtypes.int32,)
        output_shapes += (tf.TensorShape((None,)),)
        
    from .augment import Preprocess
    prep_func = Preprocess(args, mode)

    generator = tf.data.Dataset.from_generator(
        dataloader(args, mode, dataset, shuffle=True if mode == 'train' else False), 
        output_types=output_types,
        output_shapes=output_shapes,
    )
    for f in prep_func.prep_func_list:
        generator = generator.map(lambda *args: f(*args), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    if mode != 'test':
        if args.embedding == 'arcface':
            generator = generator.map(lambda x, y, z: ({'main_input': x, 'arcface_input': y}, 
                                                       {'arcface_output': y}), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif args.embedding == 'dual':
            generator = generator.map(lambda x, y, z: ({'main_input': x, 'arcface_input': y}, 
                                                       {'main_output': y, 'arcface_output': y}), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            generator = generator.map(lambda x, y, z: ({'main_input': x}, {'main_output': y}), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    else:
        if args.embedding == 'arcface':
            generator = generator.map(lambda x, y, z: ({'main_input': x, 'arcface_input': y}, 
                                                       {'arcface_output': y},
                                                       {'patient': z}), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif args.embedding == 'dual':
            generator = generator.map(lambda x, y, z: ({'main_input': x, 'arcface_input': y}, 
                                                       {'main_output': y, 'arcface_output': y},
                                                       {'patient': z}), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            generator = generator.map(lambda x, y, z: ({'main_input': x}, {'main_output': y},
                                                       {'patient': z}), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    generator = generator.batch(batch_size)
    generator = generator.prefetch(tf.data.experimental.AUTOTUNE)

    return generator
