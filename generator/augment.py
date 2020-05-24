import numpy as np
from functools import wraps

import tensorflow as tf
import tensorflow_addons as tfa

def prep_gate(func):
    @wraps(func)
    def _gate(*args):
        return func(*args)
    return _gate

class Preprocess:
    mean_std = [[0.50532048, 0.52774713, 0.49474949],
                [0.17169899, 0.17590892, 0.18991528]]

    def __init__(self, args, mode, angle_range=15):
        self.args = args
        self.mode = mode
        self.angle_range = angle_range

        self.mean, self.std = self.mean_std

        self.prep_func_list = [self._standardize]
        if self.mode == 'train':
            if self.args.crop:
                self.prep_func_list.append(self._resizecrop)
            
            else:
                self.prep_func_list.append(self._resize)

            if self.args.angle:
                self.prep_func_list.append(self._rotate)
            
            if self.args.vflip:
                self.prep_func_list.append(self._verticalflip)

            if self.args.hflip:
                self.prep_func_list.append(self._horizontalflip)
            
            if self.args.distortion:
                self.prep_func_list.append(self._distortion)

            if self.args.contrast:
                self.prep_func_list.append(self._contrast)

            if self.args.noise:
                self.prep_func_list.append(self._noise)

        else:
            self.prep_func_list.append(self._resize)

    @prep_gate
    def _standardize(self, x, y, data):
        if self.args.standardize == 'minmax1':
            pass
        elif self.args.standardize == 'minmax2':
            x -= .5
            x /= .5
        elif self.args.standardize == 'norm':
            x -= self.mean
            x /= self.std
        elif self.args.standardize == 'eachnorm':
            x = (x-tf.math.reduce_mean(x))/tf.math.reduce_std(x)
        else:
            raise
        
        return x, y, data

    @prep_gate
    def _crop(self, x, y, data):
        if tf.math.reduce_sum(data) > 0:
            x = tf.image.random_crop(x, data)
        return x, y, data

    @prep_gate
    def _resize(self, x, y, data):
        img_min, img_max = tf.math.reduce_min(x), tf.math.reduce_max(x)
        x = tf.image.resize(x, (self.args.img_size, self.args.img_size))
        x = tf.clip_by_value(x, img_min, img_max)
        return x, y, data

    @prep_gate
    def _resizecrop(self, x, y, data):
        img_min, img_max = tf.math.reduce_min(x), tf.math.reduce_max(x)
        x = tf.image.resize_with_crop_or_pad(x, self.args.img_size, self.args.img_size)
        x = tf.clip_by_value(x, img_min, img_max)
        return x, y, data

    @prep_gate
    def _rotate(self, x, y, data):
        img_min, img_max = tf.math.reduce_min(x), tf.math.reduce_max(x)
        x = tfa.image.rotate(x, tf.random.uniform(minval=-self.args.angle, maxval=self.args.angle, shape=[])*np.pi/180, interpolation='BILINEAR')
        x = tf.clip_by_value(x, img_min, img_max)
        return x, y, data

    @prep_gate
    def _noise(self, x, y, data):
        img_min, img_max = tf.math.reduce_min(x), tf.math.reduce_max(x)
        x += tf.random.normal(stddev=.1, shape=(self.args.img_size, self.args.img_size, self.args.img_channel))
        x = tf.clip_by_value(x, img_min, img_max)
        return x, y, data

    @prep_gate
    def _contrast(self, x, y, data):
        img_min, img_max = tf.math.reduce_min(x), tf.math.reduce_max(x)
        # return tf.image.random_contrast(x, .5, 1.5)
        x *= tf.random.uniform(minval=.9, maxval=1.1, shape=(self.args.img_size, self.args.img_size, self.args.img_channel))
        x = tf.clip_by_value(x, img_min, img_max)
        return x, y, data

    @prep_gate
    def _distortion(self, x, y, data, s=.5):
        img_min, img_max = tf.math.reduce_min(x), tf.math.reduce_max(x)
        def __random_apply(func, x, p):
            return tf.cond(
                tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                        tf.cast(p, tf.float32)),
                lambda: func(x),
                lambda: x)

        def __color_jitter(x):
            x = tf.image.random_brightness(x, max_delta=0.8*s)
            x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
            x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
            x = tf.image.random_hue(x, max_delta=0.2*s)
            x = tf.clip_by_value(x, img_min, img_max)
            return x

        def __color_drop(x):
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
            return x

        x = __random_apply(__color_jitter, x, p=.8)
        x = __random_apply(__color_drop, x, p=.2)
        return x, y, data

    @prep_gate
    def _verticalflip(self, x, y, data):
        x = tf.image.random_flip_up_down(x)
        return x, y, data

    @prep_gate
    def _horizontalflip(self, x, y, data):
        x = tf.image.random_flip_left_right(x)
        return x, y, data
