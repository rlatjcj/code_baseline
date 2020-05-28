import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


mean_std = {
    'cub': [[0.48552202, 0.49934904, 0.43224954], 
            [0.18172876, 0.18109447, 0.19272076]],
}

class Augment:    
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        self.mean, self.std = mean_std[self.args.dataset]

    def _standardize(self, x):
        x = tf.cast(x, tf.float32)
        x /= 255.
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
            raise ValueError()

        return x

    ## need img shape ##
    def _pad(self, x, shape):
        length = tf.reduce_max(shape)
        paddings = [
            [(length-shape[0])//2, length-((length-shape[0])//2+shape[0])], 
            [(length-shape[1])//2, length-((length-shape[1])//2+shape[1])], [0, 0]]
        x = tf.pad(x, paddings, 'CONSTANT', constant_values=0)
        return x

    def _random_crop(self, x, shape):
        begin, size, bboxes = tf.image.sample_distorted_bounding_box(
            shape, [[[0, 0, 1, 1]]], min_object_covered=.5)
        x = tf.slice(x, begin, size)
        return x

    def _center_crop(self, x, shape):
        length = tf.reduce_min(shape[:2])
        x = tf.image.crop_to_bounding_box(x, (shape[0]-length)//2, (shape[1]-length)//2, length, length)
        return x
    
    ####################

    def _resize(self, x):
        x = tf.image.resize(x, (self.args.img_size, self.args.img_size), tf.image.ResizeMethod.BICUBIC)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _resize_crop_pad(self, x):
        x = tf.image.resize_with_crop_or_pad(x, self.args.img_size, self.args.img_size)
        return x

    def _rotate(self, x):
        x = tfa.image.rotate(x, tf.random.uniform(minval=-self.args.angle, maxval=self.args.angle, shape=[])*np.pi/180, interpolation='BILINEAR')
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _noise(self, x):
        x = tf.cast(x, tf.float32)
        x += tf.random.normal(
            stddev=self.args.noise, 
            shape=(self.args.img_size, self.args.img_size, self.args.img_channel)) * \
                tf.constant(127, tf.float32)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _contrast(self, x):
        x = tf.image.random_contrast(x, lower=1-self.args.contrast, upper=1+self.args.contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _brightness(self, x):
        x = tf.image.random_brightness(x, max_delta=self.args.brightness)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _saturation(self, x):
        x = tf.image.random_saturation(x, lower=1-self.args.contrast, upper=1+self.args.contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _hue(self, x):
        x = tf.image.random_hue(x, max_delta=self.args.hue)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _gray(self, x):
        if tf.random.uniform([]) < .2:
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
        return x

    def _verticalflip(self, x):
        x = tf.image.random_flip_up_down(x)
        return x

    def _horizontalflip(self, x):
        x = tf.image.random_flip_left_right(x)
        return x


class WeakAugment(Augment):
    def __init__(self, args, mode):
        super().__init__(args, mode)
        self.augment_list = []
        if self.mode == 'train':
            if self.args.crop:
                self.augment_list.append(self._random_crop)
            else:
                self.augment_list.append(self._pad)
            
            self.augment_list.append(self._resize)
            
            if self.args.angle:
                self.augment_list.append(self._rotate)
            
            if self.args.vflip:
                self.augment_list.append(self._verticalflip)

            if self.args.hflip:
                self.augment_list.append(self._horizontalflip)

            if self.args.brightness:
                self.augment_list.append(self._brightness)

            if self.args.contrast:
                self.augment_list.append(self._contrast)

            if self.args.saturation:
                self.augment_list.append(self._saturation)

            if self.args.hue:
                self.augment_list.append(self._hue)

            if self.args.gray:
                self.augment_list.append(self._gray)

            if self.args.noise:
                self.augment_list.append(self._noise)

        else:
            self.augment_list.append(self._center_crop)
            self.augment_list.append(self._resize)

        self.augment_list.append(self._standardize)