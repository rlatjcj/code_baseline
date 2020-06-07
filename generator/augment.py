import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


CROP_PADDING = 32

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
            image_size=shape, 
            bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]), 
            min_object_covered=.1,
            aspect_ratio_range=(3. / 4., 4. / 3.),
            area_range=(0.08, 1.0),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)

        x = tf.slice(x, begin, size)
        return x

    def _center_crop(self, x, shape):
        image_height = shape[0]
        image_width = shape[1]
        padded_center_crop_size = tf.cast(
            ((self.args.img_size/(self.args.img_size+CROP_PADDING)) * 
                tf.cast(tf.math.minimum(image_height, image_width), tf.float32)),
            tf.int32)

        offset_height = ((image_height - padded_center_crop_size) + 1) // 2
        offset_width = ((image_width - padded_center_crop_size) + 1) // 2
        x = tf.image.crop_to_bounding_box(
            x, offset_height, offset_width, padded_center_crop_size, padded_center_crop_size)
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
        angle = tf.random.uniform(minval=-self.args.angle, maxval=self.args.angle, shape=[])*np.pi/180
        x = tfa.image.rotate(x, angle, interpolation='BILINEAR')
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

    def _contrast(self, x, contrast=None):
        contrast = contrast or self.args.contrast
        x = tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _brightness(self, x, brightness=None):
        brightness = brightness or self.args.brightness
        x = tf.image.random_brightness(x, max_delta=brightness)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _saturation(self, x, saturation=None):
        saturation = saturation or self.args.saturation
        x = tf.image.random_saturation(x, lower=1-saturation, upper=1+saturation)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _hue(self, x, hue=None):
        hue = hue or self.args.hue
        x = tf.image.random_hue(x, max_delta=hue)
        x = tf.saturate_cast(x, tf.uint8)
        return x

    def _gray(self, x, p=.2):
        if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
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


class SimAugment(Augment):
    def __init__(self, args, mode):
        super().__init__(args, mode)
        self.augment_list = []
        if self.mode == 'train':
            self.augment_list.append(self._random_crop)
            self.augment_list.append(self._resize)
            self.augment_list.append(self._color_jitter)
            self.augment_list.append(self._gray)
        else:
            self.augment_list.append(self._center_crop)
            self.augment_list.append(self._resize)

        self.augment_list.append(self._standardize)

    def _color_jitter(self, x, p=.8):
        if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)):
            x = self._brightness(x, brightness=.8)
            x = self._contrast(x, contrast=.8)
            x = self._saturation(x, saturation=.8)
            x = self._hue(x, hue=.2)
        return x
