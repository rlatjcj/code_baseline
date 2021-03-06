import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K

def crossentropy(args):
    def _loss(y_true, y_pred):
        if args.classes == 1:
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        else:
            return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return _loss

def focal(args, alpha=.25, gamma=2.):
    def _loss(y_true, y_pred):
        if args.classes == 1:
            return tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred, alpha=alpha, gamma=gamma)
        else:
            y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon())
            ce = -y_true * tf.math.log(y_pred)
            loss = alpha * tf.math.pow(1-y_pred, gamma) * ce
            return tf.reduce_sum(loss, axis=-1)
    return _loss

def dice_loss(args):
    def _loss(y_true, y_pred, smooth=1.):
        loss = 0.
        y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
        for num_label in range(args.classes):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - loss/args.classes
    return _loss

def dice_loss_wo(args):
    def _loss(y_true, y_pred, smooth=1.):
        loss = 0.
        y_pred = K.clip(y_pred, K.epsilon(), 1.-K.epsilon())
        if args.classes > 1:
            for num_label in range(1, args.classes):
                y_true_f = K.flatten(y_true[...,num_label])
                y_pred_f = K.flatten(y_pred[...,num_label])
                intersection = K.sum(y_true_f * y_pred_f)
                loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
            return 1 - loss/(args.classes-1)
        else:
            raise ValueError('Model must have multi classes output.')
    return _loss

def ce_dice_loss(args):
    def _loss(y_true, y_pred):
        return crossentropy(args)(y_true, y_pred) + dice_loss(args)(y_true, y_pred)
    return _loss