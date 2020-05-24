import tensorflow as tf
import tensorflow.keras.backend as K

def top_k_accuracy(k=5):
    def _acc(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k)

    _acc.__name__ = 'top_{}_acc'.format(k)
    return _acc

def dice(y_true, y_pred, classes=None, smooth=1.):
    loss = 0.
    classes =  classes if classes else K.int_shape(y_pred)[-1]
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    
    if classes > 1:
        for num_label in range(classes):
            y_true_f = K.flatten(y_true[...,num_label])
            y_pred_f = K.flatten(y_pred[...,num_label])
            intersection = K.sum(y_true_f * y_pred_f)
            loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return loss / classes
    else:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        loss += (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return loss

def dice_class(c=1):
    def _dice(y_true, y_pred):
        return dice(y_true[...,c], y_pred[...,c], classes=1)

    _dice.__name__ = 'dice_class{}'.format(c)
    return _dice