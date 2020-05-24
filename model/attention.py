import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D

from model.layers import _normalization
from model.layers import _activation


def _se_block(inputs, se_ratio=16, name=None):
    channel = K.int_shape(inputs)[-1]
    
    x = GlobalAveragePooling2D(name=name+'_gap')(inputs)
    x = Dense(channel//se_ratio, name=name+'_fc1')(x)
    x = _activation(x, activation='relu', name=name+'_relu')
    x = Dense(channel, name=name+'_fc2')(x)
    x = _activation(x, activation='sigmoid', name=name+'_sigmoid')
    x = Reshape([1, 1, channel], name=name+'_reshape')(x)
    x = Multiply(name=name+'_multiply')([inputs, x])
    return x


def _cbam_block(inputs, ratio=8, name=None):
    def _channel_attention(_inputs, cbam_ratio=8):
        channel = K.int_shape(_inputs)[-1]
        
        shared_layer_one = Dense(channel // cbam_ratio, 
                                 activation='relu', 
                                 kernel_initializer='he_normal', 
                                 use_bias=True, 
                                 bias_initializer='zeros',
                                 name=name+'_sl1')
        shared_layer_two = Dense(channel, 
                                 kernel_initializer='he_normal', 
                                 use_bias=True, 
                                 bias_initializer='zeros',
                                 name=name+'_sl2')
        
        avg_pool = GlobalAveragePooling2D(name=name+'_gap')(_inputs)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)
        
        max_pool = GlobalMaxPooling2D(name=name+'_gmp')(_inputs)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)
        
        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = _activation(cbam_feature, activation='sigmoid', name=name+'_sigmoid')
        
        return Multiply()([_inputs, cbam_feature])

    def _spatial_attention(_inputs, kernel_size=7):
        channel = K.int_shape(_inputs)[-1]
        cbam_feature = _inputs
        
        avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True), name=name+'_sap')(cbam_feature)
        max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True), name=name+'_smp')(cbam_feature)
        concat = Concatenate()([avg_pool, max_pool])
        cbam_feature = Conv2D(1, kernel_size, 
                              strides=1, 
                              padding='same', 
                              activation='sigmoid', 
                              kernel_initializer='he_normal', 
                              use_bias=False,
                              name=name+'_conv')(concat)
            
        return Multiply()([_inputs, cbam_feature])

    x = _channel_attention(inputs, ratio)
    x = _spatial_attention(x)
    return x