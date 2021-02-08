import tensorflow as tf

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers


INPUT_SHAPE = (512, 1024, 3)


def ge_layer(x_in, c, e=6, stride=1):
    x = layers.Conv2D(filters=c, kernel_size=(3,3), padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if stride == 2:
        x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3,3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        y = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3,3), strides=2, padding='same')(x_in)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(filters=c, kernel_size=(1,1), padding='same')(y)
        y = layers.BatchNormalization()(y)
    else:
        y = x_in
        
    x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=c, kernel_size=(1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([x, y])
    x = layers.Activation('relu')(x)
    return x


def stem(x_in, c):
    x = layers.Conv2D(filters=c, kernel_size=(3,3), strides=2, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x_split = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters=c // 2, kernel_size=(1,1), padding='same')(x_split)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters=c, kernel_size=(3,3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    y = layers.MaxPooling2D()(x_split)
    
    x = layers.Concatenate()([x, y])
    x = layers.Conv2D(filters=c, kernel_size=(3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x


def detail_conv2d(x_in, c, stride=1):
    x = layers.Conv2D(filters=c, kernel_size=(3,3), strides=stride, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x


def context_embedding(x_in, c):
    x = layers.GlobalAveragePooling2D()(x_in)
    x = layers.BatchNormalization()(x)
    
    x = layers.Reshape((1,1,c))(x)
    
    x = layers.Conv2D(filters=c, kernel_size=(1,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # broadcast
    #x = tf.broadcast_to(x, tf.shape(x_in))
    x = layers.UpSampling2D((16,32))(x)
    
    x = layers.Add()([x, x_in])
    x = layers.Conv2D(filters=c, kernel_size=(3,3), padding='same')(x)
    return x


def bilateral_guided_aggregation(detail, semantic, c):
    # detail branch
    detail_a = layers.DepthwiseConv2D(kernel_size=(3,3), padding='same')(detail)
    detail_a = layers.BatchNormalization()(detail_a)
    
    detail_a = layers.Conv2D(filters=c, kernel_size=(1,1), padding='same')(detail_a)
    
    detail_b = layers.Conv2D(filters=c, kernel_size=(3,3), strides=2, padding='same')(detail)
    detail_b = layers.BatchNormalization()(detail_b)
    
    detail_b = layers.AveragePooling2D((3,3), strides=2, padding='same')(detail_b)
    
    # semantic branch
    semantic_a = layers.DepthwiseConv2D(kernel_size=(3,3), padding='same')(semantic)
    semantic_a = layers.BatchNormalization()(semantic_a)
    
    semantic_a = layers.Conv2D(filters=c, kernel_size=(1,1), padding='same')(semantic_a)
    semantic_a = layers.Activation('sigmoid')(semantic_a)
    
    semantic_b = layers.Conv2D(filters=c, kernel_size=(3,3), padding='same')(semantic)
    semantic_b = layers.BatchNormalization()(semantic_b)
    
    semantic_b = layers.UpSampling2D((4,4), interpolation='bilinear')(semantic_b)
    semantic_b = layers.Activation('sigmoid')(semantic_b)
    
    # combining
    detail = layers.Multiply()([detail_a, semantic_b])
    semantic = layers.Multiply()([semantic_a, detail_b])
    
    # this layer is not mentioned in the paper !?
    #semantic = layers.UpSampling2D((4,4))(semantic)
    semantic = layers.UpSampling2D((4,4), interpolation='bilinear')(semantic)
    
    x = layers.Add()([detail, semantic])
    x = layers.Conv2D(filters=c, kernel_size=(3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    return x


def seg_head(x_in, c_t, s, n):
    x = layers.Conv2D(filters=c_t, kernel_size=(3,3), padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters=n, kernel_size=(3,3), padding='same')(x)
    x = layers.UpSampling2D((s,s), interpolation='bilinear')(x)
    
    return x


def bisenetv2(num_classes=2, out_scale=2, c_t=128):
    x_in = layers.Input(INPUT_SHAPE)

    # semantic branch
    x = stem(x_in, 16)
    x = ge_layer(x, 32, stride=2)
    x = ge_layer(x, 32, stride=1)

    x = ge_layer(x, 64, stride=2)
    x = ge_layer(x, 64, stride=1)

    x = ge_layer(x, 128, stride=2)

    x = ge_layer(x, 128, stride=1)
    x = ge_layer(x, 128, stride=1)
    x = ge_layer(x, 128, stride=1)

    x = context_embedding(x, 128)

    # detail branch
    y = detail_conv2d(x_in, 64, stride=2)
    y = detail_conv2d(y, 64, stride=1)

    y = detail_conv2d(y, 64, stride=2)
    y = detail_conv2d(y, 64, stride=1)
    y = detail_conv2d(y, 64, stride=1)

    y = detail_conv2d(y, 128, stride=2)
    y = detail_conv2d(y, 128, stride=1)
    y = detail_conv2d(y, 128, stride=1)

    x = bilateral_guided_aggregation(y, x, 128)

    x = seg_head(x, c_t, out_scale, num_classes)
    
    model = models.Model(inputs=[x_in], outputs=[x])

    return model


def bisenetv2_compiled(**kwargs):
    model = bisenetv2(**kwargs)
    model.compile(optimizers.SGD(momentum=0.9), 
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model


def bisenetv2_output_shape(num_classes, scale):
    return ((INPUT_SHAPE[0] // 8) * scale, 
            (INPUT_SHAPE[1] // 8) * scale, 
            num_classes)
