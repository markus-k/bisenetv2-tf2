import tensorflow as tf

from model import INPUT_SHAPE


def class_map_road(seg):
    # map class 0=anything, 1=road
    return tf.where(seg == 7, [0,1], [1,0])


def cityscapes_prep(output_shape, class_map_func=None):
    def prep_map(sample):
        img = sample['image_left']
        seg = sample['segmentation_label']

        img /= 255

        img = tf.image.resize(img, INPUT_SHAPE[0:2])
        seg = tf.image.resize(seg, output_shape[0:2])

        if class_map_func:
            seg = class_map_func(seg)
        else:
            seg = tf.keras.utils.to_categorical(seg, num_classes=output_shape[-1])
    
        return img, seg
    
    return prep_map
