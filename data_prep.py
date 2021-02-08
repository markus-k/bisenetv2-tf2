import json
import tensorflow as tf
import numpy as np
import cv2

from model import INPUT_SHAPE


def class_map_road(seg):
    # map class 0=anything, 1=road
    return tf.where(seg == 7, [0, 1.0], [1.0, 0])


def cityscapes_prep(output_shape, input_shape=INPUT_SHAPE, class_map_func=None, float_range=True):
    def prep_map(sample):
        img = sample['image_left']
        seg = sample['segmentation_label']

        if float_range:
            img /= 255

        img = tf.image.resize(img, input_shape[0:2])
        seg = tf.image.resize(seg, output_shape[0:2])

        if callable(class_map_func):
            seg = class_map_func(seg)
        else:
            seg = tf.keras.utils.to_categorical(seg, num_classes=output_shape[-1])

        return img, seg

    return prep_map


def create_labelme_segmentation(contents):
    meta = json.loads(contents.numpy().decode('utf-8'))
    seg = np.zeros((meta['imageHeight'], meta['imageWidth']))

    # TODO: create a good way to have multiple classes
    for shape in meta['shapes']:
        points = np.array(shape['points']).astype(np.int32)
        cv2.fillPoly(seg, [points], (1))

    return seg, meta['imageHeight'], meta['imageWidth']


def labelme_prep(output_shape, input_shape, float_range=True):
    def labelme_map(json_file):
        contents = tf.io.read_file(json_file)
        seg, h, w = tf.py_function(create_labelme_segmentation, [contents], [tf.float32, tf.int32, tf.int32])
        seg = tf.reshape(seg, (h, w, 1))
        seg = tf.image.resize(seg, output_shape[0:2], method='nearest')
        seg = tf.where(seg == 1, [0, 1.0], [1.0, 0])

        jpeg_filename = tf.strings.regex_replace(json_file, '\.json', '.jpg')
        jpeg_contents = tf.io.read_file(jpeg_filename)
        img = tf.io.decode_jpeg(jpeg_contents, channels=3)
        img = tf.image.resize(img, input_shape[0:2])

        if float_range:
            img /= 255

        return img, seg

    return labelme_map


def uwula_prep(output_shape, input_shape, float_range=True):
    road_color = (32,224,224)

    def uwula_map(jpeg_filename):
        jpeg_contents = tf.io.read_file(jpeg_filename)
        img = tf.io.decode_jpeg(jpeg_contents, channels=3)
        img = tf.image.resize(img, input_shape[0:2])

        if float_range:
            img /= 255

        seg_filename = tf.strings.regex_replace(jpeg_filename, '\.jpg', '.png')
        seg_contents = tf.io.read_file(seg_filename)
        seg = tf.io.decode_png(seg_contents, channels=3)
        seg = tf.image.resize(seg, output_shape[0:2], method='nearest')

        seg_i = tf.math.reduce_sum(seg, axis=-1)
        seg_i = tf.expand_dims(seg_i, axis=-1)
        seg_i = tf.where(seg_i > 0, [0.0, 1.0], [1.0, 0.0])

        return img, seg_i

    return uwula_map
