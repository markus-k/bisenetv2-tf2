#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

from data_prep import cityscapes_prep, class_map_road

OUTPUT_SHAPE = (128, 256, 2)

cityscapes = tfds.load('cityscapes/semantic_segmentation')
test_ds = cityscapes['test'].map(cityscapes_prep(OUTPUT_SHAPE, class_map_road))

def representative_dataset():
    for data in test_ds.take(50).batch(1):
        yield [data[0]]
        

converter = tf.lite.TFLiteConverter.from_saved_model('model.tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.experimental_new_converter = False # https://github.com/google-coral/edgetpu/issues/168#issuecomment-656115637
tflite_quant_model = converter.convert()

with open('model_quant.tflite', 'wb') as fd:
    fd.write(tflite_quant_model)