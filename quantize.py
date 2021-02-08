#!/usr/bin/env python3
import argparse
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from model import ArgmaxMeanIOU
from data_prep import cityscapes_prep, class_map_road


parser = argparse.ArgumentParser(description='Quantize BiSeNetV2 .tf model')
parser.add_argument('tf_model')
parser.add_argument('quant_model')
parser.add_argument('--cpu', action='store_true')

args = parser.parse_args()

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print('Loading model...')
model = tf.keras.models.load_model(args.tf_model, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
INPUT_SHAPE = model.input_shape[1:]
OUTPUT_SHAPE = model.output_shape[1:]

print(f'Input shape:  {INPUT_SHAPE}')
print(f'Output shape: {OUTPUT_SHAPE}')

cityscapes = tfds.load('cityscapes/semantic_segmentation')
test_ds = cityscapes['test'].map(cityscapes_prep(OUTPUT_SHAPE, class_map_road, input_shape=INPUT_SHAPE))

def representative_dataset():
    for data in test_ds.take(50).batch(1):
        yield [data[0]]
        

converter = tf.lite.TFLiteConverter.from_saved_model(args.tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.experimental_new_converter = False # https://github.com/google-coral/edgetpu/issues/168#issuecomment-656115637
tflite_quant_model = converter.convert()

with open(args.quant_model, 'wb') as fd:
    fd.write(tflite_quant_model)

print('Model successfully converted.')
