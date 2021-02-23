#!/usr/bin/env python3

import time
import argparse
import os
import statistics

import tensorflow as tf
import tflite_runtime.interpreter as tflite

import numpy as np

# Stop claiming CUDA devices!
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


parser = argparse.ArgumentParser(description='EdgeTPU BiSeNetV2 benchmark')
parser.add_argument('model', help='Model path')
parser.add_argument('--device', default='usb', choices=['usb', 'pci', 'cpu'], help='Device to run model on')
parser.add_argument('--count', type=int, default=10, help='Number of invokations')

args = parser.parse_args()

DEVICE = args.device
USE_EDGETPU = (DEVICE != 'cpu')

model_path = args.model

COUNT = args.count

if USE_EDGETPU:
    interpreter = tflite.Interpreter(model_path,
      experimental_delegates=[tflite.load_delegate('libedgetpu.so.1', options={'device': DEVICE})])
else:
    interpreter = tflite.Interpreter(model_path)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SHAPE = input_details[0]['shape'][1:4]
OUTPUT_SHAPE = output_details[0]['shape'][1:4]

print(input_details)
print(output_details)

times = []
times2 = []
first = True

for i in range(COUNT+1):
    shape = (1, *INPUT_SHAPE)
    input_data = np.zeros(shape, dtype=input_details[0]['dtype'])

    s1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    s2 = time.time()
    interpreter.invoke()
    e2 = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    e1 = time.time()

    if not first:
        times.append(e2-s2)
        times2.append(e1-s1)
    else:
        first = False

    print(f'invoke: {e2-s2:.3f}s ({1/(e2-s2):.2f} fps)')
    print(f'invoke+load: {e1-s1:.3f}s ({1/(e1-s1):.2f} fps)')


print()
print('Invoke:')
invoke_avg = sum(times) / len(times)
print(f'Average: {invoke_avg:.3f}s ({1/invoke_avg:.2f} fps)')
print(f'min/max/stdev: {min(times):.03f}/{max(times):.03f}/{statistics.stdev(times):.03f}')

print()
print('Total:')
total_avg = sum(times2) / len(times2)
print(f'Average: {total_avg:.3f}s ({1/total_avg:.2f} fps)')
print(f'min/max/stdev: {min(times2):.03f}/{max(times2):.03f}/{statistics.stdev(times2):.03f}')
