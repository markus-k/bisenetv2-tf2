# BiSeNet V2 TensorFlow 2 implementation

This is a implementation of [BiSeNet V2](https://arxiv.org/pdf/2004.02147.pdf) in TensorFlow 2. It supports quantization and running on a Google EdgeTPU.

## Training

Currently the Notebook `BiSeNetV2-TFKeras.ipynb` is used for training. It has code for training from the Cityscapes dataset. There are also some helper functions available to train from data produced by labelme or our internal training data.

## Model parameters

Input and output shapes can be adjusted when creating the model. Be aware that these have to be compatible with the internal strides of the model, so it's best to stick to a power of 2 in each dimension. The default input resolution is 512x1024, and the default output resolution is 512x1024 (8x upscaled), as used in the original paper.

We currently only train with two output classes (Background, Road), but this can be easily adjusted.

## Quantization

The `quantize.py` script can be used for quantizing a saved `.tf` model. For usage, see `python quantize.py --help`.

The resulting quantized TF-Lite model can then be compiled for EdgeTPUs with the `edgetpu_compiler`.

## Performance

The model currently has 78 operations running on EdgeTPU and 4 operations running on CPU using two subgraphs.

Using 256x512 input and output size with two classes, we achieve about 63 FPS on a PCIe EdgeTPU and 17 FPS on a USB2-connected EdgeTPU on a desktop machine. This can probably be tweaked a little bit further.

## License

The code published in this repository is licensed under the MIT-License.

```
Copyright 2021 Markus Kasten

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
