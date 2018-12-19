# OpenVINO-YoloV3
Under construction on December 15, 2018  
**Python demo program is under construction and it will not work.**  
  
Inspired from **https://github.com/mystic123/tensorflow-yolo-v3.git**
  
**【Notice】December 19, 2018 OpenVINO has supported RaspberryPi + NCS2 !!  
https://software.intel.com/en-us/articles/OpenVINO-RelNotes#inpage-nav-2-2**  
  
**<Intel Core i7-8750H, CPU Only, 4 FPS - 5 FPS>**  
[<img src="media/01.gif" width=80%>](https://youtu.be/vOcj_3ByK68)  
## CPP Version YoloV3 (Full size YoloV3 / Dec 16, 2018 Operation confirmed)
**[cpp version is here](cpp)** "cpp/object_detection_demo_yolov3_async"
  
## Environment

- LattePanda Alpha (Intel 7th Core m3-7y30) or LaptopPC (Intel 8th Core i7-8750H)
- Ubuntu 16.04 x86_64
- OpenVINO toolkit 2018 R4 (2018.4.420)
- Python 3.5
- OpenCV 3.4.3
- Tensorflow v1.11.0 or Tensorflow-GPU v1.11.0 (pip install)
- YoloV3 (MS-COCO)
- USB Camera (PlaystationEye) / Movie file (mp4)
  
## OpenVINO Supported Layers (As of Dec 16, 2018)
<table><tbody></tbody><thead><tr><th>Layers</th><th>GPU</th><th>CPU</th><th>MYRIAD</th><th>GNA</th><th>FPGA</th><th>ShapeInfer</th></tr></thead><tbody><tr><td>Activation-Clamp</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Activation-ELU</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Activation-Leaky ReLU</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Activation-PReLU</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Activation-ReLU</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Activation-ReLU6</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Activation-Sigmoid/Logistic</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Activation-TanH</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>ArgMax</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>BatchNormalization</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Concat</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Const</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td></tr><tr><td>Convolution-Dilated</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Convolution-Grouped</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Convolution-Ordinary</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Crop</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>CTCGreedyDecoder</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Deconvolution</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>DetectionOutput</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Eltwise-Max</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Eltwise-Mul</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Eltwise-Sum</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Flatten</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>FullyConnected (Inner Product)</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>GRN</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Interp</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>LRN (Norm)</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Memory</td><td>Not Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>MVN</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Normalize</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Permute</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Pooling(AVG,MAX)</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Power</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>PriorBox</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>PriorBoxClustered</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Proposal</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>PSROIPooling</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>RegionYolo</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>ReorgYolo</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Resample</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Reshape</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>ROIPooling</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Scale</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td></tr><tr><td>ScaleShift</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>SimplerNMS</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Slice</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>SoftMax</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>SpatialTransformer</td><td>Not Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Split</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Supported</td></tr><tr><td>Tile</td><td>Supported</td><td>Supported</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Supported</td></tr><tr><td>Unpooling</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td></tr><tr><td>Upsampling</td><td>Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td><td>Not Supported</td></tr></tbody></table><hr />
  
# How to install Bazel (version 0.17.2, x86_64 only)
### 1. Bazel introduction command
```bash
$ cd ~
$ curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1dvR3pdM6vtkTWqeR-DpgVUoDV0EYWil5" > /dev/null
$ CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
$ curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1dvR3pdM6vtkTWqeR-DpgVUoDV0EYWil5" -o bazel
$ sudo cp ./bazel /usr/local/bin
$ rm ./bazel
```
### 2. Supplementary information
**https://github.com/PINTO0309/Bazel_bin.git**

# How to check the graph structure of a ".pb" file [Part.1]
Simple structure analysis.
### 1. Build and run graph structure analysis program
```bash
$ cd ~
$ git clone -b v1.11.0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.11.0
$ bazel build tensorflow/tools/graph_transforms:summarize_graph
$ bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=xxxx.pb
```
### 2. Sample of display result
YoloV3
```bash
Found 1 possible inputs: (name=inputs, type=float(1), shape=[?,416,416,3]) 
No variables spotted.
Found 1 possible outputs: (name=output_boxes, op=ConcatV2) 
Found 62002034 (62.00M) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 536 Const, 372 Identity, 87 Mul, 75 Conv2D, 72 FusedBatchNorm, 72 Maximum, 28 Add, \
24 Reshape, 14 ConcatV2, 9 Sigmoid, 6 Tile, 6 Range, 5 Pad, 4 SplitV, 3 Pack, 3 RealDiv, 3 Fill, \
3 Exp, 3 BiasAdd, 2 ResizeNearestNeighbor, 2 Sub, 1 Placeholder
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- \
--graph=/home/b920405/git/OpenVINO-YoloV3/pbmodels/frozen_yolo_v3.pb \
--show_flops \
--input_layer=inputs \
--input_layer_type=float \
--input_layer_shape=-1,416,416,3 \
--output_layer=output_boxes
```
tiny-YoloV3
```bash
Found 1 possible inputs: (name=inputs, type=float(1), shape=[?,416,416,3]) 
No variables spotted.
Found 1 possible outputs: (name=output_boxes, op=ConcatV2) 
Found 8858858 (8.86M) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 134 Const, 63 Identity, 21 Mul, 16 Reshape, 13 Conv2D, 11 FusedBatchNorm, 11 Maximum, \
10 ConcatV2, 6 Sigmoid, 6 MaxPool, 4 Tile, 4 Add, 4 Range, 3 RealDiv, 3 SplitV, 2 Pack, 2 Fill, \
2 Exp, 2 Sub, 2 BiasAdd, 1 Placeholder, 1 ResizeNearestNeighbor
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- \
--graph=/home/b920405/git/OpenVINO-YoloV3/pbmodels/frozen_tiny_yolo_v3.pb \
--show_flops \
--input_layer=inputs \
--input_layer_type=float \
--input_layer_shape=-1,416,416,3 \
--output_layer=output_boxes
```

# How to check the graph structure of a ".pb" file [Part.2]
Convert to text format.
### 1. Run graph structure analysis program
```bash
$ python3 tfconverter.py
### ".pbtxt" in ProtocolBuffer format is output.
### The size of the generated text file is huge.
```

# How to check the graph structure of a ".pb" file [Part.3]
Use Tensorboard.
### 1. Build Tensorboard
```bash
$ cd ~
$ git clone -b v1.11.0 https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout -b v1.11.0
$ bazel build tensorflow/tensorboard:tensorboard
```
### 2. Run log output program for Tensorboard
```python
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename ="xxxx.pb"
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    LOGDIR="path/to/logs"
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
```
### 3. Starting Tensorboard
```bash
$ bazel-bin/tensorflow/tensorboard/tensorboard --logdir=path/to/logs
```
### 4. Display of Tensorboard
Access `http://localhost:6006` from the browser.
  
# Neural Compute Stick 2
**https://ncsforum.movidius.com/discussion/1302/intel-neural-compute-stick-2-information**
