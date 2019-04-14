# Object Detection YOLO V3 Demo, Async API Performance Showcase

This demo showcases Object Detection with YOLO V3 and Async API.
  
Other demo objectives are:
* Video as input support via OpenCV
* Visualization of the resulting bounding boxes and text labels (from the `.labels` file) or class number (if no file is provided)
* OpenCV provides resulting bounding boxes, labels, and other information.
You can copy and paste this code without pulling Inference Engine samples helpers into your application
* Demonstration of the Async API in action. For this, the demo features two modes toggled by the **Tab** key:
    -  Old-style "Sync" way, where the frame captured with OpenCV executes back-to-back with the Detection
    -  Truly "Async" way, where the detection is performed on a current frame, while OpenCV captures the next frame

### How It Works

On the start-up, the application reads command-line parameters and loads a network to the Inference
Engine. Upon getting a frame from the OpenCV VideoCapture, it performs inference and displays the results.

## Recompile
1. Follow the procedure described at the following URL and prepare the environment.  
**[README.md#1-work-with-laptoppc-ubuntu-1604](https://github.com/PINTO0309/OpenVINO-YoloV3#1-work-with-laptoppc-ubuntu-1604)**
2. In order to correspond to your machine's architecture, you need to recompile the binary by executing the following command.
```bash
$ cd ~/OpenVINO-YoloV3/cpp
$ sudo cp main.cpp /opt/intel/openvino/deployment_tools/inference_engine/samples/object_detection_demo_yolov3_async
$ sudo cp object_detection_demo_yolov3_async.hpp /opt/intel/openvino/deployment_tools/inference_engine/samples/object_detection_demo_yolov3_async
$ cd /opt/intel/openvino/deployment_tools/inference_engine/samples
$ sudo ./build_samples.sh
$ cd ~/OpenVINO-YoloV3/cpp
$ cp $HOME/inference_engine_samples_build/intel64/Release/object_detection_demo_yolov3_async/object_detection_demo_yolov3_async .
```

## Running

Running the application with the <code>-h</code> option yields the following usage message:
```sh
./object_detection_demo_yolov3_async -h
InferenceEngine: 
    API version ............ <version>
    Build .................. <number>

object_detection_demo_yolov3_async [OPTION]
Options:

    -h                        Print a usage message.
    -i "<path>"               Required. Path to a video file (specify "cam0" to work with camera).
    -m "<path>"               Required. Path to an .xml file with a trained model.
      -l "<absolute_path>"    Optional. Required for CPU custom layers.Absolute path to a shared library with the layers implementation.
          Or
      -c "<absolute_path>"    Optional. Required for GPU custom kernels.Absolute path to the .xml file with the kernels description.
    -d "<device>"             Optional. Specify a target device to infer on (CPU, GPU). The demo will look for a suitable plugin for the specified device
    -pc                       Optional. Enable per-layer performance report.
    -r                        Optional. Output inference results raw values showing.
    -t                        Optional. Probability threshold for detections.
    -iou_t                    Optional. Filtering intersection over union threshold for overlapping boxes.
    -auto_resize              Optional. Enable resizable input with support of ROI crop and auto resize.
```

Running the application with the empty list of options yields the usage message given above and an error message.
You can use the following command to do inference on GPU with a pre-trained object detection model:
### CPU + USB Camera Mode + Full size YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/YoloV3/FP32/frozen_yolo_v3.xml -d CPU
```
### MYRIAD + USB Camera Mode + Full size YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/tiny-YoloV3/FP16/frozen_tiny_yolo_v3.xml -d MYRIAD
```
### CPU + USB Camera Mode + tiny-YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/tiny-YoloV3/FP16/frozen_yolo_v3.xml -d CPU
```
### MYRIAD + USB Camera Mode + tiny-YoloV3 (Selectable cam0/cam1/cam2)
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i cam0 -m ../lrmodels/tiny-YoloV3/FP16/frozen_tiny_yolo_v3.xml -d MYRIAD -t 0.2
```
### Movie File Mode
```bash
$ cd cpp
$ ./object_detection_demo_yolov3_async -i <path_to_video>/inputVideo.mp4 -m <path_to_model>/frozen_yolo_v3.xml -l ../lib/libcpu_extension.so -d CPU
```
**NOTE**: Public models should be first converted to the Inference Engine format (`*.xml` + `*.bin`) using the Model Optimizer tool.

The only GUI knob is to use **Tab** to switch between the synchronized execution and the true Async mode.

### Demo Output

The demo uses OpenCV to display the resulting frame with detections (rendered as bounding boxes and labels, if provided).
In the default mode, the demo reports:
* **OpenCV time**: frame decoding + time to render the bounding boxes, labels, and to display the results.
* **Detection time**: inference time for the object detection network. It is reported in the Sync mode only.
* **Wallclock time**, which is combined application-level performance.
