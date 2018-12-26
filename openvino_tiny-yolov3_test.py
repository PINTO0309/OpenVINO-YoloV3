import sys
import os
from argparse import ArgumentParser
import numpy as np
import cv2
import time
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile
from openvino.inference_engine import IENetwork, IEPlugin

yolo_scale_13 = 13
yolo_scale_26 = 26

classes = 80
coords = 4
num = 6
mask = [0,1,2]
jitter = 0.3
ignore_thresh = 0.7
truth_thresh = 1
random = 1
anchors = {10,14, 23,27, 37,58, 81,82, 135,169, 344,319}

LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                Sample will look for a suitable plugin for device specified (CPU by default)", default="CPU", type=str)
    return parser


def main_IE_infer():
    camera_width = 320
    camera_height = 240
    m_input_size=416
    fps = ""
    framepos = 0
    frame_count = 0
    vidfps = 0
    skip_frame = 0
    elapsedTime = 0

    args = build_argparser().parse_args()
    model_xml = "lrmodels/tiny-YoloV3/FP32/frozen_tiny_yolo_v3.xml" #<--- CPU
    #model_xml = "lrmodels/tiny-YoloV3/FP16/frozen_tiny_yolo_v3.xml" #<--- MYRIAD
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    #cap = cv2.VideoCapture("data/input/testvideo.mp4")
    #camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #vidfps = int(cap.get(cv2.CAP_PROP_FPS))
    #print("videosFrameCount =", str(frame_count))
    #print("videosFPS =", str(vidfps))

    time.sleep(1)

    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if "CPU" in args.device:
        plugin.add_cpu_extension("lib/libcpu_extension.so")
    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)

    #sys.exit(0)

    while cap.isOpened():
        t1 = time.time()

        ## Uncomment only when playing video files
        #cap.set(cv2.CAP_PROP_POS_FRAMES, framepos)

        ret, image = cap.read()
        if not ret:
            break

        image = cv2.resize(image, (m_input_size, m_input_size), interpolation=cv2.INTER_CUBIC)

        prepimg = image[np.newaxis, :, :, :]
        prepimg = prepimg.transpose((0, 3, 1, 2))  #NHWC to NCHW
        outputs = exec_net.infer(inputs={input_blob: prepimg})

        sys.exit(0)
        break

        outputimg = Image.fromarray(np.uint8(result), mode="P")
        outputimg.putpalette(palette)
        outputimg = outputimg.convert("RGB")
        outputimg = np.asarray(outputimg)
        outputimg = cv2.cvtColor(outputimg, cv2.COLOR_RGB2BGR)
        outputimg = cv2.addWeighted(image, 1.0, outputimg, 0.9, 0)
        outputimg = cv2.resize(outputimg, (camera_width, camera_height))

        cv2.putText(outputimg, fps, (camera_width-180,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.imshow("Result", outputimg)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        elapsedTime = time.time() - t1
        fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

        ## frame skip, video file only
        #skip_frame = int((vidfps - int(1/elapsedTime)) / int(1/elapsedTime))
        #framepos += skip_frame

    cv2.destroyAllWindows()
    del net
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main_IE_infer() or 0)

