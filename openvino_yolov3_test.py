#
#https://software.intel.com/en-us/articles/OpenVINO-Using-TensorFlow#inpage-nav-8
#

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

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plugin for device specified (CPU by default)", default="CPU", type=str)
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)
    parser.add_argument("-pc", "--performance", help="Enables per-layer performance report", action='store_true')

    return parser


class _model_postprocess():
    def __init__(self):
        graph = tf.Graph()
        f_handle = gfile.FastGFile("pbmodels/frozen_yolo_v3.pb", "rb")
        graph_def = tf.GraphDef.FromString(f_handle.read())
        with graph.as_default():
            #detections: outputs of YOLOV3 detector of shape (?, 10647, (num_classes + 5))
            new_input = tf.placeholder(tf.float32, shape=(1, 10647, 85), name="new_input")
            tf.import_graph_def(graph_def, input_map={"split:0": new_input}, name='')
        self.sess = tf.Session(graph=graph)

    def _post_process(self, detections):
        detected_boxes = self.sess.run("output_boxes:0", feed_dict={"new_input:0": detections})
        return detected_boxes


_post = _model_postprocess()


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
    model_xml = "lrmodels/YoloV3/FP32/frozen_yolo_v3.xml"
    #model_xml = "lrmodels/tiny-YoloV3/FP32/frozen_tiny_yolo_v3.xml"
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
    if args.performance:
        plugin.set_config({"PERF_COUNT": "YES"})
    # Read IR
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob   = next(iter(net.outputs))
    print("input_blob =", input_blob)
    print("out_blob =", out_blob)
    exec_net = plugin.load(network=net)

    while cap.isOpened():
        t1 = time.time()

        # Uncomment only when playing video files
        #cap.set(cv2.CAP_PROP_POS_FRAMES, framepos)

        ret, image = cap.read()
        if not ret:
            break

        #ratio = 1.0 * m_input_size / max(image.shape[0], image.shape[1])
        #shrink_size = (int(ratio * image.shape[1]), int(ratio * image.shape[0]))
        image = cv2.resize(image, (m_input_size, m_input_size), interpolation=cv2.INTER_CUBIC)

        #prepimg = _pre._pre_process(image)
        prepimg = image[np.newaxis, :, :, :]
        print("image =", image.shape)
        print("prepimg =", prepimg.shape)
        prepimg = prepimg.transpose((0, 3, 1, 2))  #NHWC to NCHW
        #res = exec_net.infer(inputs={input_blob: prepimg})
        exec_net.start_async(request_id=0, inputs={input_blob: prepimg})

        if exec_net.requests[0].wait(-1) == 0:
            outputs = exec_net.requests[0].outputs[out_blob]

        #print("len(res) =", len(res))
        print("outputs.shape =", outputs.shape)

        #result = _post._post_process(res)
        #print(result)

        #reslist = list(res.keys())
        #detector/yolo-v3/Conv_6/BiasAdd/YoloRegion
        #detector/yolo-v3/Conv_14/BiasAdd/YoloRegion
        #detector/yolo-v3/Conv_22/BiasAdd/YoloRegion

        #print("res =", reslist)
        #print("res(reslist[0][0]) =", len(res[reslist[0]][0]))
        #print("res(reslist[1][0]) =", len(res[reslist[1]][0]))
        #print("res(reslist[2][0]) =", len(res[reslist[2]][0]))
        #print("res(reslist[0][0][0]) =", len(res[reslist[0]][0][0]))
        #print("res(reslist[1][0][0]) =", len(res[reslist[1]][0][0]))
        #print("res(reslist[2][0][0]) =", len(res[reslist[2]][0][0]))
        #print("res(reslist[0][0][254]) =", len(res[reslist[0]][0][254]))
        #print("res(reslist[1][0][254]) =", len(res[reslist[1]][0][254]))
        #print("res(reslist[2][0][254]) =", len(res[reslist[2]][0][254]))

        #out1 = np.concatenate([res[reslist[0][0]], res[reslist[1][0]], res[reslist[2][0]]], axis=-1)
        #out1 = np.r_[res[reslist[0][0]], res[reslist[1][0]], res[reslist[2][0]]]
        #out2 = np.split(out1, [1, 1, 1, 1, -1], axis=-1)
        #out2 = detections_boxes(detections)
        #print(out2)
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

        # frame skip, video file only
        skip_frame = int((vidfps - int(1/elapsedTime)) / int(1/elapsedTime))
        framepos += skip_frame

    cv2.destroyAllWindows()
    del net
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main_IE_infer() or 0)

