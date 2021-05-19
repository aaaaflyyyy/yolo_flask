from flask import *

# Reference : https://github.com/ultralytics/yolov3
# Import necessary libraries
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# num of person.
num = [0]

# Initialize the Flask app
app = Flask(__name__)


def detect():
    # '0' for Camera.
    source = '0'
    # model.
    weights = 'best.pt'
    # images size.
    imgsz = 640

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        # Apply NMS
        conf_thres = 0.5
        iou_thres = 0.45
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
            ), dataset.count

            num[0] = 0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():

                    n = (det[:, -1] == c).sum()  # detections per class
                    num[0] = int(n)

                for *xyxy, conf, cls in reversed(det):

                    label = None
                    plot_one_box(xyxy, im0, label=label,
                                 color=colors[int(cls)], line_thickness=3)

            # Stream results
            ret, buffer = cv2.imencode('.jpg', im0)
            im0 = buffer.tobytes()
            # concat frame one by one and show result
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/updata_data', methods=['GET'])
def update_data():
    global num
    return jsonify(num=num)

@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
