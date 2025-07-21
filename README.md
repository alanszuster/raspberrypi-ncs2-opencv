# Raspberry Pi & Laptop Object Detection (YOLOv3, OpenVINO, SSD)

This project demonstrates real-time object detection using two main scripts:

- **main.py**: Designed for Raspberry Pi with PiCamera and optional Intel Neural Compute Stick 2 (NCS2) acceleration. Supports OpenVINO, YOLOv3, and TensorFlow SSD models.
- **main_laptop.py**: Designed for laptops/PCs using a webcam or video file. Runs on CPU, supports YOLOv3, OpenVINO, and TensorFlow SSD models.

Both scripts use OpenCV's DNN module for inference. Example YOLOv3 model files (`yolo3.cfg`, `yolov3.weights`) are provided in the `/model` folder.

## Features

- Real-time video capture and inference
- Multiprocessing for efficient frame classification (main.py)
- Overlay with FPS, detection count, and confidence threshold
- Adjustable confidence threshold via keyboard
- Example YOLOv3 model included in `/model`

## Requirements

- Raspberry Pi (tested on Pi 3/4) for `main.py`
- PiCamera for live capture (Raspberry Pi)
- Intel NCS2 (Neural Compute Stick 2) for hardware acceleration (optional, Raspberry Pi)
- Laptop/PC with webcam or video file for `main_laptop.py`
- OpenCV (with DNN and Myriad support)
- Python 3.x
- Required Python packages: `opencv-python`, `numpy`, `picamera` (for Pi), etc.

## Usage

### 1. Raspberry Pi + PiCamera (main.py)

**YOLOv3 (recommended, example model in `/model`):**

```bash
python3 main.py -b model/yolov3.weights -x model/yolo3.cfg -l labels.txt
```

**OpenVINO/Intel NCS2:**

```bash
python3 main.py -b <model.bin> -x <model.xml> -l labels.txt
```

**TensorFlow SSD:**

```bash
python3 main.py -pb <model.pb> -pbtxt <model.pbtxt> -l labels.txt
```

### 2. Laptop/PC + Webcam/Video (main_laptop.py)

**YOLOv3 (example model in `/model`):**

```bash
python3 main_laptop.py -b model/yolov3.weights -x model/yolo3.cfg -l labels.txt
```

**OpenVINO/CPU:**

```bash
python3 main_laptop.py -b <model.bin> -x <model.xml> -l labels.txt
```

**TensorFlow SSD:**

```bash
python3 main_laptop.py -pb <model.pb> -pbtxt <model.pbtxt> -l labels.txt
```

**To use a video file or image on laptop:**

```bash
python3 main_laptop.py -b model/yolov3.weights -x model/yolo3.cfg -l labels.txt -i <video_or_image_path>
```

## Arguments (common to both scripts)

- `-b`, `--bin`: Path to weights file (YOLO .weights or OpenVINO .bin)
- `-x`, `--xml`: Path to config file (YOLO .cfg or OpenVINO .xml)
- `-pb`, `--protobox`: Path to TensorFlow .pb file
- `-pbtxt`, `--protoboxtxt`: Path to TensorFlow .pbtxt file
- `-l`, `--labels`: Path to labels file (one label per line)
- `-ct`, `--conf_threshold`: Confidence threshold (default: 0.5)
- `-i`, `--input`: (main_laptop.py only) Path to image or video file

## Keyboard Controls

- `q`: Quit
- Up arrow: Increase confidence threshold
- Down arrow: Decrease confidence threshold

## Workflow & Details

- **main.py**: Uses PiCamera for live capture on Raspberry Pi. Supports NCS2 acceleration (OpenVINO) and multiprocessing for fast frame processing. Recommended for embedded/edge deployments.
- **main_laptop.py**: Uses webcam or video/image file for input. Runs on CPU, suitable for quick testing and development on any PC/laptop.
- **Example model**: `/model/yolo3.cfg` and `/model/yolov3.weights` are provided for YOLOv3 object detection. Use with `labels.txt` (COCO classes).
- **Custom models**: You can use your own YOLOv3, OpenVINO, or TensorFlow SSD models by providing the appropriate files and arguments.
- **Labels**: `labels.txt` must contain one class name per line, matching the model's output.
- **Performance**: NCS2 (OpenVINO) provides hardware acceleration on Raspberry Pi. YOLOv3 is robust and works well on both Pi and laptop.

## Arguments

- `-b`, `--bin`: Path to OpenVINO .bin file
- `-x`, `--xml`: Path to OpenVINO .xml file
- `-pb`, `--protobox`: Path to TensorFlow .pb file
- `-pbtxt`, `--protoboxtxt`: Path to TensorFlow .pbtxt file
- `-l`, `--labels`: Path to labels file (one label per line)
- `-ct`, `--conf_threshold`: Confidence threshold (default: 0.5)

## Keyboard Controls

- `q`: Quit
- Up arrow: Increase confidence threshold
- Down arrow: Decrease confidence threshold
