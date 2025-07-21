# Raspberry Pi NCS2 OpenCV Object Detection

This project demonstrates real-time object detection using a Raspberry Pi, PiCamera, Intel Neural Compute Stick 2 (NCS2), and OpenCV's DNN module. It supports SSD, YOLO, TensorFlow, and OpenVINO models.

## Features

- Real-time video capture and inference
- Multiprocessing for efficient frame classification
- Overlay with FPS, detection count, and confidence threshold
- Adjustable confidence threshold via keyboard

## Requirements

- Raspberry Pi (tested on Pi 3/4)
- PiCamera
- Intel NCS2 (Neural Compute Stick 2)
- OpenCV (with DNN and Myriad support)
- Python 3.x
- Required Python packages: `opencv-python`, `numpy`, `picamera`, etc.

## Usage

```bash
python3 main.py -b <model.bin> -x <model.xml> -l <labels.txt>
# or for TensorFlow models:
python3 main.py -pb <model.pb> -pbtxt <model.pbtxt> -l <labels.txt>
```

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

## License

See `LICENSE`.
