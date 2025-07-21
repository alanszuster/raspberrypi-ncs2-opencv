import cv2
import time
import numpy as np
import argparse
from sys import getsizeof

def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Laptop Demo')
    parser.add_argument('-b', '--bin', help='Path to weights file (OpenVINO or YOLO)')
    parser.add_argument('-x', '--xml', help='Path to config file (OpenVINO .xml or YOLO .cfg)')
    parser.add_argument('-l', '--labels', help='Path to labels file (one label per line)')
    parser.add_argument('-pb', '--protobox', help='Path to TensorFlow .pb file')
    parser.add_argument('-pbtxt', '--protoboxtxt', help='Path to TensorFlow .pbtxt file')
    parser.add_argument('-ct', '--conf_threshold', default=0.5, type=float)
    parser.add_argument('-i', '--input', help='Path to image or video file (optional)')
    return parser.parse_args()


def load_network(args):
    if args.bin and args.xml:
        if args.bin.endswith('.weights') and args.xml.endswith('.cfg'):
            print("[INFO] YOLO/Darknet format")
            net = cv2.dnn.readNet(args.bin, args.xml)
            model_type = 'yolo'
        else:
            print("[INFO] OpenVINO format")
            net = cv2.dnn.readNet(args.xml, args.bin)
            model_type = 'openvino'
    elif args.protobox and args.protoboxtxt:
        print("[INFO] Tensorflow format")
        net = cv2.dnn.readNetFromTensorflow(args.protobox, args.protoboxtxt)
        model_type = 'ssd'
    else:
        raise ValueError("No model files provided or missing config.")
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, model_type


def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = [x.strip() for x in f]
    print(labels)
    return labels


def draw_detections(frame, detections, labels, confThreshold, font):
    for detection in detections:
        objID, confidence, xmin, ymin, xmax, ymax = detection
        if confidence > confThreshold:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))
            cv2.rectangle(frame, (xmin-1, ymin-1), (xmin+70, ymin-10), (0,255,255), -1)
            label_text = f' {labels[objID]} {round(confidence,2)}'
            cv2.putText(frame, label_text, (xmin, ymin-2), font, 0.3, (0,0,0), 1, cv2.LINE_AA)


def draw_overlay(frame, confThreshold, fps, detections, font):
    cv2.rectangle(frame, (0, 0), (150, 15), (0,0,0), -1)
    cv2.putText(frame, f'Threshold: {round(confThreshold,1)}', (10, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'FPS: {fps}', (80, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, frame.shape[0]-35), (200, frame.shape[0]), (0,0,0), -1)
    cv2.putText(frame, f'Positive detections: {detections}', (10, frame.shape[0]-15), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)


def classify_frame(net, frame, model_type):
    if model_type == 'yolo':
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)
        frame_height, frame_width = frame.shape[:2]
        data_out = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            box = boxes[i]
            x, y, w, h = box
            data_out.append((class_ids[i], confidences[i], x, y, x + w, y + h))
        print("YOLO detections:", data_out)
        return data_out
    else:
        blob = cv2.dnn.blobFromImage(frame, 0.007843, size=(300, 300),
                                     mean=(127.5,127.5,127.5), swapRB=False, crop=False)
        net.setInput(blob)
        out = net.forward()
        print("SSD raw output shape:", out.shape)
        data_out = []
        for detection in out.reshape(-1, 7):
            obj_type = int(detection[1]-1)
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            if confidence > 0:
                data_out.append((obj_type, confidence, xmin, ymin, xmax, ymax))
        print("SSD detections:", data_out)
        return data_out


def main():
    args = parse_args()
    confThreshold = args.conf_threshold
    font = cv2.FONT_HERSHEY_SIMPLEX
    detections_count = 0
    fps = 0.0

    net, model_type = load_network(args)
    labels = load_labels(args.labels)

    # Use webcam or file
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(0)

    print("[INFO] starting capture...")
    timer1 = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = classify_frame(net, frame, model_type)
        print("Detections in frame:", detections)
        draw_detections(frame, detections, labels, confThreshold, font)
        detections_count = sum(1 for d in detections if d[1] > confThreshold)

        frames += 1
        if frames >= 1:
            end1 = time.time()
            t1secs = end1 - timer1
            fps = round(frames / t1secs, 2)

        draw_overlay(frame, confThreshold, fps, detections_count, font)

        cv2.imshow('frame', frame)
        keyPress = cv2.waitKey(1)
        if keyPress == 113:  # 'q'
            break
        if keyPress == 82:  # Up arrow
            confThreshold = min(confThreshold + 0.1, 1)
        if keyPress == 84:  # Down arrow
            confThreshold = max(confThreshold - 0.1, 0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
