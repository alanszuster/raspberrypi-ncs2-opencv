
import cv2
import time
import numpy as np
import argparse
from multiprocessing import Process, Queue
from picamera.array import PiRGBArray
from picamera import PiCamera
from sys import getsizeof


def parse_args():
	parser = argparse.ArgumentParser(description='NCS2 PiCamera')
	parser.add_argument('-b', '--bin')
	parser.add_argument('-x', '--xml')
	parser.add_argument('-l', '--labels')
	parser.add_argument('-pb', '--protobox')
	parser.add_argument('-pbtxt', '--protoboxtxt')
	parser.add_argument('-ct', '--conf_threshold', default=0.5, type=float)
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
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
	return net, model_type


def load_labels(labels_path):
	with open(labels_path, 'r') as f:
		labels = [x.strip() for x in f]
	print(labels)
	return labels


def classify_frame(net, inputQueue, outputQueue):
	# model_type must be passed in or set globally
	global model_type
	while True:
		if not inputQueue.empty():
			frame = inputQueue.get()
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
				outputQueue.put(data_out)
			else:
				blob = cv2.dnn.blobFromImage(frame, 0.007843, size=(300, 300),
											mean=(127.5,127.5,127.5), swapRB=False, crop=False)
				net.setInput(blob)
				out = net.forward()
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
				outputQueue.put(data_out)


def draw_detections(frame, detections, labels, confThreshold, font):
	for detection in detections:
		objID, confidence, xmin, ymin, xmax, ymax = detection
		if confidence > confThreshold:
			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))
			cv2.rectangle(frame, (xmin-1, ymin-1), (xmin+70, ymin-10), (0,255,255), -1)
			label_text = f' {labels[objID]} {round(confidence,2)}'
			cv2.putText(frame, label_text, (xmin, ymin-2), font, 0.3, (0,0,0), 1, cv2.LINE_AA)


def draw_overlay(frame, confThreshold, fps, qfps, detections, t2secs, font, frameWidth, frameHeight):
	cv2.rectangle(frame, (0, 0), (90, 15), (0,0,0), -1)
	cv2.putText(frame, f'Threshold: {round(confThreshold,1)}', (10, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
	cv2.rectangle(frame, (220, 0), (300, 25), (0,0,0), -1)
	cv2.putText(frame, f'VID FPS: {fps}', (225, 10), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f'NCS FPS: {qfps}', (225, 20), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
	cv2.rectangle(frame, (0, 265), (170, 300), (0,0,0), -1)
	cv2.putText(frame, f'Positive detections: {detections}', (10, 280), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame, f'Elapsed time: {round(t2secs,2)}', (10, 290), font, 0.3, (0, 255, 255), 1, cv2.LINE_AA)


def main():
	args = parse_args()
	confThreshold = args.conf_threshold
	frameWidth = 304
	frameHeight = 304
	font = cv2.FONT_HERSHEY_SIMPLEX
	detections_count = 0
	fps = 0.0
	qfps = 0.0

	net, model_type_local = load_network(args)
	global model_type
	model_type = model_type_local
	labels = load_labels(args.labels)

	camera = PiCamera()
	camera.resolution = (frameWidth, frameHeight)
	camera.framerate = 35
	rawCapture = PiRGBArray(camera, size=(frameWidth, frameHeight))
	time.sleep(0.1)

	inputQueue = Queue(maxsize=1)
	outputQueue = Queue(maxsize=1)

	print("[INFO] starting process...")
	p = Process(target=classify_frame, args=(net, inputQueue, outputQueue))
	p.daemon = True
	p.start()

	print("[INFO] starting capture...")
	timer1 = time.time()
	frames = 0
	queuepulls = 0
	timer2 = 0
	t2secs = 0
	out = None

	while True:
		for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
			if queuepulls == 1:
				timer2 = time.time()

			frame = frame.array

			if inputQueue.empty():
				inputQueue.put(frame)

			if not outputQueue.empty():
				out = outputQueue.get()
				queuepulls += 1
				print(len(out))
				print(getsizeof(out))

			if out is not None:
				draw_detections(frame, out, labels, confThreshold, font)
				detections_count += sum(1 for d in out if d[1] > confThreshold)

			draw_overlay(frame, confThreshold, fps, qfps, detections_count, t2secs, font, frameWidth, frameHeight)

			cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
			cv2.resizeWindow('frame', frameWidth, frameHeight)
			cv2.imshow('frame', frame)

			frames += 1
			if frames >= 1:
				end1 = time.time()
				t1secs = end1 - timer1
				fps = round(frames / t1secs, 2)
			if queuepulls > 1:
				end2 = time.time()
				t2secs = end2 - timer2
				qfps = round(queuepulls / t2secs, 2)

			rawCapture.truncate(0)

			keyPress = cv2.waitKey(1)
			if keyPress == 113:  # 'q'
				break
			if keyPress == 82:  # Up arrow
				confThreshold = min(confThreshold + 0.1, 1)
			if keyPress == 84:  # Down arrow
				confThreshold = max(confThreshold - 0.1, 0)

		break
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
