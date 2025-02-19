# ================================
# Image Object Detection with YOLO
# ================================

# RUN WITH EXAMPLE COMMAND BELOW:

# python YOLO_img.py -i img_IO/work_table.jpg -o img_IO/work_table_processed.jpg -y yolov3 -d 10" into command prompt

import numpy as np
import argparse
import time
import cv2
import os
from pathlib import Path

"""User inputs through command line"""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLOv3 directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
ap.add_argument("-d", "--display_time", type=int, default=4, help="time in seconds the image is shown")
ap.add_argument("-s", "--show_output", default=False, help="it shows the output image", action='store_true')
args = ap.parse_args()


def convert_points_to_darknet(size, box):
	print ("Width={}, Height={}".format(size[0],size[1]))
	print ("Xmin={}, Xmax={}, Ymin={}, Ymax={}".format(box[0],box[1],box[2],box[3]))
	dw = 1./size[0]
	dh = 1./size[1]
	x = (box[0] + box[1])/2.0
	y = (box[2] + box[3])/2.0
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	return (x,y,w,h)
	
def darknet_to_voc(bb, w, h):
	bboxWidth = bb[2] * w
	bboxHeight = bb[3] * h
	centerX = bb[0] * w
	centerY = bb[1] * h

	voc = [None] * 4
	voc[0] = centerX - (bboxWidth/2)
	voc[1] = centerY - (bboxHeight/2)
	voc[2] = centerX + (bboxWidth/2)
	voc[3] = centerY + (bboxHeight/2)

	print ("VOC: " + " ".join([str(a) for a in voc]) + '\n')

def get_model():
	"""Load YOLOv3 use cv2 built in DNN module."""
	model = cv2.dnn.readNetFromDarknet(os.path.sep.join([args.yolo, "yolov3.cfg"]), 
												os.path.sep.join([args.yolo, "yolov3.weights"]))

	# load COCO class labels (open file (concatenated) -> extract string -> removed lead and end
	# whitespace, split by line break)
	labels = open(os.path.sep.join([args.yolo, "coco.names"])).read().strip().split("\n")

	# get output layer names (getLayerNames not subscriptable)
	getLayer = model.getLayerNames()
	out_layer_names = [getLayer[i[0] - 1] for i in model.getUnconnectedOutLayers()]

	return labels, model, out_layer_names


def get_input():
	"""Read image and get spatial dimensions."""
	image = cv2.imread(args.input)
	image_height, image_width = image.shape[:2]

	return image, image_width, image_height


def preprocess_input(model, image):
	"""Process image and set it as input."""
	blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (416, 416), swapRB=True, crop=False)
	model.setInput(blob)


def forward_pass(model, layer_names):
	"""Pass forward, keep track of time."""
	tick = time.time()
	layer_outputs = model.forward(layer_names)
	tock = time.time()
	print("YOLOv3 took {:.3f} seconds".format(tock-tick))

	return layer_outputs


def filter_output(layer_outputs, image_width, image_height):
	"""Initialize lists for bounding box."""
	boxes = []
	confidences = []
	classIDs = []

	# process output
	for output in layer_outputs:
		for detection in output:
			scores = detection[5:]  # detection starts with locational variables (0 to 1) then confidences
			classID = np.argmax(scores)
			confidence = scores[classID] 

			if confidence > args.confidence: # filter out low confidence
				box_data = detection[:4] * np.array([image_width, image_height, image_width, image_height]) 
				(center_X, center_Y, box_width, box_height) = box_data.astype("int")
				x = int(center_X - (box_width / 2))
				y = int(center_Y - (box_height / 2))

				# record box data, confidence, and class ID for the detected (note boxes is 2d)
				boxes.append([x, y, int(box_width), int(box_height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# with box dimension, we can now call non-maxima suppression (filtering out overlapping)
	indices = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.threshold)

	return boxes, confidences, classIDs, indices


def get_color(labels):
	"""Initialize random colors for difference objects."""
	np.random.seed(1)
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

	return colors
	

def draw_box(image, boxes, confidences, classIDs, indices, labels):
	"""Draw all bounding boxes."""
	colors = get_color(labels)
	if len(indices)>0:
		for i in indices.flatten():
			x,y,w,h = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]

			# draw rectangles and put text
			color = [int(c) for c in colors[classIDs[i]]]
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
			object_name = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
			cv2.putText(image, object_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	
	return image

def save_txt(image_height, image_width, boxes, confidences, classIDs, indices, labels):
	if len(indices)>0:
		for i in indices.flatten():
			
			classIndex = classIDs[i]
			label = labels[classIndex]
			confidence = confidences[i]

			if label == "person" and confidence > 0.5:
				x,y,w,h = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
				xmin = x
				ymin = y
				xmax = x + w
				ymax = y + h
				
				b = (float(xmin), float(xmax), float(ymin), float(ymax))
				bb = convert_points_to_darknet((image_width,image_height), b)

				save_path_without_extension = str(Path(args.input).parent / Path(args.input).stem) 
				output_save =  save_path_without_extension + '.txt'
				
				out_file = open(output_save, 'w')
				out_file.write(str(0) + " " + " ".join([str("{0:.6f}".format(a)) for a in bb]) + '\n')
				

def show_output(output_image):
	"""Show output for specified time."""
	print(args.output)
	cv2.imwrite(args.output, output_image)
	print("Finished!")
	print("Image will be displayed for {} seconds".format(args.display_time))
	cv2.imshow("Image", output_image)
	cv2.waitKey(args.display_time*1000)


def run():
	"""Organize and call the useful functions."""
	labels, model, layer_names = get_model()
	image, image_width, image_height = get_input()
	preprocess_input(model, image)
	yolo_output = forward_pass(model, layer_names)
	boxes, confidences, classIDs, indices = filter_output(yolo_output, image_width, image_height)
	save_txt(image_height, image_width, boxes, confidences, classIDs, indices, labels)
	
	if args.show_output:
		output_image = draw_box(image, boxes, confidences, classIDs, indices, labels)
		show_output(output_image)

run()


