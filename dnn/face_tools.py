import torch 
import random
import cv2
import numpy as np
import os
from pathlib import Path

def input_for_ssd(image):
	return cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

def input_for_retinaface(image):
	return cv2.dnn.blobFromImage(cv2.resize(image, (640, 640)), 1.0, (640, 640), (104.0, 117.0, 123.0))
	
def get_scores_ssd(img, path, conf, detections):
	(h, w) = img.shape[:2]
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
		if confidence > conf:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")


			refPoint = [(startX, startY), (endX, endY)]
			cropped = img[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]

			count_face = 0
			file_exists = True
			output_path = None

			while (file_exists):
				output_path = "{}\\{}_face{}.jpg".format(Path(path).parent, Path(path).stem, count_face)
				file_exists = os.path.exists(output_path)
				count_face += 1

			print (output_path)
			cv2.imwrite(output_path, cropped)

			# draw the bounding box of the face along with the associated probability
			#text = "{:.2f}%".format(confidence * 100)
			#y = startY - 10 if startY - 10 > 10 else startY + 10
			#cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			#cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	#show the output image
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)

def extract_faces(path, conf, net):
	image = cv2.imread(path)
	extract_faces_from_image(image, path, conf, net)

def extract_faces_from_image(image, path, conf, net):
	
	blob = input_for_ssd(image)

	net.setInput(blob)

	getLayer = net.getLayerNames()
	out_layer_names = [getLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	detections = net.forward()

	get_scores_ssd(image, path, conf, detections)