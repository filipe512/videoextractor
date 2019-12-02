#python detect_faces.py --image C:\\VMs\\DNN\\Frames 
#--prototxt C:\\Users\\Filipe\\git\\videoextractor\\weights\\deploy.prototxt 
#--model C:\Users\Filipe\git\videoextractor\weights\res10_300x300_ssd.caffemodel

import os
import argparse
import cv2
from pathlib import Path
from face_tools import extract_faces


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image or directory")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.6, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

image_formats = ['.jpg']

     
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

input_path = args["image"]

if os.path.isdir(input_path):  
	file_list = [os.path.join(d, x)
    for d, dirs, files in os.walk(input_path)
    for x in files if Path(x).suffix in image_formats]
	
	for file in file_list:
		extract_faces(file, args["confidence"], net)

elif os.path.isfile(input_path): 
	extract_faces(input_path, args["confidence"], net) 