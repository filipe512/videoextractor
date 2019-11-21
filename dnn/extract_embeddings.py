from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True, help="input path of face directory")
ap.add_argument("-e", "--output", required=True, help="path to output serialized db of facial embeddings")
ap.add_argument("-m", "--embedding-model", required=True, help="path to OpenCV's deep learning face embedding model")
args = ap.parse_args()


print("Loading Openface imlementation of Facenet model")
embedder = cv2.dnn.readNetFromTorch(args.embedding_model)

print("Load image dataset..")
imagePaths = list(paths.list_images(args.input_dir))

embeddings = []
face_names = []

total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	face = cv2.imread(imagePath)

	name = imagePath.split(os.path.sep)[-2]
	
	face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
	embedder.setInput(face_blob)
	vector_embeddings = embedder.forward()

	# add the name of the person + corresponding face embedding to their respective lists
	face_names.append(name)
	embeddings.append(vector_embeddings.flatten())
	total += 1

# dump the facial embeddings + names to disk
data = {"embeddings": embeddings, "names": face_names}
f = open(args.output, "wb")
f.write(pickle.dumps(data))
f.close()