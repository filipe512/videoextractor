# Import required modules
import cv2 as cv
import math
import time
import argparse

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


genderNet = None
ageNet = None

def init_gender_net(model_path, proto_path):
	global genderNet
	if (genderNet is None):
		genderNet = cv.dnn.readNet(model_path, proto_path)

def init_age_net(model_path, proto_path):
	global ageNet
	if (ageNet is None):
		ageNet = cv.dnn.readNet(model_path, proto_path)

def get_gender(face, model_path, proto_path):
	init_gender_net(model_path, proto_path)
	blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
	genderNet.setInput(blob)
	genderPreds = genderNet.forward()
	gender = genderList[genderPreds[0].argmax()]

	return {"gender": gender, "conf": genderPreds[0].max()}
	#print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))


def get_age(face, model_path, proto_path):
	init_age_net(model_path, proto_path)
	blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
	ageNet.setInput(blob)
	agePreds = ageNet.forward()
	age = ageList[agePreds[0].argmax()]
	print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))