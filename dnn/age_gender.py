# Import required modules
import cv2 as cv
import math
import time
import argparse

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image face.')
parser.add_argument('--agemodel', default="./weights/age_net.caffemodel")
parser.add_argument('--ageproto', default="./weights/age_deploy.prototxt")
parser.add_argument('--gendermodel', default="./weights/gender_net.caffemodel")
parser.add_argument('--genderproto', default="./weights/gender_deploy.prototxt")


args = parser.parse_args()


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
print (args.agemodel)
ageNet = cv.dnn.readNet(args.agemodel, args.ageproto)
genderNet = cv.dnn.readNet(args.gendermodel, args.genderproto)

face = cv.imread(args.input)
blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
genderNet.setInput(blob)
genderPreds = genderNet.forward()
gender = genderList[genderPreds[0].argmax()]
# print("Gender Output : {}".format(genderPreds))
print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

ageNet.setInput(blob)
agePreds = ageNet.forward()
age = ageList[agePreds[0].argmax()]
print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))