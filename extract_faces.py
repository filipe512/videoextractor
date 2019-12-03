#python extract_faces.py 
#--input C:\Users\ribeirfi\git\Pytorch_Retinaface\curve\trailer.m4v 
#--prototxt ./weights/deploy.prototxt 
#--model ./weights/res10_300x300_ssd.caffemodel 
#--output .

import argparse
import os
import time
import cv2

from imutils.video import FileVideoStream
from imutils.video import FPS
from pathlib import Path
from dnn.face_tools import extract_faces_from_image
from dnn.face_tools import save_faces_from_image
from dnn.age_gender_tools import get_gender, get_age


image_formats = ['.jpg']

def extract(input_path, output_folder, rate):
    STR_FRAME_PATH = "{}\\frame{}.jpg"
    video_args = {}
    video_args["video"] = input_path
        
    fvs = FileVideoStream(video_args["video"],queue_size=300).start()
    time.sleep(1.0)
    
    framecount = 0
    fps = FPS().start()
    net = cv2.dnn.readNetFromCaffe(args.face_prototxt, args.face_model)

    while fvs.running():
        frame = fvs.read()
        img_raw = None

        framecount = framecount + 1
        
        if framecount%int(rate)==0:
            print ("Extrating frame {}".format(framecount), end="\r")
            path = "{}\\{}{}.jpg".format(Path(args.input).parent, Path(args.input).stem, framecount)

            faces = extract_faces_from_image(frame, path, args.confidence, net)
            faces_to_save = []

            for face in faces:
                gender = get_gender(face, args.gender_model, args.gender_proto)
                if gender["gender"] == "Female" and gender["conf"] > 0.98:
                    faces_to_save.append(face)


            save_faces_from_image(faces_to_save, path)
        else:
            continue
            
        fps.update()

    fps.stop()
    fvs.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input video path to extract frames')
    parser.add_argument('--output',required=True, help='Output path where to extract frames')
    parser.add_argument("--face_prototxt", default="./weights/deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
    parser.add_argument("--face_model", default="./weights/res10_300x300_ssd.caffemodel", help="path to Caffe pre-trained model")
    parser.add_argument('--age_model', default="./weights/age_net.caffemodel")
    parser.add_argument('--age_proto', default="./weights/age_deploy.prototxt")
    parser.add_argument('--gender_model', default="./weights/gender_net.caffemodel")
    parser.add_argument('--gender_proto', default="./weights/gender_deploy.prototxt")
    parser.add_argument("-c", "--confidence", type=float, default=0.6, help="minimum probability to filter weak detections")
    parser.add_argument('--rate', default=15, help='Only saves frames every X frames. It helps to speed up frame extraction')
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    extract(args.input, args.output, args.rate)

    