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

image_formats = ['.jpg']

def extract(input_path, output_folder, rate):
    STR_FRAME_PATH = "{}\\frame{}.jpg"
    video_args = {}
    video_args["video"] = input_path
        
    fvs = FileVideoStream(video_args["video"],queue_size=300).start()
    time.sleep(1.0)
    
    framecount = 0
    fps = FPS().start()
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    while fvs.running():
        frame = fvs.read()
        img_raw = None

        framecount = framecount + 1
        
        if framecount%int(rate)==0:
            print ("Extrating frame {}".format(framecount), end="\r")
            path = "{}\\{}{}.jpg".format(Path(args.input).parent, Path(args.input).stem, framecount)

            extract_faces_from_image(frame, path, args.confidence, net)
            #cv2.imwrite(STR_FRAME_PATH.format(output_folder, str(framecount)),frame)
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
    parser.add_argument("--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    parser.add_argument("-c", "--confidence", type=float, default=0.6, help="minimum probability to filter weak detections")
    parser.add_argument("--model", required=True, help="path to Caffe pre-trained model")
    parser.add_argument('--rate', default=15, help='Only saves frames every X frames. It helps to speed up frame extraction')
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    extract(args.input, args.output, args.rate)

    