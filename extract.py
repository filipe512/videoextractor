import argparse
import os
import time
import cv2

from imutils.video import FileVideoStream
from imutils.video import FPS
from pathlib import Path

image_formats = ['.jpg']

def extract(input_path, output_folder, rate):
    STR_FRAME_PATH = "{}\\frame{}.jpg"
    video_args = {}
    video_args["video"] = input_path
        
    fvs = FileVideoStream(video_args["video"],queue_size=300).start()
    time.sleep(1.0)
    
    framecount = 0
    fps = FPS().start()
        
    while fvs.running():
        frame = fvs.read()
        img_raw = None

        framecount = framecount + 1
        
        if framecount%int(rate)==0:
            print ("Extrating frame {}".format(framecount), end="\r")
            cv2.imwrite(STR_FRAME_PATH.format(output_folder, str(framecount)),frame)
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
    parser.add_argument('--rate', default=1, help='Only saves frames every X frames. It helps to speed up frame extraction')
    args = parser.parse_args()
    
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    extract(args.input, args.output, args.rate)

    