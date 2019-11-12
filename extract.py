import argparse
import os
import time
import cv2

from imutils.video import FileVideoStream
from imutils.video import FPS
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input video path to extract frames')
parser.add_argument('--output', default='tmp_frame', help='Output path where to extract frames')
parser.add_argument('--rate', default=1, help='Only saves frames every X frames. It helps to speed up frame extraction')


args = parser.parse_args()


if __name__ == '__main__':
    STR_FRAME_PATH = "{}\\frame{}.jpg"

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    video_args = {}
    video_args["video"] = args.input
        
    fvs = FileVideoStream(video_args["video"],queue_size=300).start()
    time.sleep(1.0)
    
    framecount = 0
    fps = FPS().start()
        
    while fvs.running():
        frame = fvs.read()
        img_raw = None

        framecount = framecount + 1
        
        if framecount%int(args.rate)==0:
            cv2.imwrite(STR_FRAME_PATH.format(args.output, str(framecount)),frame)
        else:
            continue
            
        fps.update()

    fps.stop()
    fvs.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))