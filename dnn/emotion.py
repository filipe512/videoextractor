import cv2 as cv
import numpy as np


class Emotion:
 # ####################################################################################################
 ## Constructor
 def __init__(self):
     self.inpWidth = 64        # Resized image width passed to network
     self.inpHeight = 64       # Resized image height passed to network
     self.scale = 1.0          # Value scaling factor applied to input pixels
     self.mean = [127,127,127] # Mean BGR value subtracted from input image
     self.rgb = False          # True if model expects RGB inputs, otherwise it expects BGR


 def process(self, frame):
     font = cv.FONT_HERSHEY_PLAIN
     siz = 0.8
     white = (255, 255, 255)
     
     # Load the network if needed:
     if not hasattr(self, 'net'):
         backend = cv.dnn.DNN_BACKEND_DEFAULT
         target = cv.dnn.DNN_TARGET_CPU
         self.classes = [ "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" ]
         self.model = 'FER+ ONNX'
         self.net = cv.dnn.readNet('./weights/emotion_ferplus.onnx', '')
         self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
         self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
             
     frameHeight = frame.shape[0]
     frameWidth = frame.shape[1]
     mid = int((frameWidth - 110) / 2) + 110 # x coord of midpoint of our bars
     leng = frameWidth - mid - 6             # max length of our bars
     maxconf = 999

     gframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     blob = cv.dnn.blobFromImage(gframe, self.scale, (self.inpWidth, self.inpHeight), self.mean, self.rgb, crop=True)

     self.net.setInput(blob)
     out = self.net.forward()

     msgbox = np.zeros((96, frame.shape[1], 3), dtype = np.uint8) + 80

     out = out.flatten()

     index = np.argmax(out)
     label = self.classes[index]
     print ("[Final] Label: {} - Conf: {} ".format(label, out[index] * 100))

     for i in range(8):
         conf = out[i] * 100
         if conf > maxconf: conf = maxconf
         if conf < -maxconf: conf = -maxconf
         
         print ("{} - {}".format (self.classes[i], conf))
         
         
if __name__ == '__main__':
    frame = cv.imread('./../Pytorch_Retinaface/curve/trailer2670_face0.jpg')
    Emotion().process(frame)