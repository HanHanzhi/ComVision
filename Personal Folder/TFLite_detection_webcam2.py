import math
import requests
import importlib.util
import random
from threading import Thread
import time
import sys
import numpy as np
import cv2
import argparse
import os
import glob
url = 'http://192.168.50.19:5000/upload'

#from camtopc import *

CWD_PATH = '/Users/han/Desktop/NTU Courseware/MDP copy/Personal Folder/tflite1/'
OPEN_WINDOW = True


    
def usecamera(videotimer):
    global scores
    global scoring
    global label
    global object_name
    global file_array

    # this is where the output file will go
    with open('/Users/han/Desktop/NTU Courseware/MDP copy/Personal Folder/output.txt', "a") as f:
        f.truncate(0)
    scoring = 0
    file_array = set()
    # Define VideoStream class to handle streaming of video from webcam in separate processing thread
    # Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/

    def upload(filename, url):
        with open(filename, 'rb') as f:
            files = {'file': f}
            r = requests.post(url, files=files)
            print(r.content, r.status_code)

    class VideoStream:
        """Camera object that controls video streaming from the Picamera"""

        def __init__(self, resolution=(1080, 720), framerate=30):
            # Initialize the PiCamera and the camera image stream
            self.stream = cv2.VideoCapture(0)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC,
                                  cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3, resolution[0])
            ret = self.stream.set(4, resolution[1])

            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
            self.stopped = False

        def start(self):
            # Start the thread that reads frames from the video stream
            Thread(target=self.update, args=()).start()
            return self

        def update(self):
            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

        def read(self):
            # Return the most recent frame
            return self.frame

        def stop(self):
            # Indicate that the camera and thread should be stopped
            self.stopped = True

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=False, default="Sample_TFLite_model")
    # required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    # Get path to current working directory
    #CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    #frame_rate_calc = 1
    #freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    time.sleep(1)

    # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    ymaxavg = [0]*31
    yminavg = [0]*31
    xminavg = [0]*31
    xmaxavg = [0]*31
    average = [0]*31
    output = [-1, 500, 500, 180]
    
    while (videotimer != 0):
        videotimer = videotimer-1
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        # Bounding box coordinates of detected objects
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[
            0]  # Class index of detected objects
        # print(output_details)
        try:
            scores = interpreter.get_tensor(output_details[0]['index'])[
                0]  # Confidence of detected objectz
            # print(scores)
    #         print(output_details[2]['index'])
        except:
            break

        # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (10, 255, 0), 2)
                object_name = int(labels[int(classes[i])]) 

                yminavg[object_name-10] += ymin
                xminavg[object_name-10] += xmin
                ymaxavg[object_name-10] += ymax
                xmaxavg[object_name-10] += xmax
                average[object_name-10] += 1

                scoring = int(scores[i]*100)
                # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(
                    scores[i]*100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                # Make sure not to draw label too close to top of window
                label_ymin = max(ymin, labelSize[1] + 10)
                # Draw white box to put label text in
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
                    xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

                

        # Draw framerate in corner of frame
        #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        # All the results have been drawn on the frame, so it's time to display it.
        if OPEN_WINDOW:
            cv2.imshow('Object detector', frame)

        if scoring > 75:

            # print(label)
            # print(file_array)
            if object_name not in file_array:
                if object_name != 0:
                    cv2.imwrite(
                        f'/Users/han/Desktop/NTU Courseware/MDP copy/Personal Folder/tflite1/images/{object_name}.jpg', frame)
                    #cv2.imwrite(f'/Users/han/Desktop/NTU Courseware/MDP copy/Personal Folder/tflite1/tflite/images_1/image{a},{b}.jpg', frame)
                    # print(label,xmin,xmax,ymin,ymax)
        # with open('/home/pi/tflite1/tflite/images_1/image1.txt', "a") as f:
         #   f.write(label)
          #  f.write('\n')
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()

    
    #print(xminavg[19])
    #print(xmaxavg[19])
    #print(yminavg[19])
    #print(ymaxavg[19])
    # print(average)
    return -1