#!/usr/bin/python3

import cv2
import os
import numpy as np
import threading
import imutils
import queue
import time
import warnings

warnings.filterwarnings("ignore")
q=queue.Queue()
skip = 1
count = 0

class Stream:
    frame_width = -1
    frame_height = -1
    fps = -1
    threshold = 0.50
    refresh = -1
    boost = 1.25
    frameLim = 5
    output = {}

def Receive():
    print("OpenCV version: ",cv2.__version__)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp" 
    cap = cv2.VideoCapture("rtsp://service:Bosch123!@172.31.0.8:554?line=1&inst=1",cv2.CAP_FFMPEG)
    Stream.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Stream.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Stream.fps = int(cap.get(cv2.CAP_PROP_FPS))
    Stream.refresh = int(Stream.fps/5.0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4) #int( 1/ Stream.boost * Stream.fps))
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    print("Frames processed in 1 second: ",int(1/ Stream.boost * Stream.fps))
    print("Stream FPS as receved:", Stream.fps)
    ret, frame = cap.read()
    q.put(frame)
    count = 0
    delay = int(1 / Stream.fps*1.0)
    while ret:
        #count += 1
        #if count > Stream.threshold:
            #cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        #    count = 0
        ret, frame = cap.read()
        q.put(frame)
        #time.sleep( delay)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    cap.release()
        #    break

        

def Display():
    
    # grab the width and the height of the video stream
    
    # initialize the FourCC and a video writer object
    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #writer = cv2.VideoWriter("output.mp4", fourcc, Stream.fps, (Stream.frame_width, Stream.frame_height))
    # path to the weights and model files
    weights = "ssd_mobilenet/frozen_inference_graph.pb"
    model = "ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    # load the MobileNet SSD model trained  on the COCO dataset
    net = cv2.dnn.readNetFromTensorflow(weights, model)

    # load the class labels the model was trained on
    class_names = []
    with open("ssd_mobilenet/coco_names.txt", "r") as f:
        class_names = f.read().strip().split("\n")

    # create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    ## Initial window ##
    delay = int(1/ Stream.fps * 1000)
    count = 0
    while True:
        if not q.empty():
            frame=q.get()
            #frame = imutils.resize(frame, width=800)
            #start = datetime.datetime.now()
            #h = frame.shape[0]
            #w = frame.shape[1]
            
            if count % Stream.frameLim == 0:  
                # create a blob from the frame
                blob = cv2.dnn.blobFromImage(
                    frame, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
                # pass the blog through our network and get the output predictions
                net.setInput(blob)
                Stream.output = net.forward() # shape: (1, 1, 100, 7)
                count = 0

            count += 1

            for detection in Stream.output[0, 0, :, :]: # output[0, 0, :, :] has a shape of: (100, 7)
                # the confidence of the model regarding the detected object
                probability = detection[2]
                # if the confidence of the model is lower than 50%,
                # we do nothing (continue looping)
                if probability < Stream.threshold:
                    #del(detection)
                    continue

                # extract the ID of the detected object to get
                # its name and the color associated with it
                class_id = int(detection[1])
                label = class_names[class_id - 1].upper()
                ### ADD CODE TO FILTER OUT SPECIFIC LABELS ###
                color = colors[class_id]
                B, G, R = int(color[0]), int(color[1]), int(color[2])
                # perform element-wise multiplication to get
                # the (x, y) coordinates of the bounding box
                box = [int(a * b) for a, b in zip(detection[3:7], [Stream.frame_width, 
                                                                   Stream.frame_height,
                                                                   Stream.frame_width,
                                                                   Stream.frame_height])]
                box = tuple(box)
                # draw the bounding box of the object
                cv2.rectangle(frame, box[:2], box[2:], (B, G, R), thickness=2)

                # draw the name of the predicted object along with the probability
                text = "{} = {:.2f}%".format(label, probability*100.0)
                cv2.putText(frame, text, (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Output", frame)
            #cv2.waitKey( delay )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #writer.release()
            cv2.destroyAllWindows()
            break






# release the video capture, video writer, and close all windows
if __name__=='__main__':
    p1=threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.daemon = True
    p1.start()
    p2.start()



