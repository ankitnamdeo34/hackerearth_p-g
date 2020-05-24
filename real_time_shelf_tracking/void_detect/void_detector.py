import argparse
import cv2
import numpy as np
import os
import pickle
import json
config  = "yolov3-tiny_custom.cfg"
weights = "void_weights/yolov3-tiny_custom_last.weights"
names = "custom.names"

video_path = r"inputs/video.mp4"

CONF_THRESH, NMS_THRESH = 0 , 0

# Load the network
net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Draw the filtered bounding boxes with their class to the image
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
# set bounding box colours
# colors = np.random.uniform(0, 255, size=(len(classes), 3))
pan_col = [0,0,255]
colors = [pan_col]
# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores

cap = cap = cv2.VideoCapture(video_path)

counter = 0
while (cap.isOpened()):
    ret, img = cap.read()
#    img = frame[:, 280:1120]
    if ret == False:
        break
    counter += 1  # i.e. at 30 fps, this advances one second
    # Display the resulting frame
    try:
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)
        class_ids, confidences, b_boxes = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

        # creating bounding boxes
        for index in indices:
            x, y, w, h = b_boxes[index]
            print(x,y,w,h)
            class_name = classes[class_ids[index]]
            cv2.rectangle(img, (x, y), (x + w, y + h), [0,0,255], 2)
            cv2.putText(img, class_name, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,[0,0,255], 2)

    except:
        print("NO VOID")

    cap.set(1, counter)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

