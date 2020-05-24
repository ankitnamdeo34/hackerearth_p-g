import argparse
import cv2
import numpy as np
import os
import pickle
import json
os.chdir(r"C:\Users\shash\OneDrive\Documents\Shashank\P_and_g_hackathone\void_detect")
config  = "yolov3-tiny_custom.cfg"
weights = "backup/yolov3-tiny_custom_last.weights"
names = "custom.names"

CONF_THRESH, NMS_THRESH = 0 , 0

# Load the network
net = cv2.dnn.readNetFromDarknet(config, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Draw the filtered bounding boxes with their class to the image
with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
# set bounding box colours
# Get the output layer from YOLO
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores


r1 = 228
r2 = 287
r3 = 346
r4 = 404
r5 = 462
r6 = 510
r7 = 555
r8 = 605
row_col = (0,255,0)

r1_area = 817*(r2-r1)
r2_area = 817*(r3-r2)
r3_area = 817*(r4-r3)
r4_area = 817*(r5-r4)
r5_area = 817*(r6-r5)
r6_area = 817*(r7-r6)
r7_area = 817*(r8-r7)
total_capacity = {"p1": r1_area, "p2": r2_area, "p3": r3_area, "p4": r4_area, "p5" :r5_area, "p6": r6_area, "p7":r7_area}
current_health = {"p1": [], "p2": [], "p3": [], "p4": [], "p5": [], "p6": [], "p7": []}


def get_void_location(void_cords):
    mid_y = void_cords[1]+(void_cords[3]/2)

    if mid_y<=r2:
        return "p1"
    elif r2<=mid_y<=r3:
        return "p2"
    elif r3<=mid_y<=r4:
        return "p3"
    elif r4<=mid_y<=r5:
        return "p4"
    elif r5<=mid_y<=r6:
        return "p5"
    elif r6<=mid_y<=r7:
        return "p6"
    else:
        return "p7"


def create_shelf_elements(img):
    cv2.rectangle(img, (8, r1), (825, r2), row_col, 2)
    cv2.rectangle(img, (8, r2), (825, r3), row_col, 2)
    cv2.rectangle(img, (8, r3), (825, r4), row_col, 2)
    cv2.rectangle(img, (8, r4), (825, r5), row_col, 2)
    cv2.rectangle(img, (8, r5), (825, r6), row_col, 2)
    cv2.rectangle(img, (8, r6), (825, r7), row_col, 2)
    cv2.rectangle(img, (8, r7), (825, r8), row_col, 2)

    cv2.putText(img, "{} Minutes".format(counter), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 2)

    cv2.putText(img, "Product_1", (8, r1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,0], 2)
    cv2.putText(img, "Product_2", (8, r2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,0], 2)
    cv2.putText(img, "Product_3", (8, r3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,0], 2)
    cv2.putText(img, "Product_4", (8, r4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,0], 2)
    cv2.putText(img, "Product_5", (8, r5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,0], 2)
    cv2.putText(img, "Product_6", (8, r6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,0], 2)
    cv2.putText(img, "Product_7", (8, r7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255,255,0], 2)


    cv2.putText(img, "Area: {} ".format(r1_area), (15, r1+ 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Area: {} ".format(r2_area), (15, r2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Area: {} ".format(r3_area), (15, r3+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Area: {} ".format(r4_area), (15, r4+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Area: {} ".format(r5_area), (15, r5+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Area: {} ".format(r6_area), (15, r6+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Area: {} ".format(r7_area), (15, r7+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)



def display_current_health(img):
    cv2.putText(img, "Health: {}% ".format(current_health["p1"]), (15, r1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Health: {}% ".format(current_health["p2"]), (15, r2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Health: {}% ".format(current_health["p3"]), (15, r3+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Health: {}% ".format(current_health["p4"]), (15, r4+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Health: {}% ".format(current_health["p5"]), (15, r5+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Health: {}% ".format(current_health["p6"]), (15, r6+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)
    cv2.putText(img, "Health: {}% ".format(current_health["p7"]), (15, r7+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,255,0], 1)




cap = cap = cv2.VideoCapture("video.mp4")
counter = 0
while (cap.isOpened()):
    ret, img = cap.read()
#    img = cv2.imread(r"C:\Users\shash\OneDrive\Documents\Shashank\P_and_g_hackathone\void_detect\output\1083.jpg")
#    img = frame[:, 280:1120]
    if ret == False:
        break

    create_shelf_elements(img)


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
        len(indices)
        # creating bounding boxes
        void_capacity = {"p1": [], "p2": [],"p3": [],"p4": [],"p5": [],"p6": [],"p7": []}

        for index in indices:
#            print(index)
            x, y, w, h = b_boxes[index]
            void_loc = get_void_location(void_cords=[x,y,w,h])
            void_area = w*h
            void_capacity[void_loc].append(void_area)
#            print(x,y,w,h)
            class_name = classes[class_ids[index]]
            cv2.rectangle(img, (x, y), (x + w, y + h), [0,0,255], 2)
            cv2.putText(img,"{} : {}".format(void_loc, void_area) , (int(x+4), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,[0,255,255], 1)
            cv2.putText(img, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,[0,0,255], 2)

        for key in void_capacity.keys():
            empty_perc = sum(void_capacity[key])*100/total_capacity[key]
            health = round(100 - empty_perc,2)
            current_health[key] = health

        display_current_health(img, current_health)

    except:
        print("NO VOID")

    counter += 1  # i.e. at 30 fps, this advances one second
    #cap.set(1, counter)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


cv2.imshow("dsdas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()





