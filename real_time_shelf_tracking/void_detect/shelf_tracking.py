import argparse
import cv2
import numpy as np
import os
import pickle
import json
import time

video_file_path = r"inputs/video.mp4"

config  = "yolov3-tiny_custom.cfg"
weights = "void_weights/yolov3-tiny_custom_last.weights"
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

shelf_start = 70
shelf_end  = 825
shelf_len = shelf_end - shelf_start

r1 = 226
r2 = 287
r3 = 346
r4 = 404
r5 = 459
r6 = 508
r7 = 555
r8 = 603
row_col = (0,255,0)

r1_area = shelf_len*(r2-r1)
r2_area = shelf_len*(r3-r2)
r3_area = shelf_len*(r4-r3)
r4_area = shelf_len*(r5-r4)
r5_area = shelf_len*(r6-r5)
r6_area = shelf_len*(r7-r6)
r7_area = shelf_len*(r8-r7)
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



def draw_text(img, coords, text, bg_color = [255,255,255], tx_col = [0,0,0], thickness = 1, font_scale = 0.3, margin = 2):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    # set the text start position
    text_offset_x = coords[0]
    text_offset_y = coords[1] - margin
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + margin, text_offset_y - text_height -margin))
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y -margin), font, fontScale=font_scale, color=tx_col, thickness=thickness)


def create_shelf_elements(img):
    cv2.rectangle(img, (8, r1), (825, r2), row_col, 2)
    cv2.rectangle(img, (8, r2), (825, r3), row_col, 2)
    cv2.rectangle(img, (8, r3), (825, r4), row_col, 2)
    cv2.rectangle(img, (8, r4), (825, r5), row_col, 2)
    cv2.rectangle(img, (8, r5), (825, r6), row_col, 2)
    cv2.rectangle(img, (8, r6), (825, r7), row_col, 2)
    cv2.rectangle(img, (8, r7), (825, r8), row_col, 2)

    cv2.putText(img, "{} Minutes".format(counter), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 2)

    for i in range(1,8):
        draw_text(img, coords=(7,eval("r{}".format(i))), text="Product {}".format(i), bg_color=[255,255,0], thickness=1, font_scale=0.6)

    # cv2.putText(img, "Product_1", (7, r1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0], 2)
    # cv2.putText(img, "Product_2", (7, r2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0], 2)
    # cv2.putText(img, "Product_3", (7, r3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0], 2)
    # cv2.putText(img, "Product_4", (7, r4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0], 2)
    # cv2.putText(img, "Product_5", (7, r5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0], 2)
    # cv2.putText(img, "Product_6", (7, r6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0], 2)
    # cv2.putText(img, "Product_7", (7, r7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255,255,0], 2)

    for i in range(1,8):
        draw_text(img, coords=(15,eval("r{}".format(i)) +15), text="Area: {}".format(eval("r{}_area".format(i))), bg_color=[0,255,255])

    # cv2.putText(img, "Area: {} ".format(r1_area), (15, r1+ 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,255,0], 1)
    # cv2.putText(img, "Area: {} ".format(r2_area), (15, r2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,255,0], 1)
    # cv2.putText(img, "Area: {} ".format(r3_area), (15, r3+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,255,0], 1)
    # cv2.putText(img, "Area: {} ".format(r4_area), (15, r4+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,255,0], 1)
    # cv2.putText(img, "Area: {} ".format(r5_area), (15, r5+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,255,0], 1)
    # cv2.putText(img, "Area: {} ".format(r6_area), (15, r6+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,255,0], 1)
    # cv2.putText(img, "Area: {} ".format(r7_area), (15, r7+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,255,0], 1)



def display_health(img):
    for i in range(1,8):
        draw_text(img, coords=(15,eval("r{}".format(i)) +25), text="Health: {}% ".format(current_health["p{}".format(i)]))

    # cv2.putText(img, "Health: {}% ".format(current_health["p1"]), (15, r1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,0], 1)
    # cv2.putText(img, "Health: {}% ".format(current_health["p2"]), (15, r2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [203,192,255], 1)
    # cv2.putText(img, "Health: {}% ".format(current_health["p3"]), (15, r3+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [203,192,255], 1)
    # cv2.putText(img, "Health: {}% ".format(current_health["p4"]), (15, r4+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [203,192,255], 1)
    # cv2.putText(img, "Health: {}% ".format(current_health["p5"]), (15, r5+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [203,192,255], 1)
    # cv2.putText(img, "Health: {}% ".format(current_health["p6"]), (15, r6+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [203,192,255], 1)
    # cv2.putText(img, "Health: {}% ".format(current_health["p7"]), (15, r7+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0,0,0], 1)
#    return img




cap = cap = cv2.VideoCapture(video_file_path)
counter = 0

while (cap.isOpened()):
    ret, img = cap.read()
    if ret == False:
        break

    img2 = img.copy()
    try:
        height, width = img2.shape[:2]

        blob = cv2.dnn.blobFromImage(img2, 0.00392, (416, 416), swapRB=True, crop=False)
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
            x, y, w, h = b_boxes[index]
            void_loc = get_void_location(void_cords=[x,y,w,h])
            void_area = w*h
            void_capacity[void_loc].append(void_area)
            class_name = classes[class_ids[index]]
            cv2.rectangle(img, (x, y), (x + w, y + h), [0,0,255], 2)
            cv2.putText(img,"{} : {}".format(void_loc, void_area) , (int(x+4), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,[0,255,255], 1)
            # cv2.putText(img, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,[0,0,255], 2)
            draw_text(img,text=class_name, coords= (x,y), bg_color=[0,0,255])

        msg_oos = []
        for key in void_capacity.keys():
            empty_perc = sum(void_capacity[key])*100/total_capacity[key]
            health = round(100 - empty_perc,2)
            current_health[key] = health
            if health<= 40:
                msg = "Product {} is about to OOS, HEALTH : {}%".format(key,health)
                msg_oos.append(msg)

        if msg_oos:
            for i, msg in enumerate(msg_oos):
                draw_text(img,(130,70+25*i),text=msg,tx_col=[0,0,255],font_scale=0.65, thickness=1, margin=10)
            cv2.waitKey(100)

                # cv2.putText(img,"Triggered: {}".format(str(key)),( 100,100), cv2.FONT_HERSHEY_SIMPLEX, 3,[0,0,255], 5)

        print(current_health)
        create_shelf_elements(img)
        display_health(img)


    except:
        print("NO VOID")

    counter += 1  # i.e. at 30 fps, this advances one second
    cap.set(1, counter)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
