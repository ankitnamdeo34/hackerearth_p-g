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

cap= cv2.VideoCapture("video.mp4")

counter = 0
i = 0
cap.set(1, i)

r1 = 230
r2 = 287
r3 = 346
r4 = 404
r5 = 462
r6 = 510
r7 = 555
r8 = 605
row_col = (0,255,0)


while (cap.isOpened()):
    ret, shelf_img = cap.read()
    if ret == False:
        break
  #  shelf_img = frame[:, 290:1120]

    cv2.rectangle(shelf_img, (8, r1), (825, r2), row_col, 2)
    cv2.rectangle(shelf_img, (8, r2), (825, r3), row_col, 2)
    cv2.rectangle(shelf_img, (8, r3), (825, r4), row_col, 2)
    cv2.rectangle(shelf_img, (8, r4), (825, r5), row_col, 2)
    cv2.rectangle(shelf_img, (8, r5), (825, r6), row_col, 2)
    cv2.rectangle(shelf_img, (8, r6), (825, r7), row_col, 2)
    cv2.rectangle(shelf_img, (8, r7), (825, r8), row_col, 2)

    # Display the resulting frame
    # Display the resulting frame
    cv2.imshow('frame', shelf_img)
#    cv2.imwrite("output/{}.jpg".format(str(i)), img)
    i += 1  # i.e. at 30 fps, this advances one second
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


shelf_img = cv2.imread(r"C:\Users\shash\OneDrive\Documents\Shashank\P_and_g_hackathone\void_detect\output\390.jpg")
(width, height, channel) = shelf_img.shape


cv2.imshow("test", shelf_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


### Writing frames back to video

import cv2
import os

image_folder = r'C:\Users\shash\OneDrive\Documents\Shashank\P_and_g_hackathone\void_detect\output'
video_name = 'video.mp4'

images = [os.path.join(image_folder,img) for img in os.listdir(image_folder)]
images.sort(key=os.path.getctime)

frame = cv2.imread(images[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 29, (width,height))
for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()


#
#
# import urllib
# import cv2
# from win32api import GetSystemMetrics
#
# #the [x, y] for each right-click event will be stored here
# right_clicks = list()
#
# #this function will be called whenever the mouse is right-clicked
# def mouse_callback(event, x, y, flags, params):
#
#     #right-click event value is 2
#     if event == 2:
#         global right_clicks
#
#         #store the coordinates of the right-click event
#         right_clicks.append([x, y])
#
#         #this just verifies that the mouse data is being collected
#         #you probably want to remove this later
#         print(right_clicks)
#
# img = cv2.imread(r"C:\Users\shash\OneDrive\Documents\Shashank\P_and_g_hackathone\void_detect\output\390.jpg",0)
# window_width = int(img.shape[1])
# window_height = int(img.shape[0])
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', window_width, window_height)
#
# #set mouse callback function for window
# cv2.setMouseCallback('image', mouse_callback)
#
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()