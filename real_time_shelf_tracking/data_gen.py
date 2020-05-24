import cv2
import os
os.chdir(r"C:\Users\shash\OneDrive\Documents\Shashank\P_and_g_hackathone")
# Opens the Video file
cap = cv2.VideoCapture("time_lapse.mp4")
i = 390
cap.set(1, i)
while (cap.isOpened()):
    ret, frame = cap.read()
    img_crop = frame[:, 280:1120]
    if ret == False:
        break
    cv2.imwrite("frames/{}.jpg".format(str(i)), img_crop)
    i += 10  # i.e. at 30 fps, this advances one second
    cap.set(1, i)

cap.release()
cv2.destroyAllWindows()

