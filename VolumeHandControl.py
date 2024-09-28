import cv2
import time
import numpy as np

# cam = cv2.VideoCapture('rtsp://192.168.0.39:8080/h264_ulaw.sdp') # To access from Mobile, we need ipwebcam and server should be up

cam = cv2.VideoCapture('rtsp://192.168.0.39:8080/h264_ulaw.sdp')

wCam, hCam = 640, 480
cam.set(3, wCam)
cam.set(4, wCam)
pTime = 0

while True:
  success, img = cam.read()

  cTime = time.time()
  fps = 1/(cTime - pTime)
  pTime = cTime
  cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
  cv2.imshow("Img", img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cam.release()
cv2.destroyAllWindows()