import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
  success, img = cam.read()

  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(imgRGB)
  cTime = time.time()
  fps = 1/(cTime - pTime)
  pTime = cTime

  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      for id, lm in enumerate(handLms.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        if id == 4:
          cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
      mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

  cv2.putText(img, f"FPS: {str(int(fps))}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
  cv2.imshow("Image", img)
  cv2.waitKey(1)