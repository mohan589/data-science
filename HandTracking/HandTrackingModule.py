import cv2
import mediapipe as mp
import time

# static_image_mode=False,
# max_num_hands=2,
# model_complexity=1,
# min_detection_confidence=0.5,
# min_tracking_confidence=0.5

class handDetector():
  def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionCon = detectionCon
    self.trackCon = trackCon

    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils

    #   for id, lm in enumerate(handLms.landmark):
    #     h, w, c = img.shape
    #     cx, cy = int(lm.x * w), int(lm.y * h)
    #     # if id == 4:
    #     cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
    #   self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

  def findHands(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = self.hands.process(imgRGB)
    if results.multi_hand_landmarks:
      for handLms in results.multi_hand_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
    return img

def main():
  pTime = 0
  cTime = 0
  cam = cv2.VideoCapture(0)
  detector = handDetector()

  cam.set(3, 640)
  cam.set(4, 480)

  while True:
    success, img = cam.read()
    img = detector.findHands(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {str(int(fps))}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()