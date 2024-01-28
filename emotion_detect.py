# import cv2
# from deepface import DeepFace
# # # import matplotlib.pyplot as plt

# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture(0)

# # # cv2.namedWindow("preview")
# # # vc = cv2.VideoCapture(0)
# # #
# # # if vc.isOpened(): # try to get the first frame
# # #     rval, frame = vc.read()
# # # else:
# # #     rval = False

# # if not cap.isOpened():
# #   cap = cv2.VideoCapture(0)
# # if not cap.isOpened():
# #   raise IOError("Cannot open webcam")

# while True:
#   ret, frame = cap.read()
#   result = DeepFace.analyze(frame, actions = ['emotion'])

#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#   faces = faceCascade.detectMultiScale(gray, 1.1, 4)

#   for(x,y,w,h) in faces:
#     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

#   font = cv2.FONT_HERSHEY_SIMPLEX

#   cv2.putText(frame, result['dominant_emotion'], (50,50), font, 3, (0,0,255), 2, cv2.LINE_4)
#   cv2.imshow('Original video', frame)

#   if cv2.waitKey(2) & 0xFF == ord('q'):
#     break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while True:
  retrn, frame = capture.read()
  result = DeepFace.analyze(frame, actions = ['emotion'])

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(gray, 1.1, 4)

  for(x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

  font = cv2.FONT_HERSHEY_SIMPLEX

  cv2.putText(frame, result['dominant_emotion'], (50,50), font, 3, (0,0,255), 2, cv2.LINE_4)
  cv2.imshow('webcam', frame)

  if cv2.waitKey(1) & 0xFF == ord('x'):
      break

capture.release()
cv2.destroyAllWindows()

# cv2.waitKey(0)
