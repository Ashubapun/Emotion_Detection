# # import cv2
# # from deepface import DeepFace
# # # # import matplotlib.pyplot as plt

# # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # cap = cv2.VideoCapture(0)

# # # # cv2.namedWindow("preview")
# # # # vc = cv2.VideoCapture(0)
# # # #
# # # # if vc.isOpened(): # try to get the first frame
# # # #     rval, frame = vc.read()
# # # # else:
# # # #     rval = False

# # # if not cap.isOpened():
# # #   cap = cv2.VideoCapture(0)
# # # if not cap.isOpened():
# # #   raise IOError("Cannot open webcam")

# # while True:
# #   ret, frame = cap.read()
# #   result = DeepFace.analyze(frame, actions = ['emotion'])

# #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #   faces = faceCascade.detectMultiScale(gray, 1.1, 4)

# #   for(x,y,w,h) in faces:
# #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

# #   font = cv2.FONT_HERSHEY_SIMPLEX

# #   cv2.putText(frame, result['dominant_emotion'], (50,50), font, 3, (0,0,255), 2, cv2.LINE_4)
# #   cv2.imshow('Original video', frame)

# #   if cv2.waitKey(2) & 0xFF == ord('q'):
# #     break

# # cap.release()
# # cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from deepface import DeepFace

# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# capture = cv2.VideoCapture(0)

# while True:
#   retrn, frame = capture.read()
#   result = DeepFace.analyze(frame, actions = ['emotion'])

#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#   faces = faceCascade.detectMultiScale(gray, 1.1, 4)

#   for(x,y,w,h) in faces:
#     cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

#   font = cv2.FONT_HERSHEY_SIMPLEX

#   cv2.putText(frame, result['dominant_emotion'], (50,50), font, 3, (0,0,255), 2, cv2.LINE_4)
#   cv2.imshow('webcam', frame)

#   if cv2.waitKey(1) & 0xFF == ord('x'):
#       break

# capture.release()
# cv2.destroyAllWindows()

# # cv2.waitKey(0)


# import cv2
# from deepface import DeepFace
#
# # Load the pre-trained model for emotion detection
# model = DeepFace.build_model('Emotion')
#
# # Open a connection to the webcam
# cap = cv2.VideoCapture(0)
# 
# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()
#
# # Loop to capture frames from the webcam
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # If the frame is not read correctly, exit the loop
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#
#     # Detect emotions in the frame
#     result = DeepFace.analyze(frame, actions=['emotion'])
#
#     # Get the dominant emotion
#     emotion = result['dominant_emotion']
#
#     # Draw the emotion text on the frame
#     cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
#
#     # Display the resulting frame
#     cv2.imshow('Emotion Detection', frame)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the webcam and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
