#####################################################################

# Task X : create and use a mask of tracked faces

#####################################################################

import cv2
import numpy as np

#####################################################################

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
background = cv2.imread("background.jpg", cv2.IMREAD_COLOR)

camera.set(3, 1280)
camera.set(4, 720)

ret, first_frame = camera.read()
h, w = first_frame.shape[:2]
if background is not None:
    background = cv2.resize(background, (w, h))

while True:
    ret, frame = camera.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 5, minSize=(30, 30))
    
    result = background.copy() if background is not None else frame.copy()
    
    if len(faces) > 0:
        x, y, w_face, h_face = max(faces, key=lambda f: f[2] * f[3])
        cx = int(x + 0.5 * w_face)
        cy = int(y + 0.4 * h_face)
        
        # Create oval mask (taller than wide)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (int(w_face * 0.6), int(h_face * 0.8)), 0, 0, 360, 255, -1)
        
        # Show camera in oval, background outside
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch == 255, frame, result)
    
    cv2.imshow("Face Tracking with Background Mask", result)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()

