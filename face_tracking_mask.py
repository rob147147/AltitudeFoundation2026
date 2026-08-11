#####################################################################

# Task X : create and use a mask of tracked faces

#####################################################################

import cv2
import numpy as np

#####################################################################

# define video capture with access to camera 0

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# define display window

window_name = "Face Tracking with Background Mask"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# define the face classifier

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# get the background image

background_path = "background.jpg"
background = cv2.imread(background_path, cv2.IMREAD_COLOR)

#####################################################################

# get camera dimensions and resize background

ret, first_frame = camera.read()
h, w = first_frame.shape[:2]
if background is not None:
    background = cv2.resize(background, (w, h))

#####################################################################

keep_processing = True

while (keep_processing):
    
    # read an image from the camera
    
    _, image = camera.read()

    # detect faces, by applying the face cascade to a greyscale version of the image
    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 5, minSize=(30, 30))
    
    # set result to background, or frame if no background loaded
    
    result = background.copy() if background is not None else image.copy()
    
    # check if there are any faces to apply mask to
    
    if len(faces) > 0:
        
        # find the largest face and create oval mask
        
        x, y, w_face, h_face = max(faces, key=lambda f: f[2] * f[3])
        cx = int(x + 0.5 * w_face)
        cy = int(y + 0.4 * h_face)
        
        # create (tall) oval mask
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (int(w_face * 0.6), int(h_face * 0.8)), 0, 0, 360, 255, -1)
        
        # show camera in oval, background outside
        
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch == 255, image, result)
    
    # display background image oval outline for face
        
    cv2.imshow(window_name, result)

    # start the event loop - if user presses "x" or ESC then exit
    # wait 40ms for a key press from the user (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(40) & 0xFF

    if (key == ord('x') or key == ord('\x1b')):
        keep_processing = False

    # - if user presses "f" then switch to fullscreen

    elif (key == ord('f')):
        print("\n -- toggle fullscreen.")
        last_fs = cv2.getWindowProperty(window_name,
                                        cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                cv2.WINDOW_FULLSCREEN &
                                ~(int(last_fs)))


#####################################################################

# Author : Oscar Ryley, based on the outreach tasks by Toby Breckon
# Copyright (c) 2026 Dept Computer Science, Durham University, UK

#####################################################################