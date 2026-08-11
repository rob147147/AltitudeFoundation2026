#####################################################################

# Task X : track faces and overlay an image

#####################################################################

import cv2
import numpy as np

#####################################################################

# define video capture with access to camera 1

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# define display window

window_name = "Live Camera to track faces and overlay an image"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# define the face classifier

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# get the overlay image

image_path = "worm.png"
overlay_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

#####################################################################

# overlay frame function
# used to place an image ontop of another at a certain x,y point

def overlay_frame(frame, overlay, x, y):
    h, w = overlay.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    
    # position overlay centered at (x, y)
    x_start = x - w // 2
    y_start = y - h // 2
    
    # clip to boundaries of the frame
    x1_frame = max(0, x_start)
    y1_frame = max(0, y_start)
    x2_frame = min(w_frame, x_start + w)
    y2_frame = min(h_frame, y_start + h)
    
    # get corresponding overlay coordinates
    x1_overlay = x1_frame - x_start
    y1_overlay = y1_frame - y_start
    x2_overlay = x1_overlay + (x2_frame - x1_frame)
    y2_overlay = y1_overlay + (y2_frame - y1_frame)
    
    # extract regions with matching dimensions
    frame_region = frame[y1_frame:y2_frame, x1_frame:x2_frame]
    overlay_region = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    
    # blend with alpha
    if overlay.shape[2] == 4:
        alpha = overlay_region[..., 3].astype(np.float32) / 255.0
        for c in range(3):
            frame_region[..., c] = (frame_region[..., c].astype(np.float32) * (1 - alpha) + 
                                     overlay_region[..., c].astype(np.float32) * alpha).astype(np.uint8)

#####################################################################

keep_processing = True

while (keep_processing):
    # read an image from the camera
    
    _, image = camera.read()

    # detect faces, by applying the face cascade to a greyscale version of the image
    # faces is a list of face points, with width and height
    
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 5, minSize=(30, 30))

    # check if there are any faces to overlay the image onto

    if len(faces) > 0:

        # find the largest face, and then overlay the image onto the frame at that point

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx = int(x + 0.5 * w)
        cy = int(y + 0.4 * h)
        overlay_frame(image, overlay_image, cx, cy)

        # for each face, overlay the image onto the frame

        # for face in faces:
        #     x, y, w, h = face
        #     cx = int(x + 0.5 * w)
        #     cy = int(y + 0.4 * h)
        #     overlay_frame(image, overlay_image, cx, cy)
    

    # display image with image overlayed if a face is found
    
    cv2.imshow(window_name, image)

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