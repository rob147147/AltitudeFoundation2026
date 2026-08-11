#####################################################################

# Task 2 : capture live video from an attached camera

#####################################################################

import cv2
import numpy as np

#####################################################################

# define video capture with access to camera 0

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# define display window

window_name = "Live Camera Input"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#####################################################################

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()

    # create our own 3x3 blur kernel 

    #kernel1 = np.array([[1, 1, 1],
    #                    [1, 1, 1],
    #                    [1, 1, 1]]) * 1/9
    #image = cv2.filter2D(image, -1, kernel1)


    # to create a larger blur kernel we can use a simple numpy function to help us
    
    #size = 7
    #kernel2 = np.ones((size,size),np.float32)/(size*size)
    #image = cv2.filter2D(image, -1, kernel2)


    # maybe we want to find sharp edges in our image - we call this edge detection

    #kernel3 = np.array([[-1, -1, -1],
    #                    [-1,  8, -1],
    #                    [-1, -1, -1]])
    #image = cv2.filter2D(image, -1, kernel3)


    # there is more than one way to achieve the same or similar results!

    #kernel4 = np.array([[0, 1, 0],
    #                    [1,-4, 1],
    #                    [0, 1, 0]])
    #image = cv2.filter2D(image, -1, kernel4)


    # display image

    cv2.imshow(window_name, image)

    # start the event loop - if user presses "x" or ESC then exit

    # wait 40ms or less for a key press from the user
    # (i.e. 1000ms / 25 fps = 40 ms)

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

# Author : Toby Breckon
# Copyright (c) 2022-25 Dept Computer Science, Durham University, UK

#####################################################################
