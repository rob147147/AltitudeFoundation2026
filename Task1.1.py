#####################################################################

# Task 1 : capture live video from an attached camera

#####################################################################

import cv2

#####################################################################

# define video capture with access to camera 0

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# define display window

window_name = "Live Camera Input"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#####################################################################

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()

    # optional flip (1 = left/right; 0 = top/bottom; -1 = both)

    # image = cv2.flip(image, -1)

    # optional image blurring (changing the simgaX to a larger value increased blur)

    # image = cv2.GaussianBlur(image, (15, 15), 0)

    # display image

    cv2.imshow(window_name, image)

    # start the event loop - if user presses "x" or ESC then exit

    # wait 40ms or less for a key press from the user
    # (i.e. 1000ms / 25 fps = 40 ms)

    key = cv2.waitKey(40) & 0xFF

    if (key == ord('x') or key == ord('\x1b')):
        keep_processing = False

    if (key == ord('p')):
        # print out the pixel values
        # image.shape[1] will get the width of the image (number of columns)
        # image.shape[0] will get the height of the image (number of rows)
        # image[y,x] will get the pixel value at column x and row y
        ...

    if (key == ord('r')):
        # print out the pixel values for the red channel
        # cv2 stores pixels a bgr in BGR order (not RGB) so the red channel is channel 2
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                print(f"Pixel at ({x},{y}) = {image[y,x,2]}")
        ...

    if (key == ord('g')):
        # print out the pixel values for the green channel
        # cv2 stores pixels a bgr in BGR order (not RGB) so the green channel is channel 1
        ...

    if (key == ord('b')):
        # print out the pixel values for the blue channel
        # cv2 stores pixels a bgr in BGR order (not RGB) so the blue channel is channel 0
        ...

    if (key == ord('h')):
        # print out the pixel values for the hue channel
        # cv2 stores pixels in BGR order (not RGB) so we need to convert to HSV first
        # convert to HSV using image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # then the hue channel is channel 0 of the HSV image        
        ...

    if (key == ord('s')):
        # print out the pixel values for the saturation channel
        # cv2 stores pixels in BGR order (not RGB) so we need to convert to HSV first
        # convert to HSV using image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # then the saturation channel is channel 1 of the HSV image        
        ...

    if (key == ord('v')):
        # print out the pixel values for the value channel
        # cv2 stores pixels in BGR order (not RGB) so we need to convert to HSV first
        # convert to HSV using image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # then the value channel is channel 2 of the HSV image        
        ...


    # - if user presses "f" then switch to fullscreen

    elif (key == ord('f')):
        print("\n -- toggle fullscreen.")
        last_fs = cv2.getWindowProperty(window_name,
                                        cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN &
                              ~(int(last_fs)))


#####################################################################

# Author : Toby Breckon, Karl Southern
# Copyright (c) 2022-25 Dept Computer Science, Durham University, UK

#####################################################################
