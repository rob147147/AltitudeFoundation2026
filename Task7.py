#####################################################################

# Task 7 : apply a filter to only part of an image, using a mask

#####################################################################

import cv2
import numpy as np

#####################################################################

# these are the same simple "knobs" from the Filters & Colour Grading
# lecture - brightness, contrast, hue and saturation - combined below
# into the named filter recipes from that lecture

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    # brightness: add/subtract from every pixel
    # contrast: stretch light/dark pixels apart (roughly -100 to 100)
    alpha = 1.0 + (contrast / 100.0)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=brightness)


def adjust_hue_saturation(img, hue_shift=0, saturation_scale=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

#####################################################################

# the filter recipes from the lecture - each one just combines the
# brightness / contrast / hue / saturation knobs above

def filter_golden_hour(img):
    b = adjust_hue_saturation(img, hue_shift=8, saturation_scale=0.9)
    b = adjust_brightness_contrast(b, brightness=15, contrast=0)
    return b


def filter_noir(img):
    b = adjust_hue_saturation(img, hue_shift=0, saturation_scale=0.0)
    b = adjust_brightness_contrast(b, brightness=0, contrast=20)
    return b


def filter_frost(img):
    b = adjust_hue_saturation(img, hue_shift=-10, saturation_scale=0.85)
    b = adjust_brightness_contrast(b, brightness=8, contrast=0)
    return b


def filter_punch(img):
    b = adjust_hue_saturation(img, hue_shift=0, saturation_scale=1.2)
    b = adjust_brightness_contrast(b, brightness=8, contrast=25)
    return b


def filter_fade(img):
    b = adjust_hue_saturation(img, hue_shift=0, saturation_scale=0.9)
    b = adjust_brightness_contrast(b, brightness=20, contrast=-15)
    return b


def filter_sepia(img):
    b = adjust_hue_saturation(img, hue_shift=12, saturation_scale=0.15)
    return b


def filter_invert(img):
    return cv2.bitwise_not(img)


def filter_posterize(img, levels=4):
    # round each colour channel down to a small number of steps
    step = 256 // levels
    table = np.array([(i // step) * step for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def filter_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

#####################################################################

# single-knob filters - these each expose ONE number you can change
# to see its effect directly, instead of a fixed recipe like above.
# try changing the values below, then re-run the code.

HUE_SHIFT_AMOUNT = 60          # try any number from -90 to 90
SATURATION_AMOUNT = 0.0        # try any number from 0.0 (grey) to 2.0 (vivid)
VALUE_AMOUNT = 1.6             # try any number from 0.2 (dark) to 2.0 (bright)
BRIGHTNESS_AMOUNT = 80         # try any number from -100 to 100


def filter_hue_only(img):
    # shifts every pixel's HUE round the colour wheel by one number -
    # change HUE_SHIFT_AMOUNT above and see the whole image change colour
    return adjust_hue_saturation(img, hue_shift=HUE_SHIFT_AMOUNT, saturation_scale=1.0)


def filter_saturation_only(img):
    # scales SATURATION only - change SATURATION_AMOUNT above
    # (0.0 = greyscale, 1.0 = unchanged, 2.0 = very vivid)
    return adjust_hue_saturation(img, hue_shift=0, saturation_scale=SATURATION_AMOUNT)


def filter_value_only(img):
    # scales VALUE (brightness in HSV) only - change VALUE_AMOUNT above
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * VALUE_AMOUNT, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def filter_brightness_only(img):
    # adds/subtracts BRIGHTNESS_AMOUNT from every pixel, in BGR -
    # change BRIGHTNESS_AMOUNT above (negative numbers make it darker)
    return adjust_brightness_contrast(img, brightness=BRIGHTNESS_AMOUNT, contrast=0)

#####################################################################

# choose which filter to apply - change this number (1-13) to try a
# different filter, or point apply_filter at one of the functions above.
# filters 10-13 are the easiest to experiment with - each one has a
# single number to change, above, rather than a fixed recipe

FILTER_NUMBER = 7

FILTERS = {
    1: filter_golden_hour,
    2: filter_noir,
    3: filter_frost,
    4: filter_punch,
    5: filter_fade,
    6: filter_sepia,
    7: filter_invert,
    8: filter_posterize,
    9: filter_thermal,
    10: filter_hue_only,
    11: filter_saturation_only,
    12: filter_value_only,
    13: filter_brightness_only,
}

apply_filter = FILTERS[FILTER_NUMBER]

#####################################################################

# choose whether the filter appears INSIDE the mask shape (False)
# or OUTSIDE it (True) - i.e. invert which part of the image the
# filter gets applied to

INVERT_MASK = False

#####################################################################

# define video capture with access to camera 0

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# define display window

window_name = "Filter + Mask"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

#####################################################################

keep_processing = True

while (keep_processing):

    # read an image from the camera

    _, image = camera.read()
    height, width, _ = image.shape
    cx, cy = width // 2, height // 2

    # apply the chosen filter to the WHOLE image first - we only keep
    # part of this filtered image once the mask below is applied

    filtered_image = apply_filter(image)

    # build a mask the same size as the image, starting all black (0)

    mask = np.zeros((height, width), dtype=np.uint8)

#####################################################################

    # choose ONE mask shape below by uncommenting it - only one shape
    # should be active at a time, so keep the others commented out

    # circle mask, centred on the middle of the image (currently active)

    cv2.circle(mask, (cx, cy), min(height, width) // 4, 255, -1)

    # rectangle mask, centred on the middle of the image

    # size = min(height, width) // 4
    # cv2.rectangle(mask, (cx - size, cy - size), (cx + size, cy + size), 255, -1)

    # oval mask, centred on the middle of the image

    # cv2.ellipse(mask, (cx, cy), (width // 5, height // 3), 0, 0, 360, 255, -1)

    # top half of the image only

    # cv2.rectangle(mask, (0, 0), (width, height // 2), 255, -1)

    # left half of the image only

    # cv2.rectangle(mask, (0, 0), (width // 2, height), 255, -1)

#####################################################################

    # invert the mask if requested - this swaps which part of the
    # image the filter gets applied to (see INVERT_MASK above)

    if INVERT_MASK:
        mask = cv2.bitwise_not(mask)

    # make sure the mask covers all three channels (BGR)

    mask_3ch = cv2.merge([mask, mask, mask])

    # combine: the FILTERED image where the mask is white (255),
    # the ORIGINAL image everywhere else

    result = np.where(mask_3ch == 255, filtered_image, image)

    # display image

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

# Author : Karl Southern, based on the outreach tasks by Toby Breckon
# Copyright (c) 2026 Dept Computer Science, Durham University, UK

#####################################################################
