#####################################################################

# Task X : track faces and overlay an image

#####################################################################

import cv2
import numpy as np

#####################################################################

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
worm_image = cv2.imread("worm.png", cv2.IMREAD_UNCHANGED)

camera.set(3, 1280)
camera.set(4, 720)

def overlay_image(frame, overlay, x, y):
    h, w = overlay.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    
    # Position overlay centered at (x, y)
    x_start = x - w // 2
    y_start = y - h // 2
    
    # Clip to frame boundaries
    x1_frame = max(0, x_start)
    y1_frame = max(0, y_start)
    x2_frame = min(w_frame, x_start + w)
    y2_frame = min(h_frame, y_start + h)
    
    # Get corresponding overlay coordinates
    x1_overlay = x1_frame - x_start
    y1_overlay = y1_frame - y_start
    x2_overlay = x1_overlay + (x2_frame - x1_frame)
    y2_overlay = y1_overlay + (y2_frame - y1_frame)
    
    # Extract regions with matching dimensions
    frame_region = frame[y1_frame:y2_frame, x1_frame:x2_frame]
    overlay_region = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    
    # Blend with alpha
    if overlay.shape[2] == 4:
        alpha = overlay_region[..., 3].astype(np.float32) / 255.0
        for c in range(3):
            frame_region[..., c] = (frame_region[..., c].astype(np.float32) * (1 - alpha) + 
                                     overlay_region[..., c].astype(np.float32) * alpha).astype(np.uint8)

while True:
    ret, frame = camera.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx = int(x + 0.5 * w)
        cy = int(y + 0.4 * h)
        if worm_image is not None:
            overlay_image(frame, worm_image, cx, cy)
    
    cv2.imshow("Face Tracking with Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
