# one - Mujtaba & Richard & Sagha, Orange (10%), Green (15%), Contrast(+57), Brightness (+20), Hue (Blue 15%)
# two - Jessica & James, Saturation 65%, Brightness 52%, Sharpness 45%
# three - Freya & Josh, Contrast + Colour + Opposte Hue + Opposite Saturation
# four - Tommy & Elliot, Green, High contrast, saturation
# five - Ash & Mark, Uping Brightness + Vignette
# six - William & Vinnie, Blue Hue
# seven - Ayansh & Daniel, Hue 10% to right, 50% Saturation, 60% Value, Grey scale, 50% blur
# eight - Lewis & Jordon, Hue (do Red)
# nine - Sophie H & Sophie M, Neon Pink and blury

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_HEIGHT = 44

def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], LABEL_HEIGHT), (30, 30, 30), -1)
    cv2.putText(out, text, (14, 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def _list_cameras(max_indices=8):
    found = []
    for i in range(max_indices):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            found.append(i)
        if cap is not None:
            cap.release()
    return found



def increase_brightness(hsv_img):
    v_channel = hsv_img[..., 2].astype(np.float32) * 1.5
    hsv_img[..., 2] = np.clip(v_channel, 0, 255).astype(np.uint8)
    return hsv_img


# Helper image transforms (work in BGR or HSV as noted)
def apply_contrast_brightness_bgr(bgr, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)


def apply_vignette_bgr(bgr, strength=0.5):
    h, w = bgr.shape[:2]
    kx = cv2.getGaussianKernel(w, w * 0.5)
    ky = cv2.getGaussianKernel(h, h * 0.5)
    mask = ky * kx.T
    mask = mask / mask.max()
    mask = (1 - strength) + strength * mask
    out = bgr.astype(np.float32) * mask[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

def sharpen_bgr(bgr, strength=1.0):
    # strength scales how much sharpening we apply (blend original+sharpened)
    sharp = cv2.filter2D(bgr, -1, SHARPEN_KERNEL)
    return cv2.addWeighted(bgr, 1.0 - 0.4 * strength, sharp, 0.4 * strength, 0)


def hsv_safe_convert_and_modify(bgr, hue_shift=0, sat_scale=1.0, val_scale=1.0, set_hue=None, set_sat=None, set_val=None, invert_sat=False):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    if set_hue is not None:
        hsv[..., 0] = int(set_hue) % 180
    else:
        hsv[..., 0] = (hsv[..., 0] + int(hue_shift)) % 180

    if invert_sat:
        hsv[..., 1] = 255 - hsv[..., 1]
    if set_sat is not None:
        hsv[..., 1] = np.clip(int(set_sat), 0, 255)
    else:
        hsv[..., 1] = np.clip(hsv[..., 1] * float(sat_scale), 0, 255)

    if set_val is not None:
        hsv[..., 2] = np.clip(int(set_val), 0, 255)
    else:
        hsv[..., 2] = np.clip(hsv[..., 2] * float(val_scale), 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# FILTERS

def filter_one(img):
    # Orange/green accents, increased contrast and brightness, small hue shift
    b = hsv_safe_convert_and_modify(img, hue_shift=15, sat_scale=1.25, val_scale=1.1)
    b = apply_contrast_brightness_bgr(b, alpha=1.57, beta=20)
    return b


def filter_two(img):
    # Boost saturation, increase brightness, and add moderate sharpening
    b = hsv_safe_convert_and_modify(img, sat_scale=1.65)
    b = apply_contrast_brightness_bgr(b, alpha=1.0, beta=52)
    b = sharpen_bgr(b, strength=0.45)
    return b


def filter_three(img):
    # Contrast + colour + opposite hue + opposite saturation
    b = hsv_safe_convert_and_modify(img, hue_shift=90, sat_scale=1.2, invert_sat=True)
    b = apply_contrast_brightness_bgr(b, alpha=1.3, beta=0)
    return b


def filter_four(img):
    # Strong green tint with high contrast and saturation
    b = hsv_safe_convert_and_modify(img, set_hue=60, sat_scale=1.5, val_scale=1.05)
    b = apply_contrast_brightness_bgr(b, alpha=1.5, beta=0)
    return b


def filter_five(img):
    # Increase brightness and apply vignette for cinematic look
    b = hsv_safe_convert_and_modify(img, val_scale=1.25, sat_scale=1.05)
    b = apply_vignette_bgr(b, strength=0.6)
    b = apply_contrast_brightness_bgr(b, alpha=1.05, beta=10)
    return b


def filter_six(img):
    # Blue hue shift with a touch of saturation
    b = hsv_safe_convert_and_modify(img, set_hue=110, sat_scale=1.3)
    return b


def filter_seven(img):
    # Shift hue slightly right (10%), set mid saturation/value, grayscale-like blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # Build HSV: hue shifted by 18 (~10% of 180), sat 50%, value = blurred scaled to 60%
    h_chan = np.full_like(blurred, 18, dtype=np.uint8)
    s_chan = np.full_like(blurred, int(0.5 * 255), dtype=np.uint8)
    v_chan = np.clip((blurred.astype(np.float32) * 0.6), 0, 255).astype(np.uint8)
    hsv = cv2.merge([h_chan, s_chan, v_chan])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def filter_eight(img):
    # Red hue with boosted saturation
    b = hsv_safe_convert_and_modify(img, set_hue=0, sat_scale=1.4, val_scale=1.05)
    return b


def filter_nine(img):
    p = cv2.GaussianBlur(img, (15, 15), 0)
    
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = 160

    hsv[..., 1] = np.clip(hsv[..., 1] * 1.6 + 30, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.15 + 20, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


PANEL_FILTERS = [
    filter_one,
    filter_two,
    filter_three,
    filter_four,
    filter_five,
    filter_six,
    filter_seven,
    filter_eight,
    filter_nine,
]

def assemble_mosaic(frame, W=1280, H=720):
    # 3x3 grid
    cols = 3
    rows = 3
    panel_w = W // cols
    panel_h = H // rows
    mosaic = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(9):
        r = i // cols
        c = i % cols
        x = c * panel_w
        y = r * panel_h

        p = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        try:
            p = PANEL_FILTERS[i](p)
        except Exception:
            p = PANEL_FILTERS[0](p)

        p = label(p, f"Filter {i+1}")
        mosaic[y:y+panel_h, x:x+panel_w] = p

    return mosaic


if __name__ == '__main__':
    cams = _list_cameras(6)
    cam_idx = cams[0] if cams else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {cam_idx}")

    window_name = 'MultiFeed (9 panels)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"Using camera {cam_idx}. Controls: q/Esc to quit, f to toggle fullscreen")
    fullscreen = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            mosaic = assemble_mosaic(frame, W=1280, H=720)
            cv2.imshow(window_name, mosaic)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        cap.release()
        cv2.destroyAllWindows()
