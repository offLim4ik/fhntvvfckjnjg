import cv2
import numpy as np

kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

def detect_color(hsv):
    colors = {
        'red': ((0, 184, 206), (180, 255, 255)),
        'green': ((44, 84, 200), (67, 192, 255)),
        'blue': ((94, 93, 67), (106, 157, 97))
    }

    best_cnt = None
    best_area = 0
    best_color = None

    for name, (low, high) in colors.items():
        mask = cv2.inRange(
            hsv,
            np.array(low, dtype=np.uint8),
            np.array(high, dtype=np.uint8)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 50 and area > best_area:
                best_cnt = c
                best_area = area
                best_color = name

    if best_cnt is None:
        return None, None, None, None

    m = cv2.moments(best_cnt)
    if m['m00'] == 0:
        return None, None, None, None

    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])

    return best_color, cx, cy, best_cnt
