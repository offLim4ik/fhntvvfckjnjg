#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from datetime import datetime

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from pyzbar import pyzbar
from clover.srv import SetLEDEffect


# ===== Константы и настройки распознавания =====
KERNEL = np.ones((5, 5), np.uint8)

COLORS_HSV = {
    "Red": [
        (np.array([0, 70, 50]), np.array([10, 255, 255])),
        (np.array([170, 70, 50]), np.array([179, 255, 255]))
    ],
    "Green": [
        (np.array([35, 70, 50]), np.array([85, 255, 255]))
    ],
    "Blue": [
        (np.array([100, 70, 50]), np.array([130, 255, 255]))
    ],
    "Yellow": [
        (np.array([20, 70, 50]), np.array([35, 255, 255]))
    ],
    "Orange": [
        (np.array([10, 70, 50]), np.array([20, 255, 255]))
    ],
    "White": [
        (np.array([0, 0, 200]), np.array([179, 40, 255]))
    ],
    "Black": [
        (np.array([0, 0, 0]), np.array([179, 255, 50]))
    ]
}


# ===== Функции распознавания фигур =====
def detect_shape(cnt):
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return None
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    v = len(approx)

    if v == 3:
        return "Triangle"
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h) if h else 0
        return "Square" if 0.9 <= ratio <= 1.1 else "Rectangle"
    if v == 6:
        return "Hexagon"
    if v > 6:
        return "Circle"
    return None


# ===== Основной класс ноды =====
class DroneScannerNode:
    def __init__(self):
        rospy.init_node("drone_color_shape_qr_scanner", anonymous=True)

        self.bridge = CvBridge()
        self.last_bgr = None

        self.camera_topic = rospy.get_param("~camera_topic", "/main_camera/image_raw_throttled")
        self.report_file = rospy.get_param("~report_file", "/home/pi/scan_report.txt")
        self.min_area = rospy.get_param("~min_area", 500)

        self.annotated_pub = rospy.Publisher("~annotated", Image, queue_size=1)
        self.info_pub = rospy.Publisher("~info", String, queue_size=10)

        rospy.wait_for_service("led/set_effect")
        self.set_effect = rospy.ServiceProxy("led/set_effect", SetLEDEffect)

        rospy.Subscriber(self.camera_topic, Image, self._image_cb, queue_size=1)
        self.scan_srv = rospy.Service("~scan_once", Trigger, self._scan_once_cb)

    def _image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr_throttle(5, "cv_bridge error: %s", str(e))
            return
        if img is None or img.size == 0:
            return
        self.last_bgr = img

    def _set_led(self, color_name):
        if color_name == "Red":
            self.set_effect(effect="fill", r=255, g=0, b=0)
        elif color_name == "Green":
            self.set_effect(effect="fill", r=0, g=255, b=0)
        elif color_name == "Blue":
            self.set_effect(effect="fill", r=0, g=0, b=255)
        elif color_name == "Yellow":
            self.set_effect(effect="fill", r=255, g=255, b=0)
        elif color_name == "Orange":
            self.set_effect(effect="fill", r=255, g=128, b=0)
        elif color_name == "White":
            self.set_effect(effect="fill", r=255, g=255, b=255)
        else:
            self.set_effect(effect="fill", r=0, g=0, b=0)

    def _scan_frame(self, frame_bgr):
        output = frame_bgr.copy()
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        h, w = frame_bgr.shape[:2]
        image_area = h * w

        color_objects = []

        for color_name, ranges in COLORS_HSV.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > image_area * 0.9:
                    continue

                shape = detect_shape(cnt)
                if not shape:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                label = f"{color_name} {shape}"

                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(output, label, (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                color_objects.append({"color": color_name, "shape": shape, "area": area})

        color_objects.sort(key=lambda d: d["area"], reverse=True)

        qr_objects = []
        for barcode in pyzbar.decode(frame_bgr):
            try:
                data = barcode.data.decode("utf-8")
            except Exception:
                data = str(barcode.data)

            qr_objects.append(data)

            x, y, bw, bh = barcode.rect
            cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(output, f"QR: {data}", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return output, color_objects, qr_objects

    def _write_report_overwrite(self, header, color_objects, qr_objects):
        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(header.strip() + "\n")
            for i, obj in enumerate(color_objects, start=1):
                f.write(f"color object {i}: {obj['color']} {obj['shape']}\n")
            for i, data in enumerate(qr_objects, start=1):
                f.write(f"qr object {i}: {data}\n")

    def _scan_once_cb(self, _req):
        if self.last_bgr is None:
            msg = "no camera frame"
            rospy.logwarn(msg)
            return TriggerResponse(success=False, message=msg)

        annotated, color_objects, qr_objects = self._scan_frame(self.last_bgr)

        try:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))
        except Exception as e:
            rospy.logerr("publish annotated error: %s", str(e))

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"input: {self.camera_topic}\ntime: {ts}\n"

        rospy.loginfo(header.strip())
        for i, obj in enumerate(color_objects, start=1):
            rospy.loginfo("color object %d: %s %s", i, obj["color"], obj["shape"])
        for i, data in enumerate(qr_objects, start=1):
            rospy.loginfo("qr object %d: %s", i, data)

        try:
            self._write_report_overwrite(header, color_objects, qr_objects)
        except Exception as e:
            rospy.logerr("write report error: %s", str(e))

        lines = []
        for i, obj in enumerate(color_objects, start=1):
            lines.append(f"color object {i}: {obj['color']} {obj['shape']}")
        for i, data in enumerate(qr_objects, start=1):
            lines.append(f"qr object {i}: {data}")
        info_text = " | ".join(lines) if lines else "no objects"
        self.info_pub.publish(info_text)

        best_color = color_objects[0]["color"] if color_objects else None
        self._set_led(best_color)

        return TriggerResponse(success=True, message=info_text)


if __name__ == "__main__":
    node = DroneScannerNode()
    rospy.spin()

#чисто для пуша и комита