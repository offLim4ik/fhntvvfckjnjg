#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import rospy
import cv2
import numpy as np
from datetime import datetime

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from pyzbar import pyzbar

from clover import srv
from clover.srv import SetLEDEffect
from led_msgs.srv import SetLEDs
from led_msgs.msg import LEDState
from std_srvs.srv import Trigger as TriggerSrv


F = "Kalashnikov"
I = "Vyacheslav"

FOLDER_NAME = f"Module_G_{F}_{I}"
REPORT_NAME = f"otchet_{F}_{I}.txt"

# По ТЗ обычно ожидают /Topic_Scan_F_I (с ведущим /)
TOPIC_SCAN = f"/Topic_Scan_{F}_{I}"
TOPIC_SCAN_IMAGE = f"{TOPIC_SCAN}/image"

CAMERA_TOPIC = "main_camera/image_raw_throttled"
ARUCO_MAP_FRAME = "aruco_map"

BASE_DIR = f"/home/pi/{FOLDER_NAME}"
REPORT_PATH = os.path.join(BASE_DIR, REPORT_NAME)

KERNEL = np.ones((5, 5), np.uint8)

# Твои HSV диапазоны
COLORS_HSV = {
    "Red": [
        (np.array([0, 82, 95]), np.array([180, 255, 128])),
    ],
    "Green": [
        (np.array([67, 95, 82]), np.array([180, 208, 145])),
    ],
    "Blue": [
        (np.array([81, 71, 71]), np.array([180, 179, 117])),
    ]
}


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


def clamp8(x):
    return int(max(0, min(255, x)))


def hsv_to_rgb(h, s, v):
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return r, g, b


class LedController:
    def __init__(self, led_count=72):
        self.led_count = led_count
        rospy.wait_for_service("led/set_effect")
        rospy.wait_for_service("led/set_leds")
        self.set_effect = rospy.ServiceProxy("led/set_effect", SetLEDEffect)
        self.set_leds = rospy.ServiceProxy("led/set_leds", SetLEDs, persistent=True)

    def fill(self, r, g, b):
        self.set_effect(effect="fill", r=clamp8(r), g=clamp8(g), b=clamp8(b))

    def rainbow(self):
        states = []
        for i in range(self.led_count):
            h = int((179.0 * i) / max(1, self.led_count - 1))
            r, g, b = hsv_to_rgb(h, 255, 255)
            states.append(LEDState(index=i, r=r, g=g, b=b))
        self.set_leds(states)

    def by_color_name(self, color_name):
        if color_name == "Red":
            self.fill(255, 0, 0)
        elif color_name == "Green":
            self.fill(0, 255, 0)
        elif color_name == "Blue":
            self.fill(0, 0, 255)
        else:
            self.fill(0, 0, 0)


class Scanner:
    def __init__(self, led: LedController, report_path: str, topic_name: str, image_topic: str,
                 camera_topic: str):
        self.bridge = CvBridge()
        self.led = led

        self.report_path = report_path
        self.topic_name = topic_name
        self.image_topic = image_topic
        self.camera_topic = camera_topic

        self.last_bgr = None
        self.min_area = rospy.get_param("~min_area", 500)

        # Топик по ТЗ + картинка (дополнительно)
        self.pub = rospy.Publisher(self.topic_name, String, queue_size=10, latch=True)
        self.pub_img = rospy.Publisher(self.image_topic, Image, queue_size=1)

        rospy.Subscriber(self.camera_topic, Image, self._image_cb, queue_size=1)

        self.lines_total = []
        self.scan_count = 0

        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        # очищаем отчёт при старте
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("")

        # сервис одноразового скана
        self.scan_srv = rospy.Service("scan_once", Trigger, self._scan_once_cb)

        self.pub.publish("scanner started")

    def _image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr_throttle(5, "cv_bridge error: %s", str(e))
            return
        if img is None or img.size == 0:
            return

        self.last_bgr = img

        # публикуем аннотированную картинку постоянно (это ок)
        color_objects, qr_objects, vis = self._scan_frame(img, draw=True)
        if vis is not None:
            try:
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
            except Exception as e:
                rospy.logerr_throttle(5, "publish image error: %s", str(e))

    def _scan_frame(self, frame_bgr, draw=False):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]
        image_area = h * w

        vis = frame_bgr.copy() if draw else None
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

                x, y, ww, hh = cv2.boundingRect(cnt)
                color_objects.append({"color": color_name, "shape": shape, "area": area, "bbox": (x, y, ww, hh)})

                if draw and vis is not None:
                    cv2.rectangle(vis, (x, y), (x + ww, y + hh), (255, 255, 255), 2)
                    cv2.putText(vis, f"{color_name} {shape}", (x, max(0, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        color_objects.sort(key=lambda d: d["area"], reverse=True)

        qr_objects = []
        for barcode in pyzbar.decode(frame_bgr):
            try:
                data = barcode.data.decode("utf-8")
            except Exception:
                data = str(barcode.data)
            qr_objects.append(data)

            if draw and vis is not None:
                x, y, ww, hh = barcode.rect.left, barcode.rect.top, barcode.rect.width, barcode.rect.height
                cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 255), 2)
                cv2.putText(vis, f"QR: {data}", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if draw and vis is not None:
            cv2.putText(vis, self.topic_name, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return color_objects, qr_objects, vis

    def _write_report_overwrite(self):
        with open(self.report_path, "w", encoding="utf-8") as f:
            for line in self.lines_total:
                f.write(line + "\n")

    def _publish_objects(self, color_objects, qr_objects):
        # В топик по ТЗ: либо "Color Shape", либо "qr-code:data"
        for obj in color_objects:
            self.pub.publish(f"{obj['color']} {obj['shape']}")
        for data in qr_objects:
            self.pub.publish(f"qr-code:{data}")

    def _format_lines(self, color_objects, qr_objects):
        # В терминал/отчёт по ТЗ:
        # color object 1: color type
        # qr-code object 3: data
        out = []
        idx = 1
        for obj in color_objects:
            out.append(f"color object {idx}: {obj['color']} {obj['shape']}")
            idx += 1
        for data in qr_objects:
            out.append(f"qr-code object {idx}: {data}")
            idx += 1
        return out

    def scan_once(self):
        if self.last_bgr is None:
            rospy.logwarn("no camera frame")
            self.pub.publish("no camera frame")
            return False

        color_objects, qr_objects, _ = self._scan_frame(self.last_bgr, draw=False)

        # публикация в /Topic_Scan_...
        self._publish_objects(color_objects, qr_objects)

        # лог в терминал
        lines = self._format_lines(color_objects, qr_objects)
        if not lines:
            lines = ["no objects detected"]
        for line in lines:
            rospy.loginfo(line)

        # LED при скане цветного объекта
        best_color = color_objects[0]["color"] if color_objects else None
        if best_color is not None:
            self.led.by_color_name(best_color)

        # отчёт в txt
        self.scan_count += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lines_total.append(f"scan #{self.scan_count} time: {ts}")
        self.lines_total.extend(lines)
        self.lines_total.append("")
        self._write_report_overwrite()

        return True

    def _scan_once_cb(self, _req):
        ok = self.scan_once()
        return TriggerResponse(success=bool(ok), message="ok" if ok else "no frame")


class Flight:
    def __init__(self, led: LedController, scanner: Scanner):
        self.led = led
        self.scanner = scanner

        rospy.wait_for_service("navigate")
        rospy.wait_for_service("get_telemetry")
        rospy.wait_for_service("land")

        self.navigate = rospy.ServiceProxy("navigate", srv.Navigate)
        self.get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)
        self.land = rospy.ServiceProxy("land", TriggerSrv)

    def navigate_wait(self, x=0, y=0, z=0, speed=0.5, frame_id="body", auto_arm=False):
        res = self.navigate(x=x, y=y, z=z, yaw=float("nan"), speed=speed, frame_id=frame_id, auto_arm=auto_arm)
        if not res.success:
            raise Exception(res.message)

        while not rospy.is_shutdown():
            telem = self.get_telemetry(frame_id="navigate_target")
            if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < 0.2:
                return
            rospy.sleep(0.2)

    # Поиск 1: половина красный / половина зеленый (как у тебя)
    def popolam1(self):
        states = []
        for i in range(72):
            if i < 36:
                states.append(LEDState(index=int(i), r=255, g=0, b=0))
            else:
                states.append(LEDState(index=int(i), r=0, g=255, b=0))
        self.led.set_leds(states)

    # Поиск 2: половина синий / половина белый (как у тебя)
    def popolam2(self):
        states = []
        for i in range(72):
            if i < 36:
                states.append(LEDState(index=int(i), r=0, g=0, b=255))
            else:
                states.append(LEDState(index=int(i), r=255, g=255, b=255))
        self.led.set_leds(states)

    def run(self):
        z = 0.75
        spd = 0.5

        # Взлет: синий
        self.led.fill(0, 0, 255)
        self.navigate_wait(x=0, y=0, z=z, speed=spd, frame_id="body", auto_arm=True)

        # Поиск 1
        self.popolam1()
        self.navigate_wait(x=1.5, y=0.5, z=z, speed=spd, frame_id="aruco_map")
        self.scanner.scan_once()  # скан 1
        rospy.sleep(2)

        # Поиск 2
        self.popolam2()
        self.navigate_wait(x=2, y=1, z=z, speed=spd, frame_id="aruco_map")
        rospy.sleep(2)

        self.navigate_wait(x=2.5, y=1.5, z=z, speed=spd, frame_id="aruco_map")
        self.scanner.scan_once()  # скан 2
        rospy.sleep(2)

        # Поиск 3
        self.led.rainbow()
        self.navigate_wait(x=2.5, y=2, z=z, speed=spd, frame_id="aruco_map")
        rospy.sleep(2)

        self.navigate_wait(x=0, y=2, z=z, speed=spd, frame_id="aruco_map")
        rospy.sleep(2)

        self.navigate_wait(x=0.5, y=1.5, z=z, speed=spd, frame_id="aruco_map")
        self.scanner.scan_once()  # скан 3
        rospy.sleep(2)

        # Домой
        self.navigate_wait(x=0, y=0, z=z, speed=spd, frame_id="aruco_map")

        # Посадка: зеленый
        self.led.fill(0, 255, 0)
        self.land()


if __name__ == "__main__":
    rospy.init_node(f"module_g_{F.lower()}_{I.lower()}", anonymous=True)

    led_count = rospy.get_param("~led_count", 72)
    led = LedController(led_count=led_count)

    scanner = Scanner(
        led=led,
        report_path=REPORT_PATH,
        topic_name=TOPIC_SCAN,
        image_topic=TOPIC_SCAN_IMAGE,
        camera_topic=CAMERA_TOPIC
    )

    flight = Flight(led=led, scanner=scanner)

    try:
        flight.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("error: %s", str(e))
        try:
            led.fill(0, 0, 0)
        except Exception:
            pass
