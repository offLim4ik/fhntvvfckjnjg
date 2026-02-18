#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import rospy
import cv2
import numpy as np
from datetime import datetime
import time
import pigpio

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

TOPIC_SCAN = f"Topic_Scan_{F}_{I}"
TOPIC_SCAN_IMAGE = f"{TOPIC_SCAN}/image"

CAMERA_TOPIC = "main_camera/image_raw_throttled"

ARUCO_MAP_FRAME = "aruco_map"

# ТВОЙ ПУТЬ ДО КАРТЫ
MAP_FILE = "/home/pi/catkin_ws/src/clover/aruco_pose/map/test_map.txt"

BASE_DIR = f"/home/pi/{FOLDER_NAME}"
REPORT_PATH = os.path.join(BASE_DIR, REPORT_NAME)

KERNEL = np.ones((5, 5), np.uint8)

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

# ===== PIGPIO / SERVO RESET =====
SERVO_GPIO = 13
SERVO_PULSE_SBROS = 2500
SERVO_DELAY_BEFORE = 2.0


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


class ServoController:
    """
    Управление сервой через pigpio.
    Делает sbros(): ждёт 2 сек и выставляет pulsewidth 2500 на GPIO13.
    """
    def __init__(self, gpio=SERVO_GPIO):
        self.gpio = int(gpio)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio.pi() not connected. Проверь, что запущен демон: sudo systemctl start pigpiod")
        self.pi.set_mode(self.gpio, pigpio.OUTPUT)

    def sbros(self):
        time.sleep(SERVO_DELAY_BEFORE)
        self.pi.set_servo_pulsewidth(self.gpio, SERVO_PULSE_SBROS)

    def stop(self):
        try:
            # по желанию: можно обнулить импульс, чтобы отпустить серву
            # self.pi.set_servo_pulsewidth(self.gpio, 0)
            self.pi.stop()
        except Exception:
            pass


class ArucoMap:
    """
    Читает карту Clover формата:
    # id length x y z rot_z rot_y rot_x
    <id> <len> <x> <y> <z> <rot_z> <rot_y> <rot_x>
    """

    def __init__(self, map_file: str, map_frame: str = ARUCO_MAP_FRAME):
        self.map_frame = map_frame
        self.map_file = map_file
        self.markers = []  # [{"id":int,"x":float,"y":float,"z":float}]

        rospy.wait_for_service("get_telemetry")
        self.get_telemetry = rospy.ServiceProxy("get_telemetry", srv.GetTelemetry)

        self._load_map()

    def _load_map(self):
        if not os.path.exists(self.map_file):
            rospy.logwarn("ArucoMap: map file not found: %s", self.map_file)
            self.markers = []
            return

        markers = []
        bad = 0

        with open(self.map_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#") or s.startswith("//"):
                    continue

                s = s.replace(",", " ")
                parts = [p for p in s.split() if p]

                if len(parts) < 5:
                    bad += 1
                    continue

                try:
                    mid = int(float(parts[0]))
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    markers.append({"id": mid, "x": x, "y": y, "z": z})
                except Exception:
                    bad += 1
                    continue

        self.markers = markers
        rospy.loginfo("ArucoMap: loaded %d markers from %s (bad lines: %d)", len(markers), self.map_file, bad)

    def get_drone_xyz_in_map(self):
        try:
            t = self.get_telemetry(frame_id=self.map_frame)
            return float(t.x), float(t.y), float(t.z)
        except Exception:
            return None

    def get_nearest_marker(self, use_xy_only=True):
        if not self.markers:
            return None, None, None

        drone = self.get_drone_xyz_in_map()
        if drone is None:
            return None, None, None

        dx, dy, dz = drone

        best = None
        for m in self.markers:
            mx, my, mz = m["x"], m["y"], m["z"]
            if use_xy_only:
                dist = math.sqrt((mx - dx) ** 2 + (my - dy) ** 2)
            else:
                dist = math.sqrt((mx - dx) ** 2 + (my - dy) ** 2 + (mz - dz) ** 2)
            if best is None or dist < best[0]:
                best = (dist, m)

        if best is None:
            return None, None, None

        dist, m = best
        return m["id"], (m["x"], m["y"], m["z"]), dist


class Scanner:
    def __init__(self, led: LedController, servo: ServoController, report_path: str, topic_name: str,
                 image_topic: str, camera_topic: str, aruco_map: ArucoMap):
        self.bridge = CvBridge()
        self.led = led
        self.servo = servo
        self.aruco_map = aruco_map

        self.report_path = report_path
        self.topic_name = topic_name
        self.image_topic = image_topic
        self.camera_topic = camera_topic

        self.last_bgr = None
        self.min_area = rospy.get_param("~min_area", 500)

        self.pub = rospy.Publisher(self.topic_name, String, queue_size=10, latch=True)
        self.pub_img = rospy.Publisher(self.image_topic, Image, queue_size=1)

        rospy.Subscriber(self.camera_topic, Image, self._image_cb, queue_size=1)

        self.lines_total = []
        self.scan_count = 0

        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("")

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
        for obj in color_objects:
            self.pub.publish(f"{obj['color']} {obj['shape']}")
        for data in qr_objects:
            self.pub.publish(f"qr-code:{data}")

    def _format_lines(self, color_objects, qr_objects):
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
        """
        Возвращает цель полёта по карте:
          dict {"id": mid, "x": x, "y": y, "dist": dist, "best_color": best_color}
          или None
        """
        if self.last_bgr is None:
            rospy.logwarn("no camera frame")
            self.pub.publish("no camera frame")
            return None

        color_objects, qr_objects, _ = self._scan_frame(self.last_bgr, draw=False)

        self._publish_objects(color_objects, qr_objects)

        lines = self._format_lines(color_objects, qr_objects)
        for line in lines:
            rospy.loginfo(line)

        best_color = color_objects[0]["color"] if color_objects else None
        if best_color is not None:
            # 1) включаем подсветку по цвету
            self.led.by_color_name(best_color)

            # 2) и сразу выполняем sbros() по твоей логике
            try:
                rospy.loginfo("color scanned (%s) -> sbros()", best_color)
                self.pub.publish(f"color scanned:{best_color} -> sbros()")
                self.servo.sbros()
            except Exception as e:
                rospy.logerr("sbros error: %s", str(e))
                self.pub.publish(f"sbros error:{str(e)}")

        mid, xyz, dist = self.aruco_map.get_nearest_marker(use_xy_only=True)

        if mid is None or xyz is None:
            marker_line = "nearest aruco from FILE MAP: none"
            rospy.logwarn(marker_line)
            self.pub.publish(marker_line)
            target = None
        else:
            x, y, z_map = xyz
            marker_line = f"nearest aruco from FILE MAP: id={mid}, x={x:.2f}, y={y:.2f}, dist={dist:.2f}"
            rospy.loginfo(marker_line)
            self.pub.publish(marker_line)
            target = {"id": mid, "x": x, "y": y, "dist": dist, "best_color": best_color}

        self.scan_count += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lines_total.append(f"scan #{self.scan_count} time: {ts}")
        self.lines_total.extend(lines)
        self.lines_total.append(marker_line)
        self.lines_total.append("")
        self._write_report_overwrite()

        return target

    def _scan_once_cb(self, _req):
        self.scan_once()
        return TriggerResponse(success=True, message="ok")


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

    def popolam1(self):
        states = []
        for i in range(72):
            if i < 36:
                states.append(LEDState(index=int(i), r=255, g=0, b=0))
            else:
                states.append(LEDState(index=int(i), r=0, g=255, b=0))
        self.led.set_leds(states)

    def popolam2(self):
        states = []
        for i in range(72):
            if i < 36:
                states.append(LEDState(index=int(i), r=0, g=0, b=255))
            else:
                states.append(LEDState(index=int(i), r=255, g=255, b=255))
        self.led.set_leds(states)

    def fly_to_marker_from_scan(self, z, spd):
        target = self.scanner.scan_once()
        if target is None:
            rospy.logwarn("scan: no marker target from FILE MAP")
            return False

        x = target["x"]
        y = target["y"]
        mid = target["id"]

        rospy.loginfo("fly to FILE MAP marker id=%s at x=%.2f y=%.2f (aruco_map)", str(mid), x, y)

        self.navigate_wait(x=x, y=y, z=z, speed=spd, frame_id="aruco_map")
        return True

    def run(self):
        z = 0.75
        spd = 0.5

        self.led.fill(0, 0, 255)
        self.navigate_wait(x=0, y=0, z=z, speed=spd, frame_id="body", auto_arm=True)

        self.popolam1()
        self.navigate_wait(x=1.5, y=0.5, z=z, speed=spd, frame_id="aruco_map")
        self.fly_to_marker_from_scan(z=z, spd=spd)
        rospy.sleep(2)

        self.popolam2()
        self.navigate_wait(x=2, y=1, z=z, speed=spd, frame_id="aruco_map")
        rospy.sleep(2)

        self.navigate_wait(x=2.5, y=1.5, z=z, speed=spd, frame_id="aruco_map")
        self.fly_to_marker_from_scan(z=z, spd=spd)
        rospy.sleep(2)

        self.led.rainbow()
        self.navigate_wait(x=2.5, y=2, z=z, speed=spd, frame_id="aruco_map")
        rospy.sleep(2)

        self.navigate_wait(x=0, y=2, z=z, speed=spd, frame_id="aruco_map")
        rospy.sleep(2)

        self.navigate_wait(x=0.5, y=1.5, z=z, speed=spd, frame_id="aruco_map")
        self.fly_to_marker_from_scan(z=z, spd=spd)
        rospy.sleep(2)

        self.navigate_wait(x=0, y=0, z=z, speed=spd, frame_id="aruco_map")

        self.led.fill(0, 255, 0)
        self.land()


if __name__ == "__main__":
    rospy.init_node(f"module_g_{F.lower()}_{I.lower()}", anonymous=True)

    servo = None
    try:
        led_count = rospy.get_param("~led_count", 72)
        led = LedController(led_count=led_count)

        # Servo init (pigpio)
        servo = ServoController(gpio=SERVO_GPIO)

        aruco_map = ArucoMap(map_file=MAP_FILE, map_frame=ARUCO_MAP_FRAME)

        scanner = Scanner(
            led=led,
            servo=servo,
            report_path=REPORT_PATH,
            topic_name=TOPIC_SCAN,
            image_topic=TOPIC_SCAN_IMAGE,
            camera_topic=CAMERA_TOPIC,
            aruco_map=aruco_map
        )

        flight = Flight(led=led, scanner=scanner)

        flight.run()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("error: %s", str(e))
        try:
            if 'led' in locals():
                led.fill(0, 0, 0)
        except Exception:
            pass
    finally:
        try:
            if servo is not None:
                servo.stop()
        except Exception:
            pass
