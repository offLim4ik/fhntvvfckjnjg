#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import cv2
import numpy as np
from datetime import datetime

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from pyzbar import pyzbar

from clover.srv import Navigate
from clover.srv import SetLEDEffect
from led_msgs.srv import SetLEDs
from led_msgs.msg import LEDState


class ColorShapeScannerNode:
    def __init__(self):
        rospy.init_node('color_shape_scanner', anonymous=True)

        self.bridge = CvBridge()
        self.kernel = np.ones((5, 5), np.uint8)

        # HSV ranges (как у тебя)
        self.colors_hsv = {
            "Red": [(np.array([0, 62, 80]), np.array([17, 197, 162]))],
            "Green": [(np.array([34, 131, 0]), np.array([73, 255, 197]))],
            "Blue": [(np.array([96, 103, 56]), np.array([113, 255, 104]))],
        }

        # Новый топик с размеченной картинкой + топик с текстом (цвет/фигура/QR)
        self.annotated_pub = rospy.Publisher('~annotated', Image, queue_size=1)
        self.info_pub = rospy.Publisher('~info', String, queue_size=10)

        # Храним последний кадр, а распознавание делаем ТОЛЬКО по вызову сервиса
        self.last_image_bgr = None
        self.last_image_stamp = None
        rospy.Subscriber('/main_camera/image_raw_throttled', Image, self._image_cb, queue_size=1)

        # LED/навигация сервисы
        rospy.wait_for_service('led/set_effect')
        rospy.wait_for_service('led/set_leds')
        self.set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)
        self.set_leds = rospy.ServiceProxy('led/set_leds', SetLEDs, persistent=True)

        # Сервис “сканировать один раз”: по вызову распознаёт, публикует, пишет в файл, красит ленту
        self.scan_srv = rospy.Service('~scan_once', Trigger, self._scan_once_srv)

        # Лог-файл
        self.log_file = rospy.get_param('~log_file', '/tmp/scan_log.txt')

        rospy.loginfo("ColorShapeScannerNode started.")
        rospy.loginfo("Topics:")
        rospy.loginfo("  %s", rospy.resolve_name('~annotated'))
        rospy.loginfo("  %s", rospy.resolve_name('~info'))
        rospy.loginfo("Service:")
        rospy.loginfo("  %s", rospy.resolve_name('~scan_once'))
        rospy.loginfo("Log file: %s", self.log_file)

    def _image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr_throttle(5, f"cv_bridge error: {e}")
            return
        if img is None or img.size == 0:
            return
        self.last_image_bgr = img
        self.last_image_stamp = msg.header.stamp

    def _shape_from_contour(self, cnt):
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

    def _detect(self, image_bgr):
        output = image_bgr.copy()
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        h, w = image_bgr.shape[:2]
        image_area = h * w

        detections = []  # список строк "Color Shape"
        best = None       # (area, color, shape, bbox, contour)

        for color_name, ranges in self.colors_hsv.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500 or area > image_area * 0.9:
                    continue

                shape = self._shape_from_contour(cnt)
                if shape is None:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                label = f"{color_name} {shape}"
                detections.append(label)

                # рисуем
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(output, label, (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if best is None or area > best[0]:
                    best = (area, color_name, shape, (x, y, bw, bh), cnt)

        # QR
        qr_list = []
        for barcode in pyzbar.decode(image_bgr):
            try:
                b_data = barcode.data.decode("utf-8")
            except Exception:
                b_data = str(barcode.data)
            qr_list.append(b_data)

            # обводка QR
            x, y, bw, bh = barcode.rect
            cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(output, f"QR: {b_data}", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        best_color = best[1] if best else None
        best_shape = best[2] if best else None

        return output, detections, qr_list, best_color, best_shape

    def _set_led_by_color(self, color_name):
        # красим ленту ТОЛЬКО когда скан вызвали
        if color_name == "Red":
            self.set_effect(effect='fill', r=255, g=0, b=0)
        elif color_name == "Green":
            self.set_effect(effect='fill', r=0, g=255, b=0)
        elif color_name == "Blue":
            self.set_effect(effect='fill', r=0, g=0, b=255)
        else:
            self.set_effect(effect='fill', r=0, g=0, b=0)

    def _append_log(self, text_line):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text_line + "\n")
        except Exception as e:
            rospy.logerr_throttle(5, f"Log write error: {e}")

    def _scan_once_srv(self, _req):
        if self.last_image_bgr is None:
            msg = "Нет кадра с камеры. Подпишись на /main_camera/image_raw_throttled и подожди 1-2 секунды."
            rospy.logwarn(msg)
            return TriggerResponse(success=False, message=msg)

        annotated, detections, qr_list, best_color, best_shape = self._detect(self.last_image_bgr)

        # Публикация размеченной картинки
        try:
            self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(annotated, 'bgr8'))
        except Exception as e:
            rospy.logerr(f"Publish annotated error: {e}")

        # Текстовая инфа в топик + консоль
        det_str = "; ".join(detections) if detections else "None"
        qr_str = "; ".join(qr_list) if qr_list else "None"
        info_text = f"Detections: {det_str} | Best: {best_color or 'None'} {best_shape or ''}".strip() + f" | QR: {qr_str}"

        self.info_pub.publish(info_text)
        rospy.loginfo(info_text)

        # LED по лучшей найденной фигуре/цвету
        if best_color is None:
            rospy.logwarn("Цвет/фигура не обнаружены — LED выключены.")
        self._set_led_by_color(best_color)

        # Запись в txt
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_log(f"[{ts}] {info_text}")

        return TriggerResponse(success=True, message=info_text)

    # оставил твою “пополам” — на всякий
    def popolam(self):
        for i in range(72):
            if i < 24:
                self.set_leds([LEDState(index=int(i), r=255, g=255, b=255)])
            elif 23 < i < 48:
                self.set_leds([LEDState(index=int(i), r=0, g=0, b=255)])
            else:
                self.set_leds([LEDState(index=int(i), r=255, g=0, b=0)])


def navigate_wait(navigate_srv, get_telemetry_srv, x=0, y=0, z=0, speed=0.5, frame_id='body', auto_arm=False):
    res = navigate_srv(x=x, y=y, z=z, yaw=float('nan'), speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    if not res.success:
        raise Exception(res.message)

    while not rospy.is_shutdown():
        telem = get_telemetry_srv(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < 0.2:
            return
        rospy.sleep(0.2)


def polet_with_scans(scanner: ColorShapeScannerNode):
    """
    Полёт как у тебя, но:
    - распознавание/LED срабатывают ТОЛЬКО по вызову сервиса ~scan_once
    """
    rospy.wait_for_service('navigate')
    rospy.wait_for_service('land')
    rospy.wait_for_service(rospy.resolve_name('~scan_once'))

    navigate = rospy.ServiceProxy('navigate', Navigate)
    land = rospy.ServiceProxy('land', Trigger)
    scan_once = rospy.ServiceProxy(rospy.resolve_name('~scan_once'), Trigger)

    rospy.loginfo("fly...")
    navigate(x=0, y=0, z=0.75, frame_id='body', auto_arm=True)
    scanner.popolam()
    rospy.sleep(5)

    rospy.loginfo("point A (1, 1)...")
    navigate(x=1, y=1, z=0.75, frame_id='aruco_map', speed=0.5)
    rospy.sleep(7)
    scan_once()         # <-- скан + LED + лог
    rospy.sleep(3)
    scanner.popolam()

    rospy.loginfo("point B (3, 1)...")
    navigate(x=3, y=1, z=0.75, frame_id='aruco_map', speed=0.5)
    rospy.sleep(7)
    scan_once()
    rospy.sleep(3)
    scanner.popolam()

    rospy.loginfo("point C (0, 2)...")
    navigate(x=0, y=2, z=0.75, frame_id='aruco_map', speed=0.5)
    rospy.sleep(7)
    scan_once()
    rospy.sleep(3)
    scanner.popolam()

    rospy.loginfo("DOMOY (0, 0)...")
    navigate(x=0, y=0, z=0.75, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)

    land()
    scanner.set_effect(r=0, g=0, b=0, effect='fill')
    rospy.loginfo("posadka")


if __name__ == '__main__':
    scanner = ColorShapeScannerNode()
    try:
        # Если хочешь только сканер — закомментируй следующую строку и просто rospy.spin()
        polet_with_scans(scanner)
        # rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            # вернуть “пополам” в конце (как у тебя)
            scanner.popolam()
        except Exception:
            pass
