import rospy
import cv2
import numpy as np
from math import nan
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge
from clover import long_callback, srv
import tf2_ros
import tf2_geometry_msgs
from std_srvs.srv import Trigger
import math
from pyzbar.pyzbar import decode as qr_read
from clover.srv import SetLEDEffect

rospy.init_node('flight')

set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)  # define proxy to ROS-service

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

bridge = CvBridge()

mask_pub = rospy.Publisher('~top', Image, queue_size=1)

last_pos = [0, 0, '']


def navigate_wait(x=0, y=0, z=0, yaw=math.nan, speed=0.5, frame_id='body', tolerance=0.2, auto_arm=False):
    global last_pos
    res = navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    last_pos = [x, y, frame_id]
    if not res.success:
        return res

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            return res
        rospy.sleep(0.2)


@long_callback
def image_callback(msg):
    global last_pos
    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    barcodes = qr_read(img)
    for barcode in barcodes:
        set_effect(r=255, g=0, b=0, effect='blink')
        navigate(x=0, y=0, z=0, frame_id='body')
        rospy.sleep(0.1)
        print(barcode.data.decode('utf-8'))
        rospy.sleep(5)
        set_effect(r=255, g=255, b=0, effect='blink')
        navigate(x=last_pos[0], y=last_pos[1], frame_id=last_pos[2])
        rospy.sleep(5)


colors = set()


def image_callback_2(msg):
    global colors
    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red = cv2.inRange(img_hsv, (146, 88, 94), (255, 255, 255))
    green = cv2.inRange(img_hsv, (45, 193, 94), (130, 255, 255))
    blue = cv2.inRange(img_hsv, (104, 59, 94), (160, 255, 255))
    yellow = cv2.inRange(img_hsv, (10, 121, 150), (35, 255, 255))

    color = False

    if cv2.countNonZero(red):
        color_name = 'red'
        color = True
        mask = red
        set_effect(r=255, g=0, b=0, effect='blink')

    if cv2.countNonZero(green):
        color_name = 'green'
        color = True
        mask = green
        set_effect(r=0, g=255, b=0, effect='blink')

    if cv2.countNonZero(blue):
        color_name = 'blue'
        color = True
        mask = blue
        set_effect(r=0, g=0, b=255, effect='blink')

    if cv2.countNonZero(yellow):
        color_name = 'yellow'
        color = True
        mask = yellow
        set_effect(r=255, g=255, b=0, effect='fast_blink')

    if color:
        colors.append(color_name)
        # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, cnts[0], -1, (0,255,0), 2)
        print(color_name)


image_sub = rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback, queue_size=1)
im_sub = rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback_2, queue_size=1)
set_effect(r=0, g=255, b=0, effect='blink')
navigate_wait(z=1.5, frame_id='body', auto_arm=True)
rospy.sleep(5)
set_effect(r=255, g=255, b=0, effect='blink')
navigate_wait(z=1.5, frame_id='aruco_109')
navigate_wait(z=1.5, frame_id='aruco_111')
navigate_wait(z=1.5, frame_id='aruco_126')
navigate_wait(z=1.5, frame_id='aruco_125')
navigate_wait(z=1.5, frame_id='aruco_109')
navigate_wait(z=1.5, frame_id='aruco_112')
navigate_wait(z=1.5, frame_id='aruco_121')
navigate_wait(z=1.5, frame_id='aruco_151')
navigate_wait(z=1.5, frame_id='aruco_158')
navigate_wait(z=1.5, frame_id='aruco_149')
navigate_wait(z=1.5, x=5.5, y=3.5, frame_id='aruco_map')
rospy.sleep(3)
land()

with open('file.txt', 'w+') as file:
    file.write(colors)