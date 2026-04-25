from std_srvs.srv import Trigger
import rospy
import math
from clover.srv import SetLEDEffect
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

rospy.init_node('flight')
bridge = CvBridge()

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
land = rospy.ServiceProxy('land', Trigger)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect, persistent=True)

mask_pub = rospy.Publisher('/Kalashnikov', Image, queue_size=1)


def navigate_wait(x=0, y=0, z=0, speed=0.5, frame_id='body', auto_arm=False):
    res = navigate(x=x, y=y, z=z, yaw=float('nan'), speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    if not res.success:
        raise Exception(res.message)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < 0.2:
            return
        rospy.sleep(0.2)


def land_wait():
    land()
    while get_telemetry().armed:
        rospy.sleep(0.2)


last_color = None


def image_callback(msg):
    global last_color

    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red = cv2.inRange(img_hsv, (0, 206, 215), (180, 255, 255))

    color = False
    if cv2.countNonZero(red):
        color_name = 'red'
        color = True
        mask = red
        set_effect(effect='blink',r=255,g=0,b=0)
    if color:
        cnts = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.putText(img,color_name,(10,30),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
        cv2.drawContours(img,cnts[0],-1,(0,255,0),2)

        if last_color != color_name:
            rospy.loginfo(color_name)
            last_color = color_name

    mask_pub.publish(bridge.cv2_to_imgmsg(img,'bgr8'))
im_sub = rospy.Subscriber('main_camera/image_raw_throttled',Image,image_callback,queue_size=1)
def flight():
    navigate_wait(z=1, frame_id='body', auto_arm=True)
    rospy.sleep(5)
    set_effect(effect='fill', r=0, g=255, b=0)
    navigate_wait(x=0.5, y=1.5, z=0.5, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)
    image_callback
    rospy.sleep(5)
    navigate_wait(x=2.5, y=1.5, z=0.5, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)
    image_callback
    rospy.sleep(5)
    navigate_wait(x=0, y=0, z=0.5, frame_id='aruco_map', speed=0.5)
    rospy.sleep(5)
    land_wait()
