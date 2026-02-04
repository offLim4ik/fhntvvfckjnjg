import math
import rospy
from clover import srv

from std_srvs.srv import Trigger
import pigpio
from clover.srv import SetLEDEffect
import time

rospy.init_node('flight')


pi = pigpio.pi()
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
get_telemetry_1 = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)

navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)
pi.set_mode(13, pigpio.OUTPUT)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)  # define proxy to ROS-service

def sbros():

    pi.set_servo_pulsewidth(13, 1000)
    time.sleep(2)
    pi.set_servo_pulsewidth(13, 2000)

def lent(r=0,g=0,b=0,ef='fill'):
    set_effect(r=r, g=g, b=b, effect=ef)
    rospy.sleep(0.5)


def navigate_wait(x=0, y=0, z=0, yaw=math.nan, speed=0.5, frame_id='body', tolerance=0.2, auto_arm=False):
    res = navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    if not res.success:
        return res

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            return res
        rospy.sleep(0.2)


def sb(x=0, y=0, z=0.8, yaw=math.nan, speed=0.5, frame_id='aruco_139', tolerance=0.15, auto_arm=False):
    res = navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    tele = get_telemetry_1(frame_id='aruco_139')

    if not res.success:
        return res

    if math.isnan(tele.z):
        return None

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        print(math.sqrt(telem.x ** 2 + telem.y ** 2))
        if math.sqrt(telem.x ** 2 + telem.y ** 2) < tolerance:
            sbros()
            rospy.sleep(1)
            return res
        rospy.sleep(0.2)

print('Take off 1 meter')

lent(b=255,ef='blink')
navigate(z=1.5, frame_id='body', auto_arm=True)
rospy.sleep(5)

lent(r=255,g=255,ef='blink')

navigate(z=1.4,x=3,y=-0.65, frame_id='aruco_map', auto_arm=True, yaw=math.nan)
rospy.sleep(5)

navigate(z=1.6, x=5.5, y=0.5, frame_id='aruco_map', yaw=math.nan)
rospy.sleep(4)

navigate(z=1.6, x=5.5, y=0.5, frame_id='aruco_map', yaw=math.nan)
rospy.sleep(4)

navigate(x=2, frame_id='body')
rospy.sleep(3)

navigate_wait(z=1.4, x=3.5, y=4.65, frame_id='aruco_map', yaw=math.nan)
navigate_wait(z=1.4, x=2, y=4.65, frame_id='aruco_map', yaw=math.nan)

#navigate(y=2, frame_id='body')
#rospy.sleep(3)

navigate_wait(z=2, x=1, y=3, frame_id='aruco_map', yaw=math.nan)


lent(r=255, ef='blink')

sb()

lent(r=255,g=255,ef='blink')

navigate_wait(z=1, x=0.5, y=0.5, frame_id='aruco_map', yaw=math.nan)

lent(ef='rainbow')

land()
