#!/usr/bin/env python
from __future__ import division  
import rospy
import math
import tf
import geometry_msgs.msg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.io
from sklearn.svm import SVC
from random import shuffle
import sys
import select
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import baxter_interface.gripper as gripper

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False

def move(x, y, z):
    pose = geometry_msgs.msg.Pose()

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = 0.5179492136281051
    pose.orientation.y = -0.4655689719259183
    pose.orientation.z = 0.5199804480287061
    pose.orientation.w = -0.4945649382535493

    group.set_pose_target(pose)
    plan = group.plan()
    group.go(wait=True)

print "============ Starting tutorial setup"
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('erg',
                anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
# group = moveit_commander.MoveGroupCommander("right_arm")
group = moveit_commander.MoveGroupCommander("left_arm")
display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory)
print "============ Reference frame: %s" % group.get_planning_frame()
print "============ Reference frame: %s" % group.get_end_effector_link()
group.set_goal_tolerance(0.05)

data = []

if __name__ == '__main__':
    rospy.on_shutdown(lambda: np.save("hand_robot_data2", data))
    listener = tf.TransformListener()
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        try:
            if heardEnter():
                # right arm values
                # x = np.random.uniform(0.7, 1.2, 1)[0]
                # y = np.random.uniform(-0.5, 0, 1)[0]
                # z = np.random.uniform(0.4, 0.8, 1)[0]
                # left arm values
                x = np.random.uniform(0.8, 1.1, 1)[0]
                y = np.random.uniform(-0.25, 0.7, 1)[0]
                z = np.random.uniform(0.0, 0.9, 1)[0]
                move(x, y, z)
                while not heardEnter():
                    continue
                (trans, rot) = listener.lookupTransform('left_gripper', '/hand', rospy.Time(0))
                (transR, rotR) = listener.lookupTransform('/base', 'left_gripper', rospy.Time(0))
                print 'Robot' + str(transR)
                print 'Position: ' + str(trans)
                print 'Orientation: ' + str(rot) + "\n"
                d = np.hstack((trans, rot))
                data.append((transR, d))
            # (trans, rot) = listener.lookupTransform('/base', 'left_gripper', rospy.Time(0))
            # print 'Position: ' + str(trans)
            # print 'Orientation: ' + str(rot) + "\n"
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        rate.sleep()