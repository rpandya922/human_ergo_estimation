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
import sklearn.neighbors as neighbors
import sensor_msgs.msg as sensor_msgs

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False

def move():
    pose = geometry_msgs.msg.Pose()
    x = np.random.uniform(0.5, 1.1, 1)[0]
    y = 0.28
    z = np.random.uniform(0.3, 0.8, 1)[0]
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

global sb
global pos
pos = []
def callback(data):
    global pos
    pos = data.position
# x = [0.6, 1]
# y = 0.28
# z = [0.1, 0.8]
all_data = []
if __name__ == '__main__':
    rospy.on_shutdown(lambda: np.save("arm_joints_data", all_data))
    listener = tf.TransformListener()
    print "here"
    sb = rospy.Subscriber("kinmodel_state", sensor_msgs.JointState, callback)
    while not rospy.is_shutdown():
        if heardEnter():
            move()
            while not heardEnter():
                continue

            chosen = pos
            print "Recorded Position: " + str(chosen)
            (trans, rot) = listener.lookupTransform('/base', 'left_gripper', rospy.Time(0))
            print "Baxter Arm: " + str(trans)

            while not heardEnter():
                continue

            feasible = []
            while not heardEnter():
                print "Position" + str(pos)
                feasible.append(pos)
            all_data.append((trans, pos, feasible))