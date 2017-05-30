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

def preprocess(data):
    X = np.array([np.array(s[0]) for s in data])
    y = np.array([s[1][3:] for s in data])
    return X, y
def predict(pred, real):
    for i in range(len(pred)):
        p = pred[i]
        y = real[i]
        print min([np.linalg.norm(p - y), np.linalg.norm(-p - y)])

train = np.load("./hand_robot_data2.npy")
shuffle(train)
X, y = preprocess(train)
knn = neighbors.KNeighborsRegressor(n_neighbors=1)
knn.fit(X, y)

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
    listener = tf.TransformListener()
    pub = tf.TransformBroadcaster()
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        try:
            (transR, rotR) = listener.lookupTransform('/base', 'left_gripper', rospy.Time(0))
            rot = knn.predict([transR])[0]
            pub.sendTransform((0, 0, 0), rot, rospy.Time.now(), '/hand_prediction', 'left_gripper')

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        rate.sleep()