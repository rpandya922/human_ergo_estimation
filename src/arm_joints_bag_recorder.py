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

rospy.init_node('recorder', anonymous=True)
global sb
global pos
pos = []
def callback(data):
    global pos
    pos = data.position

all_data = []
if __name__ == '__main__':
    rospy.on_shutdown(lambda: np.save("arm_joints_bag_data", all_data))
    listener = tf.TransformListener()
    print "here"
    sb = rospy.Subscriber("kinmodel_state", sensor_msgs.JointState, callback)
    while not rospy.is_shutdown():
        if heardEnter():
            print "Recorded Position: " + str(pos)
            (trans, rot) = listener.lookupTransform('/base', 'left_gripper', rospy.Time(0))
            print "Baxter Arm: " + str(trans)
            all_data.append((trans, pos))