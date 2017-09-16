from __future__ import division
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../weight_learning')
import utils
from openravepy import *
import prpy
import numpy as np
active_modes =\
[[-0.22964738,0.42342584,0.08824346,-1.38410772,1.18876443,-0.29730261,-0.06190612],\
[-0.22964738,0.42342584,0.08824346,-1.38410772,1.18876443,-0.29730261,-0.06190612],\
[-0.67555171,0.43232173,0.2872307,-0.97430584,0.08594727,0.03039019,-0.23130636],\
[-0.83088649,0.59642484,0.39192865,-0.77610032,-0.0028982,0.21659383,-0.39098951],\
[-0.09302827,0.43295604,0.23444378,-1.44741989,1.22932189,-0.02045981,0.00253398]]
active_avg = np.mean(active_modes, axis=0)

passive_modes =\
[[-0.23565003,-0.26928066,0.16037327,-1.56364965,1.49692392,-0.4025099,-0.42278177],\
[-0.23565003,-0.26928066,0.16037327,-1.56364965,1.49692392,-0.4025099,-0.42278177],\
[-0.23565003,-0.26928066,0.16037327,-1.56364965,1.49692392,-0.4025099,-0.42278177],\
[-0.15748688,0.35850085,0.03547777,-1.77420555,1.11982494,-0.45716513,-0.15503821],\
[-0.15748688,0.35850085,0.03547777,-1.77420555,1.11982494,-0.45716513,-0.15503821]]
passive_avg = np.mean(passive_modes, axis=0)

OBJECT = 'handles'
env, human, robot, target, target_desc = utils.load_environment_file('../data/%s_cad_problem_def.npz' % OBJECT)
env.SetViewer('qtcoin')

newrobots = []
for ind in range(4):
    newrobot = RaveCreateRobot(env,human.GetXMLId())
    newrobot.Clone(human,0)
    for link in newrobot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(0)
    newrobots.append(newrobot)
for link in robot.GetLinks():
    for geom in link.GetGeometries():
        geom.SetTransparency(0)
with env:
    # human.SetDOFValues(passive_avg, human.GetActiveManipulator().GetArmIndices())
    human.SetDOFValues(passive_modes[0], human.GetActiveManipulator().GetArmIndices())
    inds = np.array([1, 2, 3, 4])
    for j,ind in enumerate(inds):
        newrobot = newrobots[j]
        env.Add(newrobot,True)
        newrobot.SetTransform(human.GetTransform())
        newrobot.SetDOFValues(passive_modes[ind], human.GetActiveManipulator().GetArmIndices())
env.UpdatePublishedBodies()
while True:
    continue