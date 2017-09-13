import time
import openravepy
if not __openravepy_build_doc__:
    from openravepy import *
    from numpy import *

def main(env,options):
    "Main example code."
    env.Load('robots/man1.zae')
    robot = env.GetRobots()[0]
    newrobots = []
    for ind in range(options.maxnumber):
        newrobot = RaveCreateRobot(env,robot.GetXMLId())
        newrobot.Clone(robot,0)
        for link in newrobot.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(options.transparency)
        newrobots.append(newrobot)
    for link in robot.GetLinks():
        for geom in link.GetGeometries():
            geom.SetTransparency(options.transparency)

    robot.SetActiveManipulator('rightarm')
    ikmodel = databases.inversekinematics.InverseKinematicsModel(robot, iktype=IkParameterization.Type.Transform6D)
    if not ikmodel.load():
        ikmodel.autogenerate()
    with env:
        # move the robot in a random collision-free position and call the IK
        while True:
            lower,upper = [v[ikmodel.manip.GetArmIndices()] for v in ikmodel.robot.GetDOFLimits()]
            robot.SetDOFValues(lower + 1,ikmodel.manip.GetArmIndices()) # set random values
            if not robot.CheckSelfCollision():
                print ikmodel.manip.GetTransform()
                solutions = ikmodel.manip.FindIKSolutions(ikmodel.manip.GetTransform(),IkFilterOptions.CheckEnvCollisions)
                if solutions is not None and len(solutions) > 0: # if found, then break
                    break
        print 'found %d solutions, rendering solutions:'%len(solutions)
        if len(solutions) < options.maxnumber:
            inds = range(len(solutions))
        else:
            inds = array(linspace(0,len(solutions)-1,options.maxnumber),int)
        for i,ind in enumerate(inds):
            print ind
            newrobot = newrobots[i]
            env.Add(newrobot,True)
            newrobot.SetTransform(robot.GetTransform())
            newrobot.SetDOFValues(solutions[ind],ikmodel.manip.GetArmIndices())
    env.UpdatePublishedBodies()
    while True:
        continue
    print('waiting...')
    time.sleep(20)
    # remove the robots
    for newrobot in newrobots:
        env.Remove(newrobot)
    del newrobots

from optparse import OptionParser
from openravepy.misc import OpenRAVEGlobalArguments

@openravepy.with_destroy
def run(args=None):
    """Command-line execution of the example.

    :param args: arguments for script to parse, if not specified will use sys.argv
    """
    parser = OptionParser(description='Shows how to generate a 6D inverse kinematics solver and use it for getting all solutions.')
    OpenRAVEGlobalArguments.addOptions(parser)
    parser.add_option('--scene',action="store",type='string',dest='scene',default='data/pr2test1.env.xml',
                      help='Scene file to load (default=%default)')
    parser.add_option('--transparency',action="store",type='float',dest='transparency',default=0.8,
                      help='Transparency for every robot (default=%default)')
    parser.add_option('--maxnumber',action="store",type='int',dest='maxnumber',default=10,
                      help='Max number of robots to render (default=%default)')
    parser.add_option('--manipname',action="store",type='string',dest='manipname',default=None,
                      help='name of manipulator to use (default=%default)')
    (options, leftargs) = parser.parse_args(args=args)
    OpenRAVEGlobalArguments.parseAndCreateThreadedUser(options,main,defaultviewer=True)

if __name__ == "__main__":
    run()