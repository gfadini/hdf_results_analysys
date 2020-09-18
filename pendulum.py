from pinocchio.utils import *
import pinocchio

def createPendulum(nbJoint, length=1.0, mass=1.0):
    rmodel = pinocchio.Model()
    color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
    colorred = [1.0,0.0,0.0,1.0]
    radius = 0.1 * length
    prefix = ''
    jointId = 0
    jointPlacement = pinocchio.SE3.Identity()
    inertia = pinocchio.Inertia(mass,
                          np.matrix([0.0,0.0,length/2]).T,
                          mass/5*np.diagflat([ 1e-2,length**2,  1e-2 ]) )
    for i in range(nbJoint):
        istr = str(i)
        name               = prefix+"joint"+istr
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = rmodel.addJoint(jointId,pinocchio.JointModelRY(),jointPlacement,jointName)
        rmodel.appendBodyToJoint(jointId,inertia,pinocchio.SE3.Identity())
        jointPlacement     = pinocchio.SE3(eye(3),np.matrix([0.0,0.0,length]).T)
        if i >= 1:
            rmodel.addFrame(pinocchio.Frame('joint_' + str(i), jointId, i-1, pinocchio.SE3.Identity(), pinocchio.FrameType.JOINT))
    rmodel.addFrame( pinocchio.Frame('tip',jointId,0,jointPlacement,pinocchio.FrameType.OP_FRAME) )
    rmodel.upperPositionLimit = np.zeros(nbJoint)+2*np.pi
    rmodel.lowerPositionLimit = np.zeros(nbJoint)-2*np.pi
    rmodel.velocityLimit      = np.zeros(nbJoint)+5.0
    
    return rmodel

def createPendulumWrapper(nbJoint,initViewer=True):
    '''
    Returns a RobotWrapper with a N-pendulum inside.
    '''
    rmodel = createPendulum(nbJoint)
    rw = pinocchio.RobotWrapper(rmodel,visual_model=None,collision_model=None)
    if initViewer: rw.initViewer(loadModel=True) 
    return rw

if __name__ == "__main__":
    rw = createPendulumWrapper(3,True)
