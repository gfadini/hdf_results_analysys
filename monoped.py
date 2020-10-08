from pinocchio.utils import *
import pinocchio
import example_robot_data

def createMonoped(nbJoint, linkLength=1.0, floatingMass=1.0, linkMass=1.0):
    rmodel = pinocchio.Model()
    prefix = ''
    baseInertia = pinocchio.Inertia(floatingMass,
                          np.matrix([0.0, 0.0, 0.0]).T,
                          np.diagflat([1e-6, 1e-6, 1e-6]))
    linkInertia = pinocchio.Inertia(linkMass,
                          np.matrix([0.0, 0.0, linkLength/2]).T,
                          linkMass/5*np.diagflat([1e-2, linkLength**2, 1e-2]))
    # PRISMATIC JOINT
    jointId = 0
    jointPlacement = pinocchio.SE3.Identity()
    jointName,bodyName = ["prismatic_joint", "mass"]
    jointId = rmodel.addJoint(jointId, pinocchio.JointModelPZ(), jointPlacement, jointName)
    rmodel.addFrame(pinocchio.Frame('base', jointId, 0, jointPlacement, pinocchio.FrameType.OP_FRAME))
    rmodel.appendBodyToJoint(jointId, baseInertia, pinocchio.SE3.Identity())

    # REVOLUTE JOINTS
    for i in range(1, nbJoint + 1):
        jointName,bodyName = ["revolute_joint_" + str(i), "link_" + str(i)]
        jointId = rmodel.addJoint(jointId,pinocchio.JointModelRY(),jointPlacement,jointName)
        rmodel.appendBodyToJoint(jointId,linkInertia, pinocchio.SE3.Identity())
        rmodel.addFrame(pinocchio.Frame('revolute_' + str(i), jointId, i-1, pinocchio.SE3.Identity(), pinocchio.FrameType.JOINT))
        jointPlacement = pinocchio.SE3(eye(3), np.matrix([0.0, 0.0, linkLength]).T)

    rmodel.addFrame( pinocchio.Frame('foot', jointId, 0, jointPlacement, pinocchio.FrameType.OP_FRAME))
    rmodel.upperPositionLimit = np.concatenate((np.array([100]),  2 * np.pi * np.ones(nbJoint)), axis=0)
    rmodel.lowerPositionLimit = np.concatenate((np.array([0.0]), -2 * np.pi * np.ones(nbJoint)), axis=0)
    rmodel.velocityLimit      = np.concatenate((np.array([100]),  5 * np.ones(nbJoint)), axis=0)

    return rmodel

def createMonopedWrapper(nbJoint, linkLength=1.0, floatingMass=1.0, linkMass=1.0):
    '''
    Returns a RobotWrapper with a monoped inside.
    '''
    rmodel = createMonoped(nbJoint,linkLength, floatingMass, linkMass)
    rw = pinocchio.RobotWrapper(rmodel,visual_model=None,collision_model=None)
    return rw

def createSoloTB():
    rmodel = pinocchio.Model()
    prefix = ''
    nbJoint = 2

    floatingMass = 1.48538/4 # one quater of total base mass
    linkLength = 0.16
    firstLinkMass = 0.148538
    secondLinkMass = 0.0376361
    firstLinkCOMLength = 0.078707
    secondLinkCOMLength = 0.102249

    baseInertia = pinocchio.Inertia(floatingMass,
                            np.matrix([0.0, 0.0, 0.0]).T,
                            np.diagflat([1e-6, 1e-6, 1e-6]))
    firstLinkInertia = pinocchio.Inertia(firstLinkMass,
                            np.matrix([0.0, 0.0, firstLinkCOMLength]).T,
                            firstLinkMass/3*np.diagflat([1e-6, firstLinkCOMLength**2, 1e-6]))
    secondLinkInertia = pinocchio.Inertia(secondLinkMass,
                            np.matrix([0.0, 0.0, secondLinkCOMLength]).T,
                            secondLinkMass/3*np.diagflat([1e-6, secondLinkCOMLength**2, 1e-6]))

    # PRISMATIC JOINT
    jointId = 0
    jointPlacement = pinocchio.SE3.Identity()
    jointName,bodyName = ["prismatic_joint", "mass"]
    jointId = rmodel.addJoint(jointId, pinocchio.JointModelPZ(), jointPlacement, jointName)
    rmodel.addFrame(pinocchio.Frame('base', jointId, 0, jointPlacement, pinocchio.FrameType.OP_FRAME))
    rmodel.appendBodyToJoint(jointId, baseInertia, pinocchio.SE3.Identity())

    # REVOLUTE JOINTS
    for i in range(1, nbJoint + 1):
        jointName,bodyName = ["revolute_joint_" + str(i), "link_" + str(i)]
        jointId = rmodel.addJoint(jointId,pinocchio.JointModelRY(),jointPlacement,jointName)
        if i != nbJoint:
            rmodel.appendBodyToJoint(jointId,firstLinkInertia, pinocchio.SE3.Identity())
        else:
            rmodel.appendBodyToJoint(jointId,secondLinkInertia, pinocchio.SE3.Identity())
        rmodel.addFrame(pinocchio.Frame('revolute_' + str(i), jointId, i-1, pinocchio.SE3.Identity(), pinocchio.FrameType.JOINT))
        jointPlacement = pinocchio.SE3(eye(3), np.matrix([0.0, 0.0, linkLength]).T)

    rmodel.addFrame( pinocchio.Frame('foot', jointId, 0, jointPlacement, pinocchio.FrameType.OP_FRAME))
    rmodel.upperPositionLimit = np.concatenate((np.array([100]),  2 * np.pi * np.ones(nbJoint)), axis=0)
    rmodel.lowerPositionLimit = np.concatenate((np.array([0.0]), -2 * np.pi * np.ones(nbJoint)), axis=0)
    rmodel.velocityLimit      = np.concatenate((np.array([100]),  5 * np.ones(nbJoint)), axis=0)

    return rmodel

def createSoloTBWrapper():
    '''
    Returns a RobotWrapper with a monoped inside.
    '''
    rmodel = createSoloTB()
    rw = pinocchio.RobotWrapper(rmodel,visual_model=None,collision_model=None)
    return rw

def loadSoloLeg(solo8 = True):
    if solo8:
        URDF_FILENAME = "solo.urdf"
        legMaxId = 4
    else:
        URDF_FILENAME = "solo12.urdf"
        legMaxId = 5
    SRDF_FILENAME = "solo.srdf"
    SRDF_SUBPATH = "/solo_description/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/solo_description/robots/" + URDF_FILENAME
    modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

    robot = example_robot_data.loadSolo(solo8)
    m1 = robot.model
    m2 = pinocchio.Model()
    for index, [j, M, name, parent, Y] in enumerate(zip(m1.joints, m1.jointPlacements, m1.names, m1.parents, m1.inertias)):
        if j.id < legMaxId:
            jointType = j.shortname()
            if jointType == 'JointModelFreeFlyer':
                # for the freeflyer we just take reduced mass and zero inertia
                jointType = 'JointModelPZ'
                Y.mass = Y.mass/4
                Y.inertia = np.diag(1e-6*np.ones(3))
                M = pinocchio.SE3.Identity()

            if index == 2:
                # start with the prismatic joint on the sliding guide
                M.translation = np.zeros(3)

            # 2D model, flatten y axis
            vector2d = M.translation
            vector2d[1] = 0.0
            M.translation = vector2d

            jid = m2.addJoint(parent, getattr(pinocchio, jointType)(), M, name)
            assert (jid == j.id)
            m2.appendBodyToJoint(jid, Y, pinocchio.SE3.Identity())

    for f in m1.frames:
        if f.parent < legMaxId:
            m2.addFrame(f)

    g2 = pinocchio.GeometryModel()
    for g in robot.visual_model.geometryObjects:
        if g.parentJoint < legMaxId:
            g2.addGeometryObject(g)

    robot.model = m2
    robot.data = m2.createData()
    robot.visual_model = g2
    robot.visual_data = pinocchio.GeometryData(g2)

    # Load SRDF file
    #q0 = example_robot_data.readParamsFromSrdf(robot.model, modelPath + SRDF_SUBPATH, False, False, 'standing')
    robot.q0 = np.zeros(m2.nq+1)

    assert ((m2.rotorInertia[:m2.joints[1].nq] == 0.).all())
    return robot