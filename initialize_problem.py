
import os
import sys
import crocoddyl
import pinocchio
pinocchio.switchToNumpyArray()
import numpy as np
import time
from power_costs import CostModelJointFriction, CostModelJouleDissipation
import modify_model

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ


def create_pendulum_problem(robot_model, conf, increase_position_weight = 0):

    # setting gravity forces
    g = - 9.81
    robot_model.gravity.linear = np.matrix([0, 0, g]).T

    # Create a cost model per the running and terminal action model
    state = crocoddyl.StateMultibody(robot_model)
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    actuation = crocoddyl.ActuationModelFull(state)

    Pref = crocoddyl.FrameTranslation(robot_model.getFrameId("tip"),
                                   np.array(conf.target))
    Vref = crocoddyl.FrameMotion(robot_model.getFrameId("tip"), pinocchio.Motion(np.zeros(6)))
    goalTrackingCost = crocoddyl.CostModelFrameTranslation(state, Pref)
    goalFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref)

    power_act =  crocoddyl.ActivationModelQuad(robot_model.nv)

    # adding reference to the model
    # MAKES ALL THE PARAMETERS ACCESSIBLE IN THE COST FUNCTION WITH THE ABSTRACTION
    state.robot_model = robot_model
    joint_friction = CostModelJointFriction(state, power_act, actuation.nu)
    joule_dissipation = CostModelJouleDissipation(state, power_act, actuation.nu)

    # Soft bound on effort limit
    maxTorque = robot_model.effortLimit[-actuation.nu:]
    torqueBounds = crocoddyl.ActivationBounds(-maxTorque, maxTorque, 1.0)
    torqueAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(torqueBounds, np.ones(actuation.nu))
    torqueCost = crocoddyl.CostModelControl(state, torqueAct, actuation.nu)

    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("joule_dissipation", joule_dissipation, conf.weight_power_losses_cost_correction)
    runningCostModel.addCost("joint_friction", joint_friction, conf.weight_friction_power_cost)
    runningCostModel.addCost("effort_limit", torqueCost, conf.weight_gripperPose/1e0)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 2**increase_position_weight * conf.weight_gripperPose)
    terminalCostModel.addCost("gripperVelocity", goalFinalVelocity, 2**increase_position_weight * conf.weight_finalVelocity)

    T = conf.T
    dt = conf.dt
    q0 = np.array([np.pi] + [0]* (robot_model.nq-1))
    x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

    runningModel.differential.armature = np.array(robot_model.armature)
    terminalModel.differential.armature = np.array(robot_model.armature)

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    ddp = crocoddyl.SolverFDDP(problem)
    ddp.robot_model = robot_model
    ddp.setCallbacks([crocoddyl.CallbackLogger(),])

    return ddp, state
