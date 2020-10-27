import numpy as np
from correlations import remy_correlation, timing_belt_correlation, gearbox_efficiency

def update_model(model, motor_mass, n_gear, lambda_l):
    # calls the sequence depending on the model
    upscale_structure(model, lambda_l)
    modify_actuation(model, np.array(motor_mass), np.array(n_gear))
    add_motor_mass(model, np.array(motor_mass))

def upscale_structure(model, lambda_l):
    # MODIFY THE INERTIAL PARAMETERS, just upscaling all limbs
    for index in range(2, 2 + len(lambda_l) - 1):
        # scale all the masses
        model.inertias[index].mass = lambda_l[index-2]**3 * model.inertias[index].mass
        # scale center of mass
        model.inertias[index].lever = lambda_l[index-2] * model.inertias[index].lever
        # scale all inertia tensors
        model.inertias[index].inertia = lambda_l[index-2]**5 * model.inertias[index].inertia
        # MODIFY all LINK DIMENSIONS, linear placement of frames with respect to link before

    model.frames[-1].placement.translation = lambda_l[-1] * model.frames[-1].placement.translation

    for index in range(3, 3 + len(lambda_l) - 1):
            model.jointPlacements[index].translation = lambda_l[index-3] * model.jointPlacements[index].translation

    return model

def modify_actuation(model, motor_mass, n_gear):
    # assign the EFFORT LIMIT and the armature according to the model
    T0, omega0, I_m, K_m, T_mu = timing_belt_correlation(motor_mass, n_gear)
    model.effortLimit = T0 * n_gear
    model.armature = I_m
    model.rotorInertia = I_m
    # modify GEAR RATIOS
    model.rotorGearRatio = n_gear
    # modify MAX VELOCITY
    model.velocityLimit = omega0/n_gear
    # saving the dissipative parameters
    model.K_m = K_m
    model.T_mu = T_mu
    model.b = 1.341e-5*np.ones(len(motor_mass))
    # modify POSITION LIMITS
    # upper and lower LIMITS (to defaults)
    model.lowerPositionLimit = np.array(model.lowerPositionLimit)
    model.upperPositionLimit = np.array(model.upperPositionLimit)

def add_motor_mass(model, motor_mass):
    if isinstance(motor_mass, float) or isinstance(motor_mass, int):
        for i in range(2, len(model.inertias)):
            model.inertias[i].mass = model.inertias[i].mass + motor_mass
    elif isinstance(motor_mass, np.ndarray):
        for i, j in enumerate(range(2, len(model.inertias))):
            model.inertias[j].mass = model.inertias[j].mass + motor_mass[i]

def remove_motor_mass(model, motor_mass):
    add_motor_mass(model, - motor_mass)
