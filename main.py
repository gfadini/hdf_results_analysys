#!/usr/bin/python
import h5py
import matplotlib
import crocoddyl
import example_robot_data
import pinocchio
import sys
import time
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pendulum, modify_model

def readFromPath(filePath):
    results_archive = h5py.File(filePath, mode='r')
    return results_archive

def cleanName(filePath, namesToRemove):
    prettyName = filePath
    for uglyStuff in namesToRemove:
        prettyName = prettyName.replace(uglyStuff, '')
    return prettyName

def frame_position(archive, index, robot_model, frame_name):
    '''
    Returns the position of a frame for a given configuration
    '''
    robot_data = robot_model.createData()
    frame_id = robot_model.getFrameId(frame_name)
    x = []
    y = []
    z = []

    for i in archive['xs'][index]:
        pinocchio.updateFramePlacements(robot_model, robot_data)
        pinocchio.forwardKinematics(robot_model, robot_data, i[:robot_model.nq], i[robot_model.nq:])
        # changed for pinocchio array
        x.append(robot_data.oMf[frame_id].translation[0])
        y.append(robot_data.oMf[frame_id].translation[1])
        z.append(robot_data.oMf[frame_id].translation[2])
    return x, y, z

def plot_frame_trajectory(archive, index, robot_model, frame_names, target, image_folder = None, extension = 'pdf', trid = True, quiet = True):
    '''
    Plots a specific or multiple frame trajectory in time, 2D or 3D
    '''

    m_m = archive['motor_mass'][index]
    n_g = archive['n_gear'][index]
    l_l = archive['lambda_l'][index]
    modify_model.update_model(robot_model, m_m, n_g, l_l)

    fig_title = 'frame_trajectory'
    plt.figure(fig_title)
    initial_positions = np.array([])
    final_positions = np.array([])
    if trid:
        ax = plt.axes(projection = '3d')
        for frame_name in frame_names:
            x, y, z = frame_position(archive, index, robot_model, frame_name)
            ax.plot3D(x[1:], y[1:], z[1:])
            ax.scatter(x[1], y[1], z[1], color = 'black')
            ax.scatter(x[-1], y[-1], z[-1], marker = '*', color = 'green')
        ax.scatter(*target, marker = 'X', color = 'red')
        # Make axes limits
        xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
        XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
        ax.set_xlim3d(XYZlim)
        ax.set_ylim3d(XYZlim)
        ax.set_zlim3d(XYZlim)
        plt.legend(frame_names)
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            pass
    else:
        for frame_name in frame_names:
            x, y, z = frame_position(archive, index, robot_model, frame_name)
            plt.gca().plot(x[1:], z[1:])
            initial_positions = np.append(initial_positions, np.array([x[1], z[1]]))
            final_positions = np.append(final_positions, np.array([x[-1], z[-1]]))
        plt.gca().scatter(target[0], target[2], marker = 'x', color = 'red')
        plt.gca().scatter(initial_positions[0::2], initial_positions[1::2], color = 'black')
        plt.gca().scatter(final_positions[0::2], final_positions[1::2], color = 'blue')
        plt.gca().set_aspect('equal')
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        plt.gca().legend(frame_names[1:] + ['target', 'initial', 'final'], loc='center right', bbox_to_anchor=(1.6, 0.5), fancybox=True, shadow=True)
        plt.gca().plot(initial_positions[0::2], initial_positions[1::2], color = 'grey', alpha = 0.7)
        plt.gca().plot(final_positions[0::2], final_positions[1::2], color = 'grey', alpha = 0.7)

    plt.title('Frame Trajectory')
    plt.xlabel('x [m]')
    if trid:
        plt.ylabel('y [m]')
    else:
        plt.ylabel('z [m]')
    plt.grid(True)
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    if not quiet:
        plt.show()


def animateSolution(archive, index, robot_model, frameNames = None, target = None, dt = 1e3, saveAnimation=False):

    m_m = archive['motor_mass'][index]
    n_g = archive['n_gear'][index]
    l_l = archive['lambda_l'][index]
    modify_model.update_model(robot_model, m_m, n_g, l_l)

    xs = archive['xs'][index]
    us = archive['us'][index]

    anim = plt.figure()

    robot_data = robot_model.createData()
    if frameNames is None:
        frameNames = [frame.name for frame in robot_model.frames]
        frameNames.remove('universe')
    imgs = []

    scalingFactor = 2

    for i in np.concatenate((np.array(xs)[0:-1:scalingFactor], np.array([xs[-1]]*10))):
        X = []
        Z = []
        pinocchio.forwardKinematics(robot_model, robot_data, i[:robot_model.nq], i[robot_model.nq:])
        pinocchio.updateFramePlacements(robot_model, robot_data)
        for frame_name in frameNames:
            frame_id = robot_model.getFrameId(frame_name)
            X.append(robot_data.oMf[frame_id].translation[0])
            Z.append(robot_data.oMf[frame_id].translation[2])
        imgs.append(plt.plot(X,Z, color='grey', marker='o', linewidth=2, markerfacecolor='black'))

    import matplotlib.animation as animation
    im_ani = animation.ArtistAnimation(anim, imgs, interval=dt*1e3, repeat_delay=1000,
                                   blit=True)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.scatter(target[0], target[2], marker = 'x', color = 'red')
    # plt.plot([0,0],[-2,2.5], ls='--', color='blue')
    plt.title('Task animation')
    if saveAnimation:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1/dt/scalingFactor), metadata=dict(artist='G. Fadini'), bitrate=-1)
        im_ani.save('task_animation.mp4', writer=writer)
    plt.show()


def extract(archive, tag, index=0):
    '''
    Function used to handle the multidimensionality of the search space
    It extracts a specific component of the saved data
    it handles the exception in which the index is an integer
    '''
    tmp_array = []
    for i in archive[tag]:
        try:
            tmp_array.append(i[index])
        except:
            tmp_array.append(i)
    return np.array(tmp_array)

def simplePlot(x, y, image_folder, extension, figTitle='plot', color='blue', plotTitle = '', xlabel='x', ylabel='y', quiet = True, points = True):
    '''
    Generic scatter or line plot
    '''
    plt.figure(figTitle)
    if y is not None and x is not None:
        plt.scatter(x, y, color = color, s=2**2)
        plt.ylim(bottom=min(y))
    elif y is None and x is not None :
        if points:
            plt.plot(x, linestyle='none', marker = 'o', markersize = 2, color=color)
        else:
            plt.plot(x, color=color)
    else:
        raise Exception('Data is needed for the plot!')
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if image_folder is not None:
        plt.savefig(image_folder + figTitle + '.' + extension, format = extension)
    if not quiet:
        plt.show()

def cmapPlot(x, y, z, image_folder, extension, figTitle='scatter', cmap='BuPu_r', plotTitle = 'scatter', xlabel='x', ylabel='y', quiet = True):
    '''
    Default 2D colormap plot
    '''
    plt.figure(figTitle)
    scatter = plt.scatter(x, y, c = z, cmap = 'BuPu_r', s=3**2)
    plt.colorbar(scatter)
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=np.min(y))
    if image_folder is not None:
        plt.savefig(image_folder + figTitle + '.' + extension, format = extension)
    if not quiet:
        plt.show()

def plot_codesign_results(archive, selectedIndexes = None, image_folder = None, extension = 'pdf', quiet = False):

    '''
    Creates the saving directory if needed
    '''
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

    if selectedIndexes is None:
        # take all values
        selectedIndexes = list(True for _ in range(archive['motor_mass'].shape[0]))

    cost = np.log(extract(archive, 'cost')[selectedIndexes])
    error =  np.log(extract(archive, 'error')[selectedIndexes])

    for index in range(archive['motor_mass'].shape[1]):
        motorMass = extract(archive, 'motor_mass', index)[selectedIndexes]
        gearRatio = extract(archive, 'n_gear', index)[selectedIndexes]
        scaling = extract(archive, 'lambda_l', index)[selectedIndexes]

        label = str(index)

        simplePlot(motorMass, cost, image_folder, extension, 'cost_mass_' + label, 'blue', 'Cost and motor mass ' + label, '$m_m \; [$Kg$]$', '$\log(cost)$')
        simplePlot(gearRatio, cost, image_folder, extension, 'cost_trasmission_' + label, 'red', 'Cost and transmission ' + label, '$n \; [\;]$', '$\log(cost)$')
        simplePlot(scaling, cost, image_folder, extension, 'cost_scale_' + label, 'orange', 'Cost and scaling ' + label, '$\lambda_l \; [\;]$', '$\log(cost)$')

    simplePlot(cost, None, image_folder, extension, 'cost_evo', 'blue', 'Cost evolution during the optimization', 'Number of Iteration', '$\log(cost)$',)
    simplePlot(error, None, image_folder, extension, 'error_evo', 'red', 'Error evolution during the optimization', 'Number of Iteration', '$\log(error)$')
    
    if not quiet:
        plt.show()

    # cmapPlot(scaling, motorMass, cost, image_folder, extension, 'motor_lambda', 'BuPu_r', 'Motor mass and scaling', '$\\lambda_l$ [ ]', '$m_m$ [Kg]')
    # cmapPlot(motorMass, gearRatio, cost, image_folder, extension, 'motor_transmission', 'BuPu_r', 'Motor mass and gear ratio', '$m_m$ [Kg]', '$n$ [ ]')
    # cmapPlot(gearRatio, scaling, cost, image_folder, extension, 'transmission_lambda', 'BuPu_r', 'Gear ratio and scaling', '$n$ [ ]', '$\\lambda_l$ [ ]')

def selectSolution(archive, key, index):
    return np.array(archive[key][index, :])

def getChampionIndex(archive):
    return np.where(np.array(archive['cost']) == np.amin(np.array(archive['cost'])))[0][0]

def plotPower(archive,  index, image_folder, extension, figTitle = 'power_plot', plotTitle = 'Power components', xlabel = 'timestep [ ]', ylabel = 'Power [W]', quiet = True):
    pm = np.zeros(archive['us'][index].shape[0])
    for i, torque in enumerate(archive['us'][index]):
        pm[i] = np.sum(torque * archive['xs'][index][i][-archive['us'][index].shape[1]:])
    simplePlot(pm, None, image_folder, extension, figTitle, 'blue', plotTitle, xlabel, ylabel, points=False)
    simplePlot(archive['pf_cost'][index], None, None, None, figTitle, 'magenta', plotTitle, xlabel, ylabel, points=False)
    simplePlot(archive['pt_cost'][index], None, image_folder, extension, figTitle, 'red', plotTitle, xlabel, ylabel, points=False)
    simplePlot(pm + archive['pf_cost'][index] + archive['pt_cost'][index], None, image_folder, extension, figTitle, 'green', plotTitle, xlabel, ylabel, points=False)
    plt.legend(['$P_m$', '$P_f$', '$P_t$', '$P_{el}$'])
    if not quiet:
        plt.show()

def energy_stats(archive, index, robot_model, dt = 1e-3):
    '''
    Computes the energy required by the motion from two positions with zero initial and final velocities TODO add kinetic energy
    Compares the result with crocoddyl mechanical power consumptions and dissipation
    '''
    m_m = archive['motor_mass'][index]
    n_g = archive['n_gear'][index]
    l_l = archive['lambda_l'][index]
    modify_model.update_model(robot_model, m_m, n_g, l_l)

    xs = archive['xs'][index]
    us = archive['us'][index]

    pt = archive['pt_cost'][index]
    pf = archive['pf_cost'][index]
    pm = np.zeros(us.shape[0])
    for i, torque in enumerate(us):
        pm[i] = np.sum(torque * xs[i][-robot_model.nv:])

    pin_data = robot_model.createData()
    q0 = xs[0][:robot_model.nq]
    qf = xs[-1][:robot_model.nq]
    v0 = xs[0][robot_model.nq:]
    vf = xs[-1][robot_model.nq:]
    idealPotential = pinocchio.computePotentialEnergy(robot_model, pin_data, qf) - pinocchio.computePotentialEnergy(robot_model, pin_data, q0)
    idealKinetic = pinocchio.computeKineticEnergy(robot_model, pin_data, qf, vf) - pinocchio.computeKineticEnergy(robot_model, pin_data, q0, v0)
    ideal = idealPotential + idealKinetic
    mechanical = np.sum(pm)*dt
    print('Mechanical energy: {:0.3f} J, ideal: {:1.3f} J, error: {:2.2} %'.format(mechanical, ideal, (mechanical-ideal)/ideal * 1e2))
    print('Thermal dissipation: {:0.3f} J'.format(np.sum(pt)*dt))
    print('Friction dissipation: {:0.3f} J'.format(np.sum(pf)*dt))
    print('Total Energy needed: {:0.3f} J'.format(np.sum(pt)*dt + np.sum(pf)*dt + mechanical))

def plotSolution(archive, index, image_folder, extension, figTitle = 'solution', plotTitle = 'Solution', xlabel = 'time [s]', ylabel = '', quiet = True):
    '''
    Plots the ddp solution, xs, us
    '''
    xs, us = archive['xs'][index], archive['us'][index]

    if xs is None:
        usPlotIdx = 111
    elif us is None:
        xsPlotIdx = 111
    else:
        xsPlotIdx = 211
        usPlotIdx = 212
    dt = 1e-2
    N = len(us[:,0])
    time = np.arange(start=0, stop=dt*N + dt, step=dt)

    plt.figure(figTitle)

    # Plotting the state trajectories
    if xs is not None:
        plt.title('Solution trajectory')
        nx = len(xs[0])
        plt.subplot(xsPlotIdx)
        [plt.plot(time, xs[:,i], label="$x_{" + str(i) + '}$') for i in range(nx)]
        plt.legend()
    plt.grid(True)

    # Plotting the control commands
    if us is not None:
        nu = len(us[0])
        plt.subplot(usPlotIdx)
        [plt.plot(time[:N], us[:,i], label="$u_{" + str(i) + '}$') for i in range(nu)]
        plt.legend()
        plt.xlabel(xlabel)
    plt.grid(True)
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + figTitle + '.' + extension, format = extension)
    if not quiet:
        plt.show()

def generationMeans(archive, tag, gen = 100):
    means=[]
    N = len(archive[tag])
    for i in range(0,int(N/gen)):
        means.append(np.mean(archive[tag][i*int(gen):(i+1)*int(gen)]))
    return means

if __name__ == "__main__":
    # can specify multiple files as arguments
    # ideally a batch with wildcards
    files = sys.argv[1:]
    for filePath in files:
        prettyName = cleanName(filePath, ['results/', '.npz', '.hdf5'])
        results_archive = readFromPath(filePath)
        solved = np.array(results_archive['solved'])
        minValue = np.array(results_archive['champion_f'])
        minVect = np.array(results_archive['champion_x'])
        print('#'*20)
        print(prettyName)
        print('Solved {:.2f}% of the problems [{:0}]'.format(sum(solved==1)/len(solved)*1e2, len(solved)))
        print('Convergence evolution \n {:}'.format(generationMeans(results_archive, 'cost', 6e3)))
        print('Minimum {:} found at {:}'.format(minValue, minVect))

        plt.ioff()
        # plot_codesign_results(results_archive, 'plots/' + prettyName + '/', extension = 'png')
        bestIndex = getChampionIndex(results_archive)
        # plotSolution(results_archive, bestIndex, 'plots/' + prettyName + '/', extension = 'png', quiet = False)
        # plotPower(results_archive, bestIndex, 'plots/' + prettyName + '/', extension = 'png', quiet = False)
        plt.close('all')

        # plot_frame_trajectory(champion, 'tip')

        # POSTPROCESSING FILTER the search space, discard the solution off treshold
        try:
            acceptable_tolerance = np.array(results_archive['th_stop']) < results_archive['th_stop_tol']
        except:
            acceptable_tolerance = np.array(results_archive['th_stop']) < 1e-6
        acceptable_cost = np.log(np.array(results_archive['cost'])) < 0
        acceptable_error = np.array(results_archive['error']) < 1e-6
        acceptable_solutions = np.logical_and(acceptable_error, acceptable_tolerance, acceptable_cost)
        print('The acceptable solutions are {:2.2f}%'.format(100 * sum(acceptable_solutions)/results_archive['cost'].shape[0]))
        cost_acceptable = results_archive['cost'][acceptable_solutions]
        error_acceptable = results_archive['error'][acceptable_solutions]
        best_acceptable = np.where(np.array(results_archive['error']) == np.amin(error_acceptable))[0][0]
        plotSolution(results_archive, best_acceptable, None, None, quiet = False)
        plotPower(results_archive, best_acceptable, None, None, quiet = False)
        # plot_codesign_results(results_archive, acceptable_solutions, None, None, quiet = False)

        robot_model = pendulum.createPendulum(nbJoint=results_archive['motor_mass'].shape[1])
        frames=[frame.name for frame in robot_model.frames]
        animateSolution(results_archive, best_acceptable, robot_model.copy(), frameNames=frames, target=np.array([0,0,1]), dt = 1e-2)
        plot_frame_trajectory(results_archive, best_acceptable, robot_model.copy(), frame_names = frames, target=np.array([0,0,1]), trid = False, quiet = False)
        energy_stats(results_archive, best_acceptable, robot_model.copy(), dt=1e-2)

        plt.figure('weird_superposition')

        selected = (acceptable_solutions*range(len(acceptable_solutions)))[-1000:]
        selected_cost = np.array(results_archive['cost'])[selected]
        c_min = np.log(min(selected_cost))
        c_max = np.log(max(selected_cost))
        alpha = (c_max - np.log(selected_cost))/(c_max - c_min)
        for i, index in enumerate(selected):
                    m_m = results_archive['motor_mass'][index]
                    n_g = results_archive['n_gear'][index]
                    l_l = results_archive['lambda_l'][index]
                    dummy_model = robot_model.copy()
                    modify_model.update_model(dummy_model, m_m, n_g, l_l)
                    initial_positions=np.array([])
                    final_positions=np.array([])
                    for frame_name in ['tip']:
                            x, y, z = frame_position(results_archive, index, dummy_model, frame_name)
                            plt.gca().plot(x[1:], z[1:], color='blue', alpha = alpha[i]**6)
                            initial_positions = np.append(initial_positions, np.array([x[1], z[1]]))
                            final_positions = np.append(final_positions, np.array([x[-1], z[-1]]))
        plt.gca().set_aspect('equal')
        plt.show()

        plt.figure('histogram')
        _ = plt.hist(np.log(np.array(results_archive['cost'])[acceptable_solutions]), bins='auto')
        plt.show()